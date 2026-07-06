from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.adaptive_weights import AdaptiveWeightSettings, apply_regime_weights
from execution.news import NewsAssessment
from core.shadow import ShadowFeatureContext, ShadowScoreBreakdown, score_shadow_context
from smc.liquidity import LiquidityContext
from smc.regime import RegimeState
from smc.trigger import TriggerContext


@dataclass(frozen=True)
class ScoreInputVector:
    """
    Structured score inputs for institutional pipeline.
    Returns vector, NOT aggregated score.
    
    Components:
    - regime_score: 0.0-1.0 (regime alignment with direction)
    - structure_alignment: 0.0-1.0 (BOS/CHoCH quality)
    - liquidity_quality: 0.0-1.0 (sweep quality)
    - trigger_confidence: 0.0-1.0 (trigger strength)
    """
    regime_score: float
    structure_alignment: float
    liquidity_quality: float
    trigger_confidence: float
    
    # Optional components (can be added later)
    htf_alignment: float = 0.0
    fvg_strength: float = 0.0
    ob_strength: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        return {
            "regime_score": self.regime_score,
            "structure_alignment": self.structure_alignment,
            "liquidity_quality": self.liquidity_quality,
            "trigger_confidence": self.trigger_confidence,
            "htf_alignment": self.htf_alignment,
            "fvg_strength": self.fvg_strength,
            "ob_strength": self.ob_strength,
        }
    
    @property
    def vector(self) -> list[float]:
        """Raw vector for expectancy calculation."""
        return [
            self.regime_score,
            self.structure_alignment,
            self.liquidity_quality,
            self.trigger_confidence,
            self.htf_alignment,
            self.fvg_strength,
            self.ob_strength,
        ]
    
    @property
    def total(self) -> float:
        """Legacy total for backward compatibility."""
        return sum(self.vector) / len(self.vector) * 100


# Regime-based activation multipliers
# Each SMC component has different weight per regime
REGIME_COMPONENT_WEIGHTS = {
    # (structure, liquidity, fvg, ob, trigger, mtf)
    "trend_strong": {
        "structure": 1.0,
        "liquidity": 0.7,
        "fvg": 1.0,
        "ob": 1.0,
        "trigger": 1.0,
        "mtf": 1.0,
    },
    "trend_weak": {
        "structure": 0.8,
        "liquidity": 0.6,
        "fvg": 0.8,
        "ob": 0.9,
        "trigger": 0.8,
        "mtf": 0.9,
    },
    "range_tight": {
        "structure": 0.3,
        "liquidity": 0.5,
        "fvg": 0.3,
        "ob": 0.8,
        "trigger": 0.3,
        "mtf": 0.2,
    },
    "range_wide": {
        "structure": 0.2,
        "liquidity": 0.9,
        "fvg": 0.2,
        "ob": 0.3,
        "trigger": 0.2,
        "mtf": 0.1,
    },
    "expansion": {
        "structure": 0.4,
        "liquidity": 1.0,
        "fvg": 0.5,
        "ob": 0.4,
        "trigger": 0.4,
        "mtf": 0.3,
    },
    "transition": {
        "structure": 0.0,
        "liquidity": 0.0,
        "fvg": 0.0,
        "ob": 0.0,
        "trigger": 0.0,
        "mtf": 0.0,
    },
}


def get_regime_multipliers(regime: str) -> dict[str, float]:
    """Get component weight multipliers for given regime."""
    return REGIME_COMPONENT_WEIGHTS.get(regime, REGIME_COMPONENT_WEIGHTS.get("transition", {}))


def apply_regime_weights_to_breakdown(
    breakdown: "ScoreBreakdown",
    regime: str,
) -> "ScoreBreakdown":
    """Apply regime-adjusted weights to score breakdown."""
    multipliers = get_regime_multipliers(regime)
    
    # Apply multipliers to relevant components
    new_htf = int(breakdown.htf_alignment * multipliers.get("mtf", 0.0))
    new_trigger = int(breakdown.trigger_confirmation * multipliers.get("trigger", 0.0))
    new_liq = int(breakdown.liquidity_displacement * multipliers.get("liquidity", 0.0))
    new_fvg = int(breakdown.fvg_alignment * multipliers.get("fvg", 0.0))
    new_ob = int(breakdown.order_block_alignment * multipliers.get("ob", 0.0))
    
    # Recalculate total
    new_total = (
        new_htf + breakdown.regime_alignment + new_trigger + new_liq +
        breakdown.premium_discount + breakdown.news_filter + breakdown.session_timing +
        new_fvg + new_ob + breakdown.mitigation_alignment + breakdown.smt_alignment +
        breakdown.shadow_bonus
    )
    
    return ScoreBreakdown(
        htf_alignment=new_htf,
        regime_alignment=breakdown.regime_alignment,
        trigger_confirmation=new_trigger,
        liquidity_displacement=new_liq,
        premium_discount=breakdown.premium_discount,
        news_filter=breakdown.news_filter,
        session_timing=breakdown.session_timing,
        fvg_alignment=new_fvg,
        order_block_alignment=new_ob,
        mitigation_alignment=breakdown.mitigation_alignment,
        smt_alignment=breakdown.smt_alignment,
        shadow_bonus=breakdown.shadow_bonus,
        total=new_total,
        structure_quality=breakdown.structure_quality,
    )


@dataclass(frozen=True)
class ScoreBreakdown:
    htf_alignment: int
    regime_alignment: int
    trigger_confirmation: int
    liquidity_displacement: int
    premium_discount: int
    news_filter: int
    session_timing: int
    fvg_alignment: int
    order_block_alignment: int
    mitigation_alignment: int
    smt_alignment: int
    shadow_bonus: int
    total: int
    structure_quality: int = 0

    def contribution_dict(self) -> dict[str, int]:
        return {
            "htf": int(self.htf_alignment),
            "regime": int(self.regime_alignment),
            "trigger": int(self.trigger_confirmation),
            "liquidity": int(self.liquidity_displacement),
            "pd": int(self.premium_discount),
            "session": int(self.session_timing),
            "news": int(self.news_filter),
            "shadow_fvg": int(self.fvg_alignment),
            "shadow_ob": int(self.order_block_alignment),
            "shadow_mitigation": int(self.mitigation_alignment),
            "shadow_smt": int(self.smt_alignment),
            "structure_quality": int(self.structure_quality),
        }


def _side_direction(side: str) -> str:
    return "bullish" if side.upper() == "BUY" else "bearish"


def _score_htf_alignment(side: str, htf_bias: str) -> int:
    side_u = side.upper()
    bias_u = htf_bias.upper()

    if side_u == "BUY" and bias_u == "BULLISH":
        return 20
    if side_u == "SELL" and bias_u == "BEARISH":
        return 20
    if bias_u == "NEUTRAL":
        return 10
    return 0


def _score_regime_alignment(side: str, regime: RegimeState) -> int:
    side_dir = _side_direction(side)
    regime_dir = regime.direction.lower()
    regime_label = regime.label.lower()

    if regime_dir == side_dir:
        base_map = {
            "trend": 15,
            "expansion": 13,
            "range": 9,
            "contraction": 6,
            "neutral": 8,
        }
    elif regime_dir == "neutral":
        base_map = {
            "trend": 8,
            "expansion": 7,
            "range": 6,
            "contraction": 5,
            "neutral": 6,
        }
    else:
        base_map = {
            "trend": 0,
            "expansion": 2,
            "range": 4,
            "contraction": 3,
            "neutral": 1,
        }

    base = base_map.get(regime_label, 5)
    boost = int(round(max(0.0, min(1.0, regime.confidence)) * 2))
    return min(15, base + boost)


def _score_trigger_confirmation(side: str, trigger: TriggerContext) -> int:
    """Score trigger confirmation (kept for backward compatibility)."""
    side_dir = _side_direction(side)
    trigger_dir = trigger.direction.lower()
    
    if trigger_dir == side_dir:
        return min(15, 7 + trigger.strength)
    
    if trigger_dir == "neutral":
        base = 3 + min(4, trigger.strength // 4)
        if trigger.structure_event is not None:
            base += 2
        if trigger.liquidity.sweep or trigger.liquidity.displacement:
            base += 2
        return min(15, base)
    
    return 0


def trigger_to_confidence(trigger: TriggerContext, side: str) -> float:
    """
    Convert trigger to probabilistic confidence (0-1).
    
    Instead of binary, returns confidence level.
    """
    side_dir = _side_direction(side)
    trigger_dir = trigger.direction.lower()
    
    if trigger_dir == side_dir:
        # Same direction - high confidence
        return min(1.0, 0.5 + (trigger.strength / 30.0))
    elif trigger_dir == "neutral":
        # Neutral - medium confidence
        return 0.3 + (trigger.strength / 50.0)
    else:
        # Opposing - no confidence
        return 0.0


def _score_liquidity_displacement(side: str, liquidity: LiquidityContext) -> int:
    score = 0
    side_dir = _side_direction(side)

    if liquidity.sweep and liquidity.sweep_direction == side_dir:
        score += 8
    elif liquidity.sweep:
        score += 3

    if liquidity.displacement and liquidity.displacement_direction == side_dir:
        score += 10
    elif liquidity.displacement:
        score += 4

    if side_dir == "bullish" and liquidity.equal_lows:
        score += 2
    elif side_dir == "bearish" and liquidity.equal_highs:
        score += 2
    elif liquidity.equal_highs or liquidity.equal_lows:
        score += 2

    return min(20, score)


def _score_premium_discount(side: str, zone: str) -> int:
    side_u = side.upper()
    zone_u = zone.upper()

    if side_u == "BUY" and zone_u == "DISCOUNT":
        return 15
    if side_u == "SELL" and zone_u == "PREMIUM":
        return 15
    if zone_u == "EQUILIBRIUM":
        return 8
    return 0


def score_session_timing(pair: str, signal_time: datetime) -> int:
    hour = signal_time.hour
    weekday = signal_time.weekday()
    quote = pair.upper().replace("/", "")[3:6]

    if weekday >= 5:
        return 0

    if 12 <= hour <= 15:
        return 12
    if 7 <= hour <= 11:
        return 10
    if 16 <= hour <= 18:
        return 8

    if quote == "JPY" and 0 <= hour <= 3:
        return 7

    if 4 <= hour <= 6:
        return 5

    if 19 <= hour <= 21:
        return 3

    return 2


def calculate_score(
    pair: str,
    side: str,
    htf_bias: str,
    zone: str,
    liquidity: LiquidityContext,
    regime: RegimeState,
    trigger: TriggerContext,
    news: NewsAssessment,
    signal_time: datetime,
    shadow: ShadowFeatureContext | None = None,
    adaptive_weights: AdaptiveWeightSettings | None = None,
    structure_quality_bonus: int = 0,
) -> ScoreBreakdown:
    breakdown, _ = calculate_score_details(
        pair=pair,
        side=side,
        htf_bias=htf_bias,
        zone=zone,
        liquidity=liquidity,
        regime=regime,
        trigger=trigger,
        news=news,
        signal_time=signal_time,
        shadow=shadow,
        adaptive_weights=adaptive_weights,
        structure_quality_bonus=structure_quality_bonus,
    )
    return breakdown


def calculate_score_details(
    pair: str,
    side: str,
    htf_bias: str,
    zone: str,
    liquidity: LiquidityContext,
    regime: RegimeState,
    trigger: TriggerContext,
    news: NewsAssessment,
    signal_time: datetime,
    shadow: ShadowFeatureContext | None = None,
    adaptive_weights: AdaptiveWeightSettings | None = None,
    structure_quality_bonus: int = 0,
) -> tuple[ScoreBreakdown, dict[str, object]]:
    htf = _score_htf_alignment(side, htf_bias)
    regime_score = _score_regime_alignment(side, regime)
    trigger_score = _score_trigger_confirmation(side, trigger)
    liq = _score_liquidity_displacement(side, liquidity)
    zone_score = _score_premium_discount(side, zone)
    news_score = max(0, min(5, news.score // 3))
    session = score_session_timing(pair, signal_time)
    shadow_scores: ShadowScoreBreakdown
    if shadow is None:
        shadow_scores = ShadowScoreBreakdown(
            fvg_alignment=0,
            order_block_alignment=0,
            mitigation_alignment=0,
            smt_alignment=0,
            total=0,
        )
    else:
        shadow_scores = score_shadow_context(shadow)
    structure_quality = max(0, min(20, int(structure_quality_bonus)))

    raw_total = max(
        0,
        min(
            100,
            htf
            + regime_score
            + trigger_score
            + liq
            + zone_score
            + news_score
            + session
            + shadow_scores.total
            + structure_quality,
        ),
    )
    raw_components = {
        "htf": int(htf),
        "regime": int(regime_score),
        "trigger": int(trigger_score),
        "liquidity": int(liq),
        "pd": int(zone_score),
        "session": int(session),
        "news": int(news_score),
        "shadow_fvg": int(shadow_scores.fvg_alignment),
        "shadow_ob": int(shadow_scores.order_block_alignment),
        "shadow_mitigation": int(shadow_scores.mitigation_alignment),
        "shadow_smt": int(shadow_scores.smt_alignment),
        "structure_quality": int(structure_quality),
    }
    weighted_components, weighted_total, adaptive_meta = apply_regime_weights(
        raw_components,
        regime_label=regime.label,
        settings=adaptive_weights,
    )
    weighted_components = dict(weighted_components)
    weighted_components["structure_quality"] = int(structure_quality)
    weighted_total = max(0, min(100, int(weighted_total) + structure_quality))

    if adaptive_meta.get("enabled", False):
        score = ScoreBreakdown(
            htf_alignment=weighted_components["htf"],
            regime_alignment=weighted_components["regime"],
            trigger_confirmation=weighted_components["trigger"],
            liquidity_displacement=weighted_components["liquidity"],
            premium_discount=weighted_components["pd"],
            news_filter=weighted_components["news"],
            session_timing=weighted_components["session"],
            fvg_alignment=weighted_components["shadow_fvg"],
            order_block_alignment=weighted_components["shadow_ob"],
            mitigation_alignment=weighted_components["shadow_mitigation"],
            smt_alignment=weighted_components["shadow_smt"],
            shadow_bonus=(
                weighted_components["shadow_fvg"]
                + weighted_components["shadow_ob"]
                + weighted_components["shadow_mitigation"]
                + weighted_components["shadow_smt"]
            ),
            total=weighted_total,
            structure_quality=structure_quality,
        )
    else:
        score = ScoreBreakdown(
            htf_alignment=htf,
            regime_alignment=regime_score,
            trigger_confirmation=trigger_score,
            liquidity_displacement=liq,
            premium_discount=zone_score,
            news_filter=news_score,
            session_timing=session,
            fvg_alignment=shadow_scores.fvg_alignment,
            order_block_alignment=shadow_scores.order_block_alignment,
            mitigation_alignment=shadow_scores.mitigation_alignment,
            smt_alignment=shadow_scores.smt_alignment,
            shadow_bonus=shadow_scores.total,
            total=raw_total,
            structure_quality=structure_quality,
        )

    return score, {
        "raw_components": raw_components,
        "weighted_components": weighted_components,
        "adaptive_weights": adaptive_meta,
        "raw_total": raw_total,
        "weighted_total": weighted_total,
    }


def calculate_score_regime_aware(
    pair: str,
    side: str,
    htf_bias: str,
    zone: str,
    liquidity: LiquidityContext,
    regime: RegimeState,
    trigger: TriggerContext,
    news: NewsAssessment,
    signal_time: datetime,
    shadow: ShadowFeatureContext | None = None,
    adaptive_weights: AdaptiveWeightSettings | None = None,
) -> tuple[ScoreBreakdown, ScoreBreakdown]:
    """
    Calculate score with regime-aware weighting.
    
    Returns tuple: (raw_breakdown, regime_adjusted_breakdown)
    
    Raw: original scoring without regime adjustments
    Adjusted: scoring with regime component weights applied
    """
    # Get base scoring
    raw_breakdown, details = calculate_score_details(
        pair=pair,
        side=side,
        htf_bias=htf_bias,
        zone=zone,
        liquidity=liquidity,
        regime=regime,
        trigger=trigger,
        news=news,
        signal_time=signal_time,
        shadow=shadow,
        adaptive_weights=adaptive_weights,
    )
    
    # Get regime label
    regime_label = regime.label.lower() if regime else "neutral"
    if regime_label not in REGIME_COMPONENT_WEIGHTS:
        regime_label = "neutral"
    
    # Apply regime weights
    adjusted_breakdown = apply_regime_weights_to_breakdown(raw_breakdown, regime_label)
    
    return raw_breakdown, adjusted_breakdown


def build_score_vector(
    regime: RegimeState,
    liquidity: LiquidityContext,
    trigger: TriggerContext,
    htf_bias: str = "NEUTRAL",
    fvg_strength: float = 0.0,
    ob_strength: float = 0.0,
    side: str = "BUY",
) -> ScoreInputVector:
    """
    Build structured score input vector from SMC components.
    
    This is the INSTITUTIONAL input format - returns vector, NOT aggregated score.
    """
    # Regime score: regime direction alignment
    regime_dir = regime.direction.lower()
    side_dir = "bullish" if side.upper() in ("BUY", "BULLISH") else "bearish"
    regime_score = 1.0 if regime_dir == side_dir else 0.0
    
    # Structure alignment: from trigger strength
    structure_alignment = trigger.strength if trigger else 0.0
    
    # Liquidity quality: compact vector-friendly proxy from available context flags.
    liquidity_events = sum(
        int(flag)
        for flag in (
            liquidity.sweep,
            liquidity.displacement,
            liquidity.equal_highs,
            liquidity.equal_lows,
        )
    )
    liquidity_quality = min(1.0, liquidity_events / 4.0)
    
    # Trigger confidence: event presence
    trigger_confidence = 1.0 if trigger and trigger.structure_event else 0.0
    
    # HTF alignment
    htf_alignment = 1.0 if htf_bias.upper() == side_dir.upper() else 0.0
    
    return ScoreInputVector(
        regime_score=regime_score,
        structure_alignment=structure_alignment,
        liquidity_quality=liquidity_quality,
        trigger_confidence=trigger_confidence,
        htf_alignment=htf_alignment,
        fvg_strength=fvg_strength,
        ob_strength=ob_strength,
    )
