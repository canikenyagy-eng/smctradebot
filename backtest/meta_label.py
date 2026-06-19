from __future__ import annotations

from dataclasses import dataclass
import math

from core.signal_engine import TradeSignal


@dataclass(frozen=True)
class MetaLabelSettings:
    enabled: bool = False
    mode: str = "analysis_only"
    probability_threshold: float = 0.55
    enable_size_adjustment: bool = False
    low_probability_multiplier: float = 0.75
    high_probability_multiplier: float = 1.10
    high_probability_threshold: float = 0.72
    score_weight: float = 0.35
    regime_weight: float = 0.20
    trigger_weight: float = 0.20
    session_weight: float = 0.10
    zone_weight: float = 0.05
    htf_weight: float = 0.10
    spread_penalty_weight: float = 0.10

    def sanitized(self) -> "MetaLabelSettings":
        mode = self.mode.strip().lower()
        if mode not in {"analysis_only", "hard_gate"}:
            mode = "analysis_only"
        threshold = max(0.0, min(1.0, float(self.probability_threshold)))
        high_threshold = max(threshold, min(1.0, float(self.high_probability_threshold)))
        low_mult = max(0.05, float(self.low_probability_multiplier))
        high_mult = max(low_mult, float(self.high_probability_multiplier))
        weights = [
            max(0.0, float(self.score_weight)),
            max(0.0, float(self.regime_weight)),
            max(0.0, float(self.trigger_weight)),
            max(0.0, float(self.session_weight)),
            max(0.0, float(self.zone_weight)),
            max(0.0, float(self.htf_weight)),
        ]
        total = sum(weights)
        if total <= 0:
            weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            total = 1.0
        normalized = [value / total for value in weights]
        return MetaLabelSettings(
            enabled=bool(self.enabled),
            mode=mode,
            probability_threshold=threshold,
            enable_size_adjustment=bool(self.enable_size_adjustment),
            low_probability_multiplier=low_mult,
            high_probability_multiplier=high_mult,
            high_probability_threshold=high_threshold,
            score_weight=normalized[0],
            regime_weight=normalized[1],
            trigger_weight=normalized[2],
            session_weight=normalized[3],
            zone_weight=normalized[4],
            htf_weight=normalized[5],
            spread_penalty_weight=max(0.0, min(1.0, float(self.spread_penalty_weight))),
        )


@dataclass(frozen=True)
class MetaLabelDecision:
    enabled: bool
    probability: float
    accepted: bool
    mode: str
    size_multiplier: float
    reason: str
    features: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "probability": round(float(self.probability), 6),
            "accepted": self.accepted,
            "mode": self.mode,
            "size_multiplier": round(float(self.size_multiplier), 6),
            "reason": self.reason,
            "features": {key: round(float(value), 6) for key, value in self.features.items()},
        }


class MetaLabelEngine:
    def __init__(self, settings: MetaLabelSettings | None = None) -> None:
        self.settings = (settings or MetaLabelSettings()).sanitized()

    @staticmethod
    def _zone_feature(signal: TradeSignal) -> float:
        side = signal.side.upper()
        zone = signal.zone.upper()
        if side == "BUY" and zone == "DISCOUNT":
            return 1.0
        if side == "SELL" and zone == "PREMIUM":
            return 1.0
        return 0.25

    @staticmethod
    def _regime_feature(signal: TradeSignal) -> float:
        label = signal.regime_label.strip().lower()
        direction = signal.regime_direction.strip().lower()
        side = signal.side.upper()
        directional_match = (
            (side == "BUY" and direction == "bullish")
            or (side == "SELL" and direction == "bearish")
        )
        if label == "trend":
            return 0.75 if directional_match else 0.55
        if label == "expansion":
            return 0.65 if directional_match else 0.50
        if label == "range":
            return 0.50
        if label == "contraction":
            return 0.35
        return 0.45

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = max(-16.0, min(16.0, value))
        return 1.0 / (1.0 + math.exp(-clipped))

    def evaluate(self, *, signal: TradeSignal, spread_pips: float = 0.0) -> MetaLabelDecision:
        if not self.settings.enabled:
            return MetaLabelDecision(
                enabled=False,
                probability=1.0,
                accepted=True,
                mode=self.settings.mode,
                size_multiplier=1.0,
                reason="disabled",
                features={},
            )

        score = max(0.0, min(1.0, signal.score / 100.0))
        regime = self._regime_feature(signal)
        trigger = max(0.0, min(1.0, signal.trigger_strength / 20.0))
        session = max(0.0, min(1.0, signal.score_breakdown.session_timing / 20.0))
        zone = self._zone_feature(signal)
        htf = max(0.0, min(1.0, signal.score_breakdown.htf_alignment / 25.0))
        spread_penalty = min(0.40, max(0.0, float(spread_pips) / 8.0) * self.settings.spread_penalty_weight)

        linear = (
            score * self.settings.score_weight
            + regime * self.settings.regime_weight
            + trigger * self.settings.trigger_weight
            + session * self.settings.session_weight
            + zone * self.settings.zone_weight
            + htf * self.settings.htf_weight
            - spread_penalty
        )
        probability = self._sigmoid((linear - 0.5) * 6.0)
        accepted = probability >= self.settings.probability_threshold

        size_multiplier = 1.0
        if self.settings.enable_size_adjustment:
            if probability < self.settings.probability_threshold:
                size_multiplier = self.settings.low_probability_multiplier
            elif probability >= self.settings.high_probability_threshold:
                size_multiplier = self.settings.high_probability_multiplier

        reason = "accepted"
        if not accepted:
            reason = f"meta_probability_below_threshold({probability:.3f}<{self.settings.probability_threshold:.3f})"

        return MetaLabelDecision(
            enabled=True,
            probability=probability,
            accepted=accepted,
            mode=self.settings.mode,
            size_multiplier=size_multiplier,
            reason=reason,
            features={
                "score": score,
                "regime": regime,
                "trigger": trigger,
                "session": session,
                "zone": zone,
                "htf": htf,
                "spread_penalty": spread_penalty,
                "linear": linear,
            },
        )
