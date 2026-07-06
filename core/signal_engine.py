from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import pandas as pd

from core.adaptive_weights import (
    AdaptiveWeightSettings,
    DynamicThresholdSettings,
    DynamicThresholdTracker,
    ScoreNormalizationSettings,
    ScoreNormalizer,
)
from core.correlation import CorrelationCap, SignalCandidate
from core.entry import EntryPlan, build_entry_plan
from core.expectancy_engine_v3 import ExpectancyEngine, ExpectancySettings
from core.live_profile import LiveProfileSelector, LiveProfileSettings
from core.pair_profiles import PairRuntimeProfile, clean_pair
from core.portfolio import CurrencyExposureRecord, exposure_expiry, signal_currency_deltas
from core.regime_gate import RegimeGate, RegimeGateSettings
from core.scoring import ScoreBreakdown, calculate_score_details, score_session_timing
from core.session_gate import SessionGate, SessionGateSettings
from core.shadow import ShadowFeatureContext, analyze_shadow_context
from core.trade_management import build_trade_management_plan
from core.trade_gate_v2 import TradeGateV2, TradeGateSettings
from data.market_data import MarketDataClient
from execution.news import NewsFilter
from smc.liquidity import LiquidityContext, analyze_liquidity
from smc.mtf import MTFContext, premium_discount_context
from smc.regime import RegimeState, analyze_regime
from smc.structure_quality import StructureQualitySettings, evaluate_structure_quality
from smc.trigger import TriggerContext, analyze_trigger
from smc.structure import StructureState, detect_bos_choch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeSignal:
    symbol: str
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    entry_mode: str
    entry_source: str
    entry_summary: str
    management_summary: str
    partial_take_profit: float | None
    partial_take_fraction: float
    break_even_r: float
    trailing_enabled: bool
    trailing_start_r: float
    trailing_lookback_bars: int
    time_stop_bars: int
    score: int
    htf_bias: str
    regime_label: str
    regime_direction: str
    zone: str
    trigger_direction: str
    trigger_event: str
    trigger_strength: int
    structure_event: str
    structure_trend: str
    generated_at: datetime
    score_breakdown: ScoreBreakdown
    meta: dict[str, object] = field(default_factory=dict)

    def fingerprint(self) -> str:
        raw = (
            f"{self.symbol}|{self.side}|{self.entry_mode}|{self.entry_source}|{self.structure_event}|{self.zone}|"
            f"{self.entry:.5f}|{self.stop_loss:.5f}|{self.take_profit:.5f}|{self.time_stop_bars}|"
            f"{self.trailing_enabled}|{self.partial_take_fraction:.2f}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class SignalEvaluation:
    accepted: bool
    signal: TradeSignal | None
    rejection_stage: str | None
    rejection_reason: str | None
    details: dict[str, object]
    score_breakdown: ScoreBreakdown | None
    news_assessment: object | None
    regime_label: str | None = None
    score_value: int | None = None
    threshold_used: int | None = None
    recommended_threshold: int | None = None


@dataclass(frozen=True)
class SignalDrop:
    pair: str
    stage: str
    reason: str
    context: dict[str, object]


class SignalEngine:
    def __init__(
        self,
        market_data: MarketDataClient,
        news_filter: NewsFilter,
        htf_timeframe: str = "H1",
        ltf_timeframe: str = "M15",
        trigger_timeframe: str = "M5",
        min_score: int = 70,
        risk_reward: float = 2.0,
        swing_window: int = 3,
        pair_correlation_threshold: float = 0.82,
        correlation_lookback: int = 120,
        regime_short_window: int = 20,
        regime_long_window: int = 80,
        enable_shadow_scoring: bool = True,
        enable_mitigation_entry: bool = True,
        enable_order_block_shadow: bool = True,
        order_block_shadow_backtest_only: bool = False,
        allow_live_order_block_shadow: bool = True,
        currency_exposure_cap: int = 2,
        pair_cooldown_minutes: int = 30,
        max_entries_per_bias: int = 2,
        bias_window_minutes: int = 240,
        regime_opposition_confidence: float = 0.70,
        contraction_min_trigger_strength: int = 9,
        range_min_trigger_strength: int = 8,
        require_displacement_in_contraction: bool = True,
        enable_strict_ltf_direction_gate: bool = False,
        enable_market_fallback_entry: bool = True,
        market_fallback_min_trigger_strength: int = 0,
        market_fallback_require_displacement: bool = False,
        enable_pip_aware_liquidity: bool = False,
        liquidity_equal_level_tolerance_pips: float = 3.0,
        liquidity_atr_tolerance_factor: float = 0.0,
        session_min_score: int = 5,
        enable_session_gate: bool = False,
        session_gate_windows_utc: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
        session_gate_backtest_only: bool = True,
        allow_live_session_gate: bool = False,
        enable_regime_label_gate: bool = False,
        regime_label_blocklist: tuple[str, ...] | list[str] | None = None,
        regime_gate_backtest_only: bool = True,
        allow_live_regime_gate: bool = False,
        enable_smt_confirmation: bool = True,
        smt_backtest_only: bool = False,
        allow_live_smt_confirmation: bool = True,
        smt_hard_gate: bool = False,
        smt_min_strength: float = 60.0,
        smt_opposite_block_strength: float = 80.0,
        smt_reference_map: dict[str, str] | None = None,
        partial_tp_enabled: bool = True,
        partial_tp_r: float = 1.0,
        partial_tp_fraction: float = 0.50,
        break_even_r: float = 1.0,
        trailing_enabled: bool = True,
        trailing_start_r: float = 1.5,
        trailing_lookback_bars: int = 6,
        time_stop_bars: int = 48,
        portfolio_currency_gross_cap: int = 4,
        portfolio_currency_net_cap: int = 2,
        portfolio_exposure_window_minutes: int = 240,
        enable_adaptive_weights: bool = False,
        adaptive_regime_weights: dict[str, dict[str, float]] | None = None,
        adaptive_weights_preset: str = "default",
        enable_score_normalization: bool = False,
        score_normalization_method: str = "minmax",
        score_normalization_window: int = 200,
        score_normalization_scale_factor: float = 1.0,
        score_normalization_backtest_only: bool = True,
        allow_live_score_normalization: bool = False,
        runtime_mode: str = "live",
        enable_market_data_freshness_gate: bool = False,
        max_live_candle_age_seconds: int = 1800,
        enable_dynamic_threshold: bool = False,
        threshold_percentile: float = 80.0,
        threshold_rolling_window: int = 200,
        apply_dynamic_threshold: bool = False,
        dynamic_threshold_backtest_only: bool = True,
        allow_live_dynamic_threshold: bool = False,
        enable_structure_quality_scoring: bool = False,
        structure_quality_replaces_raw_structure_score: bool = False,
        structure_quality_scan_bars: int = 300,
        structure_quality_min_break_pips: float = 2.0,
        structure_quality_level_bucket_pips: float = 2.0,
        structure_quality_min_score_for_bonus: float = 60.0,
        structure_quality_max_bonus: int = 8,
        structure_quality_backtest_only: bool = True,
        allow_live_structure_quality_scoring: bool = False,
        structure_quality_allowed_regimes: tuple[str, ...] | list[str] | None = None,
        structure_quality_allowed_pairs: tuple[str, ...] | list[str] | None = None,
        structure_quality_excluded_pairs: tuple[str, ...] | list[str] | None = None,
        live_exit_profile_enabled: bool = False,
        live_exit_profile_preset: str = "default",
        live_exit_use_regime_profiles: bool = True,
        live_exit_volatility_rr_enabled: bool = False,
        live_exit_volatility_rr_floor: float = 0.85,
        live_exit_volatility_rr_cap: float = 1.25,
        live_exit_liquidity_trailing_enabled: bool = False,
        live_exit_liquidity_lookback_bars: int = 8,
        live_exit_liquidity_buffer_pips: float = 1.0,
        pair_runtime_profiles: dict[str, PairRuntimeProfile] | None = None,
        # Trade Gate v2
        enable_trade_gate_v2: bool = False,
    ) -> None:
        self.market_data = market_data
        self.news_filter = news_filter
        self.htf_timeframe = htf_timeframe
        self.ltf_timeframe = ltf_timeframe
        self.trigger_timeframe = trigger_timeframe
        self.runtime_mode = runtime_mode.lower().strip()
        self.min_score = min_score
        self.risk_reward = risk_reward
        self.swing_window = swing_window
        self.regime_short_window = regime_short_window
        self.regime_long_window = regime_long_window
        self.enable_shadow_scoring = enable_shadow_scoring
        self.enable_mitigation_entry = enable_mitigation_entry
        self.currency_exposure_cap = max(0, currency_exposure_cap)
        self.pair_cooldown_minutes = max(0, pair_cooldown_minutes)
        self.max_entries_per_bias = max(0, max_entries_per_bias)
        self.bias_window_minutes = max(0, bias_window_minutes)
        self.regime_opposition_confidence = max(0.0, min(1.0, regime_opposition_confidence))
        self.contraction_min_trigger_strength = max(0, min(20, contraction_min_trigger_strength))
        self.range_min_trigger_strength = max(0, min(20, range_min_trigger_strength))
        self.require_displacement_in_contraction = require_displacement_in_contraction
        self.enable_strict_ltf_direction_gate = bool(enable_strict_ltf_direction_gate)
        self.enable_market_fallback_entry = bool(enable_market_fallback_entry)
        self.market_fallback_min_trigger_strength = max(0, min(20, int(market_fallback_min_trigger_strength)))
        self.market_fallback_require_displacement = bool(market_fallback_require_displacement)
        self.enable_pip_aware_liquidity = bool(enable_pip_aware_liquidity)
        self.liquidity_equal_level_tolerance_pips = max(0.0, float(liquidity_equal_level_tolerance_pips))
        self.liquidity_atr_tolerance_factor = max(0.0, float(liquidity_atr_tolerance_factor))
        self.session_min_score = max(0, min(20, session_min_score))
        self._session_gate = SessionGate(
            SessionGateSettings(
                enabled=enable_session_gate,
                windows_utc=tuple(session_gate_windows_utc or ()),
                backtest_only=session_gate_backtest_only,
                allow_live=allow_live_session_gate,
            )
        )
        self._regime_gate = RegimeGate(
            RegimeGateSettings(
                enabled=enable_regime_label_gate,
                blocked_regimes=tuple(regime_label_blocklist or ()),
                backtest_only=regime_gate_backtest_only,
                allow_live=allow_live_regime_gate,
            )
        )
        self._pair_profiles = {
            clean_pair(pair): profile.sanitized()
            for pair, profile in (pair_runtime_profiles or {}).items()
            if clean_pair(pair)
        }
        self._pair_session_gates = {
            pair: SessionGate(profile.session_gate_settings)
            for pair, profile in self._pair_profiles.items()
            if profile.session_gate_settings is not None
        }
        self._pair_regime_gates = {
            pair: RegimeGate(profile.regime_gate_settings)
            for pair, profile in self._pair_profiles.items()
            if profile.regime_gate_settings is not None
        }
        self.enable_smt_confirmation = self._feature_enabled_for_runtime(
            enabled=enable_smt_confirmation,
            backtest_only=smt_backtest_only,
            allow_live=allow_live_smt_confirmation,
            runtime_mode=self.runtime_mode,
        )
        self.smt_hard_gate = smt_hard_gate
        self.smt_min_strength = max(0.0, min(100.0, smt_min_strength))
        self.smt_opposite_block_strength = max(0.0, min(100.0, smt_opposite_block_strength))
        self.smt_reference_map = {k.upper(): v.upper() for k, v in (smt_reference_map or {}).items()}
        self.partial_tp_enabled = partial_tp_enabled
        self.partial_tp_r = max(0.0, partial_tp_r)
        self.partial_tp_fraction = max(0.0, min(0.95, partial_tp_fraction))
        self.break_even_r = max(0.0, break_even_r)
        self.trailing_enabled = trailing_enabled
        self.trailing_start_r = max(0.0, trailing_start_r)
        self.trailing_lookback_bars = max(1, trailing_lookback_bars)
        self.time_stop_bars = max(0, time_stop_bars)
        self.enable_order_block_shadow = self._feature_enabled_for_runtime(
            enabled=enable_order_block_shadow,
            backtest_only=order_block_shadow_backtest_only,
            allow_live=allow_live_order_block_shadow,
            runtime_mode=self.runtime_mode,
        )
        self.portfolio_currency_gross_cap = max(0, portfolio_currency_gross_cap)
        self.portfolio_currency_net_cap = max(0, portfolio_currency_net_cap)
        self.portfolio_exposure_window_minutes = max(1, portfolio_exposure_window_minutes)
        self.enable_market_data_freshness_gate = bool(enable_market_data_freshness_gate)
        self.max_live_candle_age_seconds = max(1, int(max_live_candle_age_seconds))
        self.enable_adaptive_weights = enable_adaptive_weights
        self._adaptive_weight_settings = AdaptiveWeightSettings(
            enabled=enable_adaptive_weights,
            regime_weights=adaptive_regime_weights,
            preset=adaptive_weights_preset,
        )
        self._score_normalizer = ScoreNormalizer(
            ScoreNormalizationSettings(
                enabled=enable_score_normalization,
                method=score_normalization_method,
                window=max(10, score_normalization_window),
                scale_factor=max(0.0, score_normalization_scale_factor),
                backtest_only=score_normalization_backtest_only,
                allow_live=allow_live_score_normalization,
            )
        )
        self._dynamic_threshold_tracker = DynamicThresholdTracker(
            DynamicThresholdSettings(
                enabled=enable_dynamic_threshold,
                percentile=max(0.0, min(100.0, threshold_percentile)),
                rolling_window=max(10, threshold_rolling_window),
                apply_threshold=apply_dynamic_threshold,
                backtest_only=dynamic_threshold_backtest_only,
                allow_live=allow_live_dynamic_threshold,
            )
        )
        self._last_recommended_threshold: int | None = None
        self.structure_quality_replaces_raw_structure_score = bool(structure_quality_replaces_raw_structure_score)
        self._structure_quality_settings = StructureQualitySettings(
            enabled=enable_structure_quality_scoring,
            scan_bars=structure_quality_scan_bars,
            min_break_pips=structure_quality_min_break_pips,
            level_bucket_pips=structure_quality_level_bucket_pips,
            min_score_for_bonus=structure_quality_min_score_for_bonus,
            max_bonus=structure_quality_max_bonus,
            backtest_only=structure_quality_backtest_only,
            allow_live=allow_live_structure_quality_scoring,
            allowed_regimes=tuple(structure_quality_allowed_regimes or ()),
            allowed_pairs=tuple(structure_quality_allowed_pairs or ()),
            excluded_pairs=tuple(structure_quality_excluded_pairs or ()),
        ).sanitized()
        self._live_profile = LiveProfileSelector(
            LiveProfileSettings(
                enabled=live_exit_profile_enabled,
                preset=live_exit_profile_preset,
                use_regime_profiles=live_exit_use_regime_profiles,
                volatility_rr_enabled=live_exit_volatility_rr_enabled,
                volatility_rr_floor=live_exit_volatility_rr_floor,
                volatility_rr_cap=live_exit_volatility_rr_cap,
                liquidity_trailing_enabled=live_exit_liquidity_trailing_enabled,
                liquidity_lookback_bars=live_exit_liquidity_lookback_bars,
                liquidity_buffer_pips=live_exit_liquidity_buffer_pips,
            )
        )
        self.correlation_cap = CorrelationCap(
            threshold=pair_correlation_threshold,
            lookback=correlation_lookback,
        )
        
        # Trade Gate v2
        self._trade_gate = None
        if enable_trade_gate_v2:
            self._trade_gate = TradeGateV2(
                settings=TradeGateSettings(
                    enabled=True,
                    min_regime_tradability=30,
                    check_risk_engine=False,  # Risk engine not wired yet
                    check_portfolio=False,  # Portfolio checks already done above
                    check_session=True,
                    block_transition=True,
                )
            )
        
        self._pair_cooldown_until: dict[str, datetime] = {}
        self._bias_history: dict[tuple[str, str], list[datetime]] = {}
        self._currency_exposure_records: list[CurrencyExposureRecord] = []

    @staticmethod
    def _feature_enabled_for_runtime(
        *,
        enabled: bool,
        backtest_only: bool,
        allow_live: bool,
        runtime_mode: str,
    ) -> bool:
        if not enabled:
            return False
        if not backtest_only:
            return True
        return runtime_mode.lower().strip() == "backtest" or bool(allow_live)

    @staticmethod
    def _log_rejection(pair: str, stage: str, rejection_reason: str, **context: object) -> None:
        if context:
            extra = ", ".join(f"{key}={value}" for key, value in context.items())
            logger.info("[%s] filtered at %s: %s | %s", pair, stage, rejection_reason, extra)
            return

        logger.info("[%s] filtered at %s: %s", pair, stage, rejection_reason)

    @staticmethod
    def _log_acceptance(signal: TradeSignal, score: ScoreBreakdown) -> None:
        logger.info(
            "[%s] accepted: side=%s entry=%.5f sl=%.5f tp=%.5f mode=%s source=%s score=%s bias=%s regime=%s/%s zone=%s trigger=%s event=%s strength=%s shadow=%s",
            signal.symbol,
            signal.side,
            signal.entry,
            signal.stop_loss,
            signal.take_profit,
            signal.entry_mode,
            signal.entry_source,
            score.total,
            signal.htf_bias,
            signal.regime_label,
            signal.regime_direction,
            signal.zone,
            signal.trigger_direction,
            signal.trigger_event,
            signal.trigger_strength,
            score.shadow_bonus,
        )

    def _pair_profile(self, pair: str) -> PairRuntimeProfile | None:
        return self._pair_profiles.get(clean_pair(pair))

    def pair_runtime_profile(self, pair: str) -> PairRuntimeProfile | None:
        return self._pair_profile(pair)

    @staticmethod
    def _side_from_direction(direction: str | None) -> Optional[str]:
        if direction == "bullish":
            return "BUY"
        if direction == "bearish":
            return "SELL"
        return None

    def _resolve_side(
        self,
        structure: StructureState,
        trigger: TriggerContext,
        regime: RegimeState,
        mtf: MTFContext,
    ) -> tuple[Optional[str], str]:
        source = "none"

        side = self._side_from_direction(trigger.direction)
        if side is not None:
            return side, "trigger"

        side = self._side_from_direction(structure.direction)
        if side is not None:
            return side, "structure"

        if regime.is_directional:
            side = self._side_from_direction(regime.direction)
            if side is not None:
                return side, "regime"

        side = self._side_from_direction(mtf.bias)
        if side is not None:
            return side, "htf"

        return None, source

    def _check_ltf_direction_gate(
        self,
        *,
        side: str,
        source: str,
        structure: StructureState,
        trigger: TriggerContext,
        liquidity: LiquidityContext,
    ) -> tuple[bool, dict[str, object]]:
        side_dir = self._side_direction(side)
        aligned = {
            "trigger_direction": trigger.direction == side_dir,
            "structure_direction": structure.direction == side_dir,
            "liquidity_sweep": liquidity.sweep_direction == side_dir,
            "liquidity_displacement": liquidity.displacement_direction == side_dir,
        }
        context = {
            "enabled": self.enable_strict_ltf_direction_gate,
            "side": side,
            "source": source,
            "trigger_direction": trigger.direction.upper(),
            "structure_direction": (structure.direction or "none").upper(),
            "liquidity_sweep_direction": (liquidity.sweep_direction or "none").upper(),
            "liquidity_displacement_direction": (liquidity.displacement_direction or "none").upper(),
            "aligned": aligned,
        }
        if not self.enable_strict_ltf_direction_gate:
            return True, context
        if source in {"trigger", "structure"}:
            return True, context
        if any(aligned.values()):
            return True, context
        return False, context

    def _market_fallback_settings(self, pair_profile: PairRuntimeProfile | None) -> dict[str, object]:
        allow = self.enable_market_fallback_entry
        min_strength = self.market_fallback_min_trigger_strength
        require_displacement = self.market_fallback_require_displacement
        if pair_profile is not None:
            if pair_profile.allow_market_fallback is not None:
                allow = bool(pair_profile.allow_market_fallback)
            if pair_profile.market_fallback_min_trigger_strength is not None:
                min_strength = int(pair_profile.market_fallback_min_trigger_strength)
            if pair_profile.market_fallback_require_displacement is not None:
                require_displacement = bool(pair_profile.market_fallback_require_displacement)
        return {
            "allow": bool(allow),
            "min_trigger_strength": max(0, min(20, int(min_strength))),
            "require_displacement": bool(require_displacement),
        }

    def _market_fallback_allowed(
        self,
        *,
        side: str,
        trigger: TriggerContext,
        liquidity: LiquidityContext,
        pair_profile: PairRuntimeProfile | None,
    ) -> tuple[bool, dict[str, object]]:
        settings = self._market_fallback_settings(pair_profile)
        side_dir = self._side_direction(side)
        has_aligned_displacement = (
            trigger.liquidity.displacement_direction == side_dir
            or liquidity.displacement_direction == side_dir
        )
        allowed = bool(settings["allow"])
        reason = "allowed"
        if not allowed:
            reason = "market fallback disabled"
        elif trigger.strength < int(settings["min_trigger_strength"]):
            allowed = False
            reason = "trigger strength below market fallback threshold"
        elif bool(settings["require_displacement"]) and not has_aligned_displacement:
            allowed = False
            reason = "aligned displacement required for market fallback"
        return allowed, {
            **settings,
            "reason": reason,
            "trigger_strength": trigger.strength,
            "has_aligned_displacement": has_aligned_displacement,
        }

    def _trigger_for_scoring(self, trigger: TriggerContext, structure_quality_enabled: bool) -> TriggerContext:
        if not self.structure_quality_replaces_raw_structure_score or not structure_quality_enabled:
            return trigger
        if trigger.structure_event is None:
            return trigger
        adjusted_strength = max(0, trigger.strength - 8)
        return replace(
            trigger,
            structure_event=None,
            structure_trend="NEUTRAL",
            strength=adjusted_strength,
        )

    def _fetch_frames(self, pair: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        htf = self.market_data.fetch_ohlcv(pair, self.htf_timeframe)
        ltf = self.market_data.fetch_ohlcv(pair, self.ltf_timeframe)
        trigger = self.market_data.fetch_ohlcv(pair, self.trigger_timeframe)
        return htf, ltf, trigger

    @staticmethod
    def _volatility_ratio(frame: pd.DataFrame, short_window: int = 20, long_window: int = 80) -> float:
        if frame.empty:
            return 1.0
        closes = frame["close"].astype(float)
        returns = closes.pct_change().dropna()
        if len(returns) < max(short_window, long_window):
            return 1.0
        short_vol = float(returns.tail(short_window).std(ddof=0) or 0.0)
        long_vol = float(returns.tail(long_window).std(ddof=0) or 0.0)
        if long_vol <= 1e-9:
            return 1.0
        return max(0.5, min(1.5, short_vol / long_vol))

    @staticmethod
    def _pair_currencies(pair: str) -> tuple[str, str]:
        normalized = pair.upper().replace("/", "")
        return normalized[:3], normalized[3:6]

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        return pair.upper().replace("/", "")

    @staticmethod
    def _signal_currency_legs(signal: TradeSignal) -> list[tuple[str, str]]:
        base, quote = SignalEngine._pair_currencies(signal.symbol)
        if signal.side == "BUY":
            return [(base, "LONG"), (quote, "SHORT")]
        return [(base, "SHORT"), (quote, "LONG")]

    @staticmethod
    def _prune_times(items: list[datetime], cutoff: datetime) -> list[datetime]:
        return [item for item in items if item >= cutoff]

    def _prune_exposure_records(self, now: datetime) -> None:
        self._currency_exposure_records = [
            record for record in self._currency_exposure_records if record.expires_at > now
        ]

    def _current_exposure(self, now: datetime) -> tuple[dict[str, int], dict[str, int]]:
        self._prune_exposure_records(now)
        gross: dict[str, int] = {}
        net: dict[str, int] = {}
        for record in self._currency_exposure_records:
            gross[record.currency] = gross.get(record.currency, 0) + abs(record.delta)
            net[record.currency] = net.get(record.currency, 0) + record.delta
        return gross, net

    def _check_portfolio_exposure_cap(
        self,
        signal: TradeSignal,
        *,
        staged_gross: dict[str, int] | None = None,
        staged_net: dict[str, int] | None = None,
    ) -> tuple[bool, SignalDrop | None]:
        if self.portfolio_currency_gross_cap <= 0 and self.portfolio_currency_net_cap <= 0:
            return True, None

        now = signal.generated_at
        gross, net = self._current_exposure(now)
        staged_gross = staged_gross or {}
        staged_net = staged_net or {}
        deltas = signal_currency_deltas(signal.symbol, signal.side)

        for currency, delta in deltas.items():
            before_gross = gross.get(currency, 0) + staged_gross.get(currency, 0)
            before_net = net.get(currency, 0) + staged_net.get(currency, 0)
            after_gross = before_gross + abs(delta)
            after_net = before_net + delta

            if self.portfolio_currency_gross_cap > 0 and after_gross > self.portfolio_currency_gross_cap:
                return False, SignalDrop(
                    pair=signal.symbol,
                    stage="portfolio_exposure_gross",
                    reason="portfolio currency gross exposure cap",
                    context={
                        "currency": currency,
                        "before": before_gross,
                        "after": after_gross,
                        "cap": self.portfolio_currency_gross_cap,
                    },
                )

            if self.portfolio_currency_net_cap > 0 and abs(after_net) > self.portfolio_currency_net_cap:
                return False, SignalDrop(
                    pair=signal.symbol,
                    stage="portfolio_exposure_net",
                    reason="portfolio currency net exposure cap",
                    context={
                        "currency": currency,
                        "before": before_net,
                        "after": after_net,
                        "cap": self.portfolio_currency_net_cap,
                    },
                )

        return True, None

    def _register_portfolio_exposure(self, signal: TradeSignal) -> None:
        now = signal.generated_at
        expires_at = exposure_expiry(
            now,
            timeframe=self.trigger_timeframe,
            bars=signal.time_stop_bars,
            fallback_minutes=self.portfolio_exposure_window_minutes,
        )
        deltas = signal_currency_deltas(signal.symbol, signal.side)
        for currency, delta in deltas.items():
            self._currency_exposure_records.append(
                CurrencyExposureRecord(currency=currency, delta=delta, expires_at=expires_at)
            )

    def _check_release_constraints(self, signal: TradeSignal) -> tuple[bool, SignalDrop | None]:
        symbol = signal.symbol.upper().replace("/", "")
        now = signal.generated_at

        if self.pair_cooldown_minutes > 0:
            cooldown_until = self._pair_cooldown_until.get(symbol)
            if cooldown_until is not None and now < cooldown_until:
                return False, SignalDrop(
                    pair=symbol,
                    stage="cooldown",
                    reason="pair cooldown active",
                    context={
                        "until": cooldown_until.isoformat(),
                        "minutes": self.pair_cooldown_minutes,
                    },
                )

        if self.max_entries_per_bias > 0 and self.bias_window_minutes > 0:
            key = (symbol, signal.htf_bias.upper())
            history = self._bias_history.get(key, [])
            cutoff = now - timedelta(minutes=self.bias_window_minutes)
            history = self._prune_times(history, cutoff)
            self._bias_history[key] = history

            if len(history) >= self.max_entries_per_bias:
                return False, SignalDrop(
                    pair=symbol,
                    stage="bias_limit",
                    reason="max entries per HTF bias reached",
                    context={
                        "bias": signal.htf_bias,
                        "limit": self.max_entries_per_bias,
                        "window_minutes": self.bias_window_minutes,
                    },
                )

        allowed, drop = self._check_portfolio_exposure_cap(signal)
        if not allowed:
            return allowed, drop

        return True, None

    def reset_release_state(self) -> None:
        self._pair_cooldown_until.clear()
        self._bias_history.clear()
        self._currency_exposure_records.clear()
        self._score_normalizer.reset()
        self._dynamic_threshold_tracker.reset()
        self._last_recommended_threshold = None

    def _register_released_signal(self, signal: TradeSignal) -> None:
        symbol = signal.symbol.upper().replace("/", "")
        now = signal.generated_at

        if self.pair_cooldown_minutes > 0:
            self._pair_cooldown_until[symbol] = now + timedelta(minutes=self.pair_cooldown_minutes)

        if self.max_entries_per_bias > 0 and self.bias_window_minutes > 0:
            key = (symbol, signal.htf_bias.upper())
            history = self._bias_history.get(key, [])
            cutoff = now - timedelta(minutes=self.bias_window_minutes)
            history = self._prune_times(history, cutoff)
            history.append(now)
            self._bias_history[key] = history

        self._register_portfolio_exposure(signal)

    def gate_signal_release(self, signal: TradeSignal, *, commit: bool) -> tuple[bool, SignalDrop | None]:
        allowed, drop = self._check_release_constraints(signal)
        if allowed and commit:
            self._register_released_signal(signal)
        return allowed, drop

    def _apply_currency_exposure_cap(
        self,
        candidates: list[SignalCandidate],
    ) -> tuple[list[SignalCandidate], list[SignalDrop]]:
        if self.currency_exposure_cap <= 0:
            return candidates, []

        exposure: dict[tuple[str, str], int] = {}
        kept: list[SignalCandidate] = []
        dropped: list[SignalDrop] = []

        for candidate in candidates:
            legs = self._signal_currency_legs(candidate.signal)
            blocked_leg: tuple[str, str] | None = None
            for leg in legs:
                if exposure.get(leg, 0) >= self.currency_exposure_cap:
                    blocked_leg = leg
                    break

            if blocked_leg is not None:
                currency, direction = blocked_leg
                dropped.append(
                    SignalDrop(
                        pair=candidate.pair,
                        stage="currency_exposure",
                        reason="directional currency exposure cap",
                        context={
                            "currency": currency,
                            "direction": direction,
                            "cap": self.currency_exposure_cap,
                        },
                    )
                )
                continue

            kept.append(candidate)
            for leg in legs:
                exposure[leg] = exposure.get(leg, 0) + 1

        return kept, dropped

    def _apply_portfolio_exposure_cap(
        self,
        candidates: list[SignalCandidate],
    ) -> tuple[list[SignalCandidate], list[SignalDrop]]:
        if self.portfolio_currency_gross_cap <= 0 and self.portfolio_currency_net_cap <= 0:
            return candidates, []

        staged_gross: dict[str, int] = {}
        staged_net: dict[str, int] = {}
        kept: list[SignalCandidate] = []
        dropped: list[SignalDrop] = []

        for candidate in candidates:
            allowed, drop = self._check_portfolio_exposure_cap(
                candidate.signal,
                staged_gross=staged_gross,
                staged_net=staged_net,
            )
            if not allowed and drop is not None:
                dropped.append(drop)
                continue

            kept.append(candidate)
            deltas = signal_currency_deltas(candidate.signal.symbol, candidate.signal.side)
            for currency, delta in deltas.items():
                staged_gross[currency] = staged_gross.get(currency, 0) + abs(delta)
                staged_net[currency] = staged_net.get(currency, 0) + delta

        return kept, dropped

    @staticmethod
    def _side_direction(side: str) -> str:
        return "bullish" if side.upper() == "BUY" else "bearish"

    def _check_live_data_freshness(
        self,
        *,
        pair: str,
        signal_time: datetime,
        trigger_rows: int,
    ) -> tuple[bool, dict[str, object]]:
        if not self.enable_market_data_freshness_gate or self.runtime_mode != "live":
            return True, {}

        timestamp = signal_time
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
        age_seconds = (now - timestamp).total_seconds()
        context = {
            "pair": pair.upper().replace("/", ""),
            "trigger_timeframe": self.trigger_timeframe,
            "last_candle_time": timestamp.isoformat(),
            "age_seconds": round(age_seconds, 3),
            "max_age_seconds": self.max_live_candle_age_seconds,
            "trigger_rows": trigger_rows,
        }
        if age_seconds < -60:
            context["now_utc"] = now.isoformat()
            return False, {**context, "reason": "trigger candle timestamp is in the future"}
        if age_seconds > self.max_live_candle_age_seconds:
            context["now_utc"] = now.isoformat()
            return False, {**context, "reason": "trigger candle is stale"}
        return True, context

    @staticmethod
    def _with_total(score: ScoreBreakdown, total: int) -> ScoreBreakdown:
        return ScoreBreakdown(
            htf_alignment=score.htf_alignment,
            regime_alignment=score.regime_alignment,
            trigger_confirmation=score.trigger_confirmation,
            liquidity_displacement=score.liquidity_displacement,
            premium_discount=score.premium_discount,
            news_filter=score.news_filter,
            session_timing=score.session_timing,
            fvg_alignment=score.fvg_alignment,
            order_block_alignment=score.order_block_alignment,
            mitigation_alignment=score.mitigation_alignment,
            smt_alignment=score.smt_alignment,
            shadow_bonus=score.shadow_bonus,
            total=max(0, min(100, int(total))),
            structure_quality=score.structure_quality,
        )

    def _check_regime_filter(
        self,
        *,
        side: str,
        regime: RegimeState,
        trigger: TriggerContext,
        liquidity: LiquidityContext,
    ) -> tuple[bool, str | None, dict[str, object]]:
        side_dir = self._side_direction(side)

        if (
            regime.direction in {"bullish", "bearish"}
            and regime.direction != side_dir
            and regime.confidence >= self.regime_opposition_confidence
        ):
            return (
                False,
                "strong opposite regime direction",
                {
                    "regime_label": regime.label.upper(),
                    "regime_direction": regime.direction.upper(),
                    "confidence": round(regime.confidence, 4),
                    "threshold": self.regime_opposition_confidence,
                    "side": side,
                },
            )

        if regime.label == "contraction":
            if self.require_displacement_in_contraction and not liquidity.displacement:
                return (
                    False,
                    "contraction without displacement",
                    {
                        "trigger_strength": trigger.strength,
                        "displacement": liquidity.displacement,
                        "side": side,
                    },
                )

            if trigger.strength < self.contraction_min_trigger_strength:
                return (
                    False,
                    "contraction trigger too weak",
                    {
                        "trigger_strength": trigger.strength,
                        "threshold": self.contraction_min_trigger_strength,
                        "side": side,
                    },
                )

        if regime.label == "range" and trigger.strength < self.range_min_trigger_strength and not liquidity.sweep:
            return (
                False,
                "range trigger too weak without sweep",
                {
                    "trigger_strength": trigger.strength,
                    "threshold": self.range_min_trigger_strength,
                    "liquidity_sweep": liquidity.sweep,
                    "side": side,
                },
            )

        return True, None, {}

    def _resolve_smt_reference_pair(self, pair: str, universe: set[str] | None = None) -> str | None:
        symbol = self._normalize_pair(pair)

        mapped = self.smt_reference_map.get(symbol)
        if mapped is not None and mapped != symbol:
            if universe is None or mapped in universe:
                return mapped

        if universe is None:
            return None

        base, quote = self._pair_currencies(symbol)
        for candidate in sorted(universe):
            if candidate == symbol:
                continue
            c_base, c_quote = self._pair_currencies(candidate)
            if c_quote == quote and c_base != base:
                return candidate
        return None

    def _check_smt_filter(
        self,
        *,
        side: str,
        regime: RegimeState,
        shadow: ShadowFeatureContext | None,
    ) -> tuple[bool, str | None, dict[str, object]]:
        if not self.enable_smt_confirmation or shadow is None or shadow.reference_pair is None:
            return True, None, {}

        side_dir = self._side_direction(side)
        directional_regime = regime.label in {"trend", "expansion"} and regime.direction in {"bullish", "bearish"}

        if shadow.smt is None:
            if self.smt_hard_gate and directional_regime:
                return (
                    False,
                    "SMT confirmation missing in directional regime",
                    {
                        "reference_pair": shadow.reference_pair,
                        "regime_label": regime.label.upper(),
                        "regime_direction": regime.direction.upper(),
                    },
                )
            return True, None, {}

        if shadow.smt.direction != side_dir and shadow.smt.strength >= self.smt_opposite_block_strength:
            return (
                False,
                "strong opposite SMT divergence",
                {
                    "reference_pair": shadow.reference_pair,
                    "smt_direction": shadow.smt.direction.upper(),
                    "smt_strength": shadow.smt.strength,
                    "threshold": self.smt_opposite_block_strength,
                    "side": side,
                },
            )

        if self.smt_hard_gate and directional_regime:
            if shadow.smt.direction != side_dir:
                return (
                    False,
                    "SMT direction does not confirm trade",
                    {
                        "reference_pair": shadow.reference_pair,
                        "smt_direction": shadow.smt.direction.upper(),
                        "side": side,
                    },
                )
            if shadow.smt.strength < self.smt_min_strength:
                return (
                    False,
                    "SMT strength below threshold",
                    {
                        "reference_pair": shadow.reference_pair,
                        "smt_strength": shadow.smt.strength,
                        "threshold": self.smt_min_strength,
                    },
                )

        return True, None, {}

    def evaluate_snapshot(
        self,
        pair: str,
        htf: pd.DataFrame,
        ltf: pd.DataFrame,
        *,
        trigger_frame: pd.DataFrame | None = None,
        reference_pair: str | None = None,
        reference_trigger_frame: pd.DataFrame | None = None,
        news_assessment: object | None = None,
        emit_logs: bool = True,
    ) -> SignalEvaluation:
        trigger_frame = trigger_frame if trigger_frame is not None else ltf

        if ltf.empty:
            details = {"reason": "empty ltf frame"}
            if emit_logs:
                self._log_rejection(pair, "history", "empty ltf frame")
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="history",
                rejection_reason="empty ltf frame",
                details=details,
                score_breakdown=None,
                news_assessment=news_assessment,
            )

        required_htf = max(120, self.regime_long_window)
        required_ltf = max(80, self.swing_window * 2 + 3)
        required_trigger = max(40, self.swing_window * 6)

        if len(htf) < required_htf or len(ltf) < required_ltf or len(trigger_frame) < required_trigger:
            details = {
                "htf": len(htf),
                "ltf": len(ltf),
                "trigger": len(trigger_frame),
                "required_htf": required_htf,
                "required_ltf": required_ltf,
                "required_trigger": required_trigger,
            }
            if emit_logs:
                self._log_rejection(pair, "history", "insufficient candles", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="history",
                rejection_reason="insufficient candles",
                details=details,
                score_breakdown=None,
                news_assessment=news_assessment,
            )

        pair_key = clean_pair(pair)
        pair_profile = self._pair_profile(pair_key)
        pair_min_score = pair_profile.min_score if pair_profile and pair_profile.min_score is not None else self.min_score
        pair_profile_meta = pair_profile.to_dict() if pair_profile is not None else None
        session_gate_for_pair = self._pair_session_gates.get(pair_key, self._session_gate)
        regime_gate_for_pair = self._pair_regime_gates.get(pair_key, self._regime_gate)

        signal_time = trigger_frame.index[-1].to_pydatetime()
        fresh, freshness_context = self._check_live_data_freshness(
            pair=pair,
            signal_time=signal_time,
            trigger_rows=len(trigger_frame),
        )
        if not fresh:
            details = {**freshness_context, "pair_profile": pair_profile_meta}
            if emit_logs:
                self._log_rejection(pair, "data_freshness", details.get("reason", "stale market data"), **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="data_freshness",
                rejection_reason=str(details.get("reason", "stale market data")),
                details=details,
                score_breakdown=None,
                news_assessment=news_assessment,
            )

        session_gate = session_gate_for_pair.evaluate(signal_time, self.runtime_mode)
        if not session_gate.allowed:
            details = {
                **session_gate.to_dict(),
                "windows_utc": session_gate_for_pair.settings.to_dict().get("windows_utc", []),
                "pair_profile": pair_profile_meta,
            }
            if emit_logs:
                self._log_rejection(pair, "session_gate", "outside allowed session window", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="session_gate",
                rejection_reason="outside allowed session window",
                details=details,
                score_breakdown=None,
                news_assessment=news_assessment,
            )

        liquidity_tolerance_pips = self.liquidity_equal_level_tolerance_pips if self.enable_pip_aware_liquidity else None
        liquidity_atr_factor = self.liquidity_atr_tolerance_factor if self.enable_pip_aware_liquidity else 0.0
        structure = detect_bos_choch(ltf, window=self.swing_window)
        liquidity: LiquidityContext = analyze_liquidity(
            ltf,
            swing_window=self.swing_window,
            pair=pair if self.enable_pip_aware_liquidity else None,
            tolerance_pips=liquidity_tolerance_pips,
            atr_tolerance_factor=liquidity_atr_factor,
        )
        current_price = float(trigger_frame["close"].iloc[-1])
        mtf: MTFContext = premium_discount_context(
            htf,
            current_price=current_price,
            swing_window=self.swing_window,
        )

        regime: RegimeState = analyze_regime(
            htf,
            short_window=self.regime_short_window,
            long_window=self.regime_long_window,
        )
        
        # === INSTITUTIONAL PIPELINE: REGIME CHECK FIRST ===
        # If transition regime, block signal BEFORE expensive SMC feature generation
        regime_label = regime.label.upper()
        if regime_label == "TRANSITION":
            details = {
                "regime": regime_label,
                "regime_direction": regime.direction.upper(),
            }
            if emit_logs:
                self._log_rejection(pair, "regime", "transition regime - blocked", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="regime",
                rejection_reason="transition regime - no edge",
                details=details,
                score_breakdown=None,
                news_assessment=news_assessment,
                )

        regime_gate = regime_gate_for_pair.evaluate(regime_label, self.runtime_mode)
        if not regime_gate.allowed:
            details = {
                **regime_gate.to_dict(),
                "regime_direction": regime.direction.upper(),
                "pair_profile": pair_profile_meta,
            }
            if emit_logs:
                self._log_rejection(pair, "regime_gate", "blocked regime label", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="regime_gate",
                rejection_reason="blocked regime label",
                details=details,
                score_breakdown=None,
                news_assessment=news_assessment,
                regime_label=regime_label,
            )

        trigger: TriggerContext = analyze_trigger(
            trigger_frame,
            swing_window=max(2, self.swing_window - 1),
            pair=pair if self.enable_pip_aware_liquidity else None,
            liquidity_tolerance_pips=liquidity_tolerance_pips,
            liquidity_atr_tolerance_factor=liquidity_atr_factor,
        )
        regime_direction = regime.direction.upper()

        side, source = self._resolve_side(structure, trigger, regime, mtf)
        if side is None:
            details = {
                "structure": (structure.direction or "none").upper(),
                "trigger": trigger.direction.upper(),
                "regime_label": regime_label,
                "regime": regime.direction.upper(),
                "bias": mtf.bias.upper(),
            }
            if emit_logs:
                self._log_rejection(pair, "direction", "no directional confluence", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="direction",
                rejection_reason="no directional confluence",
                details=details,
                score_breakdown=None,
                news_assessment=news_assessment,
                regime_label=regime_label,
            )

        ltf_gate_allowed, ltf_gate_context = self._check_ltf_direction_gate(
            side=side,
            source=source,
            structure=structure,
            trigger=trigger,
            liquidity=liquidity,
        )
        if not ltf_gate_allowed:
            details = {
                "regime_label": regime_label,
                "regime_direction": regime_direction,
                **ltf_gate_context,
            }
            if emit_logs:
                self._log_rejection(pair, "direction_gate", "strict LTF direction gate blocked", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="direction_gate",
                rejection_reason="strict LTF direction gate blocked",
                details=details,
                score_breakdown=None,
                news_assessment=news_assessment,
                regime_label=regime_label,
            )

        regime_allowed, regime_reason, regime_context = self._check_regime_filter(
            side=side,
            regime=regime,
            trigger=trigger,
            liquidity=liquidity,
        )
        if not regime_allowed:
            regime_context = {
                "regime_label": regime_label,
                "regime_direction": regime_direction,
                **regime_context,
            }
            if emit_logs:
                self._log_rejection(pair, "regime", regime_reason or "regime filter blocked", **regime_context)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="regime",
                rejection_reason=regime_reason or "regime filter blocked",
                details=regime_context,
                score_breakdown=None,
                news_assessment=news_assessment,
                regime_label=regime_label,
            )

        shadow: ShadowFeatureContext | None = None
        if self.enable_shadow_scoring:
            shadow = analyze_shadow_context(
                pair=pair,
                side=side,
                current_price=current_price,
                ltf_frame=ltf,
                trigger_frame=trigger_frame,
                reference_pair=reference_pair if self.enable_smt_confirmation else None,
                reference_frame=reference_trigger_frame if self.enable_smt_confirmation else None,
                include_order_block=self.enable_order_block_shadow,
            )

        structure_quality = evaluate_structure_quality(
            pair=pair,
            side=side,
            frame=ltf,
            structure_event=structure.event or "none",
            swing_window=self.swing_window,
            settings=self._structure_quality_settings,
            runtime_mode=self.runtime_mode,
            regime_label=regime_label,
        )
        scoring_trigger = self._trigger_for_scoring(trigger, structure_quality.enabled)
        structure_score_mode = (
            "structure_quality_soft"
            if scoring_trigger is not trigger
            else "raw_structure_trigger"
        )

        news = news_assessment or self.news_filter.evaluate_pair(pair)
        if not getattr(news, "allow_trading", False):
            details = {
                "regime_label": regime_label,
                "regime_direction": regime_direction,
                "summary": getattr(news, "summary", "news blocked"),
                "uncertainty": getattr(news, "uncertainty", "unknown"),
                "high_impact_events": getattr(news, "high_impact_events", 0),
                "shadow": {
                    "fvg": shadow.fvg_summary if shadow is not None else "shadow disabled",
                    "order_block": shadow.order_block_summary if shadow is not None else "shadow disabled",
                    "smt": shadow.smt_summary if shadow is not None else "shadow disabled",
                },
                "structure_quality": structure_quality.to_dict(),
            }
            if emit_logs:
                self._log_rejection(pair, "news", "high-impact uncertainty", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="news",
                rejection_reason="high-impact uncertainty",
                details=details,
                score_breakdown=None,
                news_assessment=news,
                regime_label=regime_label,
            )

        score, score_meta = calculate_score_details(
            pair=pair,
            side=side,
            htf_bias=mtf.bias,
            zone=mtf.zone,
            liquidity=liquidity,
            regime=regime,
            trigger=scoring_trigger,
            news=news,
            signal_time=signal_time,
            shadow=shadow,
            adaptive_weights=self._adaptive_weight_settings if self.enable_adaptive_weights else None,
            structure_quality_bonus=structure_quality.bonus,
        )
        normalized_score, normalization_meta = self._score_normalizer.transform(score.total, self.runtime_mode)
        if normalized_score != score.total:
            score = self._with_total(score, normalized_score)

        recommended_threshold = self._dynamic_threshold_tracker.recommended_threshold(self.runtime_mode)
        dynamic_active = self._dynamic_threshold_tracker.is_active(self.runtime_mode)
        threshold_used = pair_min_score
        if (
            dynamic_active
            and recommended_threshold is not None
            and self._dynamic_threshold_tracker.settings.apply_threshold
        ):
            threshold_used = recommended_threshold

        if emit_logs and dynamic_active and recommended_threshold is not None:
            if recommended_threshold != self._last_recommended_threshold:
                logger.info(
                    "[%s] dynamic threshold: recommended=%s used=%s apply=%s percentile=%.1f window=%s",
                    pair,
                    recommended_threshold,
                    threshold_used,
                    self._dynamic_threshold_tracker.settings.apply_threshold,
                    self._dynamic_threshold_tracker.settings.percentile,
                    self._dynamic_threshold_tracker.settings.rolling_window,
                )
                self._last_recommended_threshold = recommended_threshold

        self._dynamic_threshold_tracker.observe(score.total, self.runtime_mode)

        if score.total < threshold_used:
            details = {
                "score": score.total,
                "threshold": threshold_used,
                "static_min_score": pair_min_score,
                "global_min_score": self.min_score,
                "recommended_threshold": recommended_threshold,
                "pair_profile": pair_profile_meta,
                "regime_label": regime_label,
                "regime_direction": regime_direction,
                "htf": score.htf_alignment,
                "regime": score.regime_alignment,
                "trigger": score.trigger_confirmation,
                "liquidity": score.liquidity_displacement,
                "zone": score.premium_discount,
                "news": score.news_filter,
                "session": score.session_timing,
                "adaptive_weights": score_meta.get("adaptive_weights"),
                "normalization": normalization_meta,
                "structure_quality": structure_quality.to_dict(),
                "structure_score_mode": structure_score_mode,
                "shadow": {
                    "fvg": score.fvg_alignment,
                    "order_block": score.order_block_alignment,
                    "mitigation": score.mitigation_alignment,
                    "smt": score.smt_alignment,
                    "bonus": score.shadow_bonus,
                    "fvg_summary": shadow.fvg_summary if shadow is not None else "shadow disabled",
                    "order_block_summary": shadow.order_block_summary if shadow is not None else "shadow disabled",
                    "smt_summary": shadow.smt_summary if shadow is not None else "shadow disabled",
                },
            }
            if emit_logs:
                self._log_rejection(pair, "scoring", "score below threshold", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="scoring",
                rejection_reason="score below threshold",
                details=details,
                score_breakdown=score,
                news_assessment=news,
                regime_label=regime_label,
                score_value=score.total,
                threshold_used=threshold_used,
                recommended_threshold=recommended_threshold,
            )

        smt_allowed, smt_reason, smt_context = self._check_smt_filter(
            side=side,
            regime=regime,
            shadow=shadow,
        )
        if not smt_allowed:
            smt_context = {
                "regime_label": regime_label,
                "regime_direction": regime_direction,
                **smt_context,
            }
            if emit_logs:
                self._log_rejection(pair, "smt", smt_reason or "SMT filter blocked", **smt_context)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="smt",
                rejection_reason=smt_reason or "SMT filter blocked",
                details=smt_context,
                score_breakdown=score,
                news_assessment=news,
                regime_label=regime_label,
                score_value=score.total,
                threshold_used=threshold_used,
                recommended_threshold=recommended_threshold,
            )

        if self.session_min_score > 0 and score.session_timing < self.session_min_score:
            session_now = signal_time
            details = {
                "session_score": score.session_timing,
                "min_session_score": self.session_min_score,
                "hour_utc": session_now.hour,
                "pair": pair.upper().replace("/", ""),
                "calculated_session_score": score_session_timing(pair, session_now),
                "regime_label": regime_label,
                "regime_direction": regime_direction,
            }
            if emit_logs:
                self._log_rejection(pair, "session", "session timing below threshold", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="session",
                rejection_reason="session timing below threshold",
                details=details,
                score_breakdown=score,
                news_assessment=news,
                regime_label=regime_label,
                score_value=score.total,
                threshold_used=threshold_used,
                recommended_threshold=recommended_threshold,
            )

        live_profile_plan = self._live_profile.build_plan(
            regime_label=regime_label,
            fallback_rr=self.risk_reward,
            volatility_ratio=self._volatility_ratio(trigger_frame),
        )
        market_fallback_allowed, market_fallback_context = self._market_fallback_allowed(
            side=side,
            trigger=trigger,
            liquidity=liquidity,
            pair_profile=pair_profile,
        )
        entry_plan: EntryPlan | None = build_entry_plan(
            side=side,
            current_price=current_price,
            ltf_frame=ltf,
            risk_reward=live_profile_plan.target_rr,
            shadow=shadow,
            enable_mitigation_entry=self.enable_mitigation_entry,
            allow_market_fallback=market_fallback_allowed,
        )
        if entry_plan is None:
            details = {
                "side": side,
                "entry": round(current_price, 5),
                "regime_label": regime_label,
                "regime_direction": regime_direction,
                "market_fallback": market_fallback_context,
            }
            if emit_logs:
                self._log_rejection(pair, "entry", "unable to build entry plan", **details)
            return SignalEvaluation(
                accepted=False,
                signal=None,
                rejection_stage="entry",
                rejection_reason="unable to build entry plan",
                details=details,
                score_breakdown=score,
                news_assessment=news,
                regime_label=regime_label,
                score_value=score.total,
                threshold_used=threshold_used,
                recommended_threshold=recommended_threshold,
            )

        signal = TradeSignal(
            symbol=pair.upper().replace("/", ""),
            side=side,
            entry=entry_plan.entry,
            stop_loss=entry_plan.stop_loss,
            take_profit=entry_plan.take_profit,
            entry_mode=entry_plan.mode,
            entry_source=entry_plan.source,
            entry_summary=entry_plan.summary,
            management_summary="",
            partial_take_profit=None,
            partial_take_fraction=0.0,
            break_even_r=0.0,
            trailing_enabled=False,
            trailing_start_r=0.0,
            trailing_lookback_bars=1,
            time_stop_bars=self.time_stop_bars,
            score=score.total,
            htf_bias=mtf.bias.upper(),
            regime_label=regime_label,
            regime_direction=regime_direction,
            zone=mtf.zone.upper(),
            trigger_direction=trigger.direction.upper(),
            trigger_event=(trigger.structure_event or "NONE").upper(),
            trigger_strength=trigger.strength,
            structure_event=(structure.event or "NONE").upper(),
            structure_trend=structure.trend.upper(),
            generated_at=signal_time,
            score_breakdown=score,
            meta={
                "score_breakdown": score.contribution_dict(),
                "score_breakdown_raw": score_meta.get("raw_components", {}),
                "score_breakdown_weighted": score_meta.get("weighted_components", {}),
                "adaptive_weights": score_meta.get("adaptive_weights", {}),
                "structure_quality": structure_quality.to_dict(),
                "structure_score_mode": structure_score_mode,
                "score_normalization": normalization_meta,
                "score_raw_total": score_meta.get("raw_total"),
                "score_weighted_total": score_meta.get("weighted_total"),
                "threshold": {
                    "static_min_score": pair_min_score,
                    "global_min_score": self.min_score,
                    "pair_min_score": pair_min_score,
                    "used": threshold_used,
                    "recommended": recommended_threshold,
                    "dynamic_enabled": dynamic_active,
                    "dynamic_applied": (
                        dynamic_active and self._dynamic_threshold_tracker.settings.apply_threshold
                    ),
                },
                "live_profile": live_profile_plan.to_dict(),
                "pair_profile": pair_profile_meta,
                "ltf_direction_gate": ltf_gate_context,
                "market_fallback": market_fallback_context,
                "liquidity_tolerance": {
                    "pip_aware_enabled": self.enable_pip_aware_liquidity,
                    "mode": liquidity.equal_level_tolerance_mode,
                    "price": liquidity.equal_level_tolerance_price,
                    "pips": self.liquidity_equal_level_tolerance_pips,
                    "atr_factor": self.liquidity_atr_tolerance_factor,
                },
                "structure_confirmation": {
                    "last_swing_high_index": structure.last_swing_high_index,
                    "last_swing_low_index": structure.last_swing_low_index,
                    "last_swing_high_confirmed_at_index": structure.last_swing_high_confirmed_at_index,
                    "last_swing_low_confirmed_at_index": structure.last_swing_low_confirmed_at_index,
                    "event_confirmed_at_index": structure.event_confirmed_at_index,
                },
            },
        )
        first_partial = live_profile_plan.first_partial()
        management = build_trade_management_plan(
            side=signal.side,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            partial_tp_enabled=bool(first_partial) if live_profile_plan.enabled else self.partial_tp_enabled,
            partial_tp_r=first_partial.r_multiple if first_partial is not None else self.partial_tp_r,
            partial_tp_fraction=first_partial.fraction if first_partial is not None else self.partial_tp_fraction,
            break_even_r=live_profile_plan.break_even_r if live_profile_plan.enabled else self.break_even_r,
            trailing_enabled=live_profile_plan.trailing_enabled if live_profile_plan.enabled else self.trailing_enabled,
            trailing_start_r=live_profile_plan.trailing_start_r if live_profile_plan.enabled else self.trailing_start_r,
            trailing_lookback_bars=(
                live_profile_plan.trailing_lookback_bars if live_profile_plan.enabled else self.trailing_lookback_bars
            ),
            time_stop_bars=live_profile_plan.time_stop_bars if live_profile_plan.enabled else self.time_stop_bars,
        )
        management_summary = management.summary
        if live_profile_plan.enabled:
            management_summary = (
                f"profile={live_profile_plan.preset}/{live_profile_plan.regime_profile}"
                f" | target_rr={live_profile_plan.target_rr:.2f}"
                f" | vol_ratio={live_profile_plan.volatility_ratio:.2f}"
                f" | {management.summary}"
                f" | liq_trail={'ON' if live_profile_plan.liquidity_trailing_enabled else 'OFF'}"
                f" lookback={live_profile_plan.liquidity_lookback_bars}"
                f" buffer={live_profile_plan.liquidity_buffer_pips:.1f}p"
            )
        signal = TradeSignal(
            symbol=signal.symbol,
            side=signal.side,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_mode=signal.entry_mode,
            entry_source=signal.entry_source,
            entry_summary=signal.entry_summary,
            management_summary=management_summary,
            partial_take_profit=management.partial_take_profit,
            partial_take_fraction=management.partial_take_fraction,
            break_even_r=management.break_even_r,
            trailing_enabled=management.trailing_enabled,
            trailing_start_r=management.trailing_start_r,
            trailing_lookback_bars=management.trailing_lookback_bars,
            time_stop_bars=management.time_stop_bars,
            score=signal.score,
            htf_bias=signal.htf_bias,
            regime_label=signal.regime_label,
            regime_direction=signal.regime_direction,
            zone=signal.zone,
            trigger_direction=signal.trigger_direction,
            trigger_event=signal.trigger_event,
            trigger_strength=signal.trigger_strength,
            structure_event=signal.structure_event,
            structure_trend=signal.structure_trend,
            generated_at=signal.generated_at,
            score_breakdown=signal.score_breakdown,
            meta={**dict(signal.meta), "live_profile": live_profile_plan.to_dict()},
        )
        if emit_logs:
            self._log_acceptance(signal, score)

        return SignalEvaluation(
            accepted=True,
            signal=signal,
            rejection_stage=None,
            rejection_reason=None,
            details={
                "side": side,
                "score": score.total,
                "bias": mtf.bias.upper(),
                "zone": mtf.zone.upper(),
                "regime": regime_label,
                "regime_direction": regime_direction,
                "trigger": trigger.direction.upper(),
                "source": source,
                "threshold": {
                    "static_min_score": pair_min_score,
                    "global_min_score": self.min_score,
                    "used": threshold_used,
                    "recommended": recommended_threshold,
                    "dynamic_enabled": dynamic_active,
                    "dynamic_applied": (
                        dynamic_active and self._dynamic_threshold_tracker.settings.apply_threshold
                    ),
                },
                "adaptive_weights": score_meta.get("adaptive_weights"),
                "score_normalization": normalization_meta,
                "pair_profile": pair_profile_meta,
                "ltf_direction_gate": ltf_gate_context,
                "market_fallback": market_fallback_context,
                "liquidity_tolerance": {
                    "pip_aware_enabled": self.enable_pip_aware_liquidity,
                    "mode": liquidity.equal_level_tolerance_mode,
                    "price": liquidity.equal_level_tolerance_price,
                    "pips": self.liquidity_equal_level_tolerance_pips,
                    "atr_factor": self.liquidity_atr_tolerance_factor,
                },
                "structure_confirmation": {
                    "last_swing_high_index": structure.last_swing_high_index,
                    "last_swing_low_index": structure.last_swing_low_index,
                    "last_swing_high_confirmed_at_index": structure.last_swing_high_confirmed_at_index,
                    "last_swing_low_confirmed_at_index": structure.last_swing_low_confirmed_at_index,
                    "event_confirmed_at_index": structure.event_confirmed_at_index,
                },
                "structure_quality": structure_quality.to_dict(),
                "structure_score_mode": structure_score_mode,
                "entry": {
                    "mode": entry_plan.mode,
                    "source": entry_plan.source,
                    "summary": entry_plan.summary,
                    "zone": entry_plan.zone_kind,
                    "frame": entry_plan.zone_frame,
                },
                "entry_summary": entry_plan.summary,
                "management": {
                    "summary": management.summary,
                    "partial_take_profit": management.partial_take_profit,
                    "partial_take_fraction": management.partial_take_fraction,
                    "break_even_r": management.break_even_r,
                    "trailing_enabled": management.trailing_enabled,
                    "trailing_start_r": management.trailing_start_r,
                    "trailing_lookback_bars": management.trailing_lookback_bars,
                    "time_stop_bars": management.time_stop_bars,
                },
                "shadow": {
                    "fvg": score.fvg_alignment,
                    "order_block": score.order_block_alignment,
                    "mitigation": score.mitigation_alignment,
                    "smt": score.smt_alignment,
                    "bonus": score.shadow_bonus,
                    "fvg_summary": shadow.fvg_summary if shadow is not None else "shadow disabled",
                    "order_block_summary": shadow.order_block_summary if shadow is not None else "shadow disabled",
                    "smt_summary": shadow.smt_summary if shadow is not None else "shadow disabled",
                },
            },
            score_breakdown=score,
            news_assessment=news,
            regime_label=regime_label,
            score_value=score.total,
            threshold_used=threshold_used,
            recommended_threshold=recommended_threshold,
        )

    def generate_signal(self, pair: str) -> TradeSignal | None:
        try:
            htf, ltf, trigger = self._fetch_frames(pair)
        except Exception as exc:
            logger.warning("[%s] filtered at data: market data error | %s", pair, exc)
            return None

        reference_pair = self._resolve_smt_reference_pair(pair, None)
        reference_trigger_frame: pd.DataFrame | None = None
        if self.enable_smt_confirmation and reference_pair is not None:
            try:
                reference_trigger_frame = self.market_data.fetch_ohlcv(reference_pair, self.trigger_timeframe)
            except Exception as exc:
                logger.info("[%s] SMT reference unavailable: %s | %s", pair, reference_pair, exc)

        evaluation = self.evaluate_snapshot(
            pair,
            htf,
            ltf,
            trigger_frame=trigger,
            reference_pair=reference_pair,
            reference_trigger_frame=reference_trigger_frame,
            emit_logs=True,
        )
        if evaluation.signal is None:
            return None

        allowed, drop = self.gate_signal_release(evaluation.signal, commit=True)
        if not allowed and drop is not None:
            self._log_rejection(drop.pair, drop.stage, drop.reason, **drop.context)
            return None
        return evaluation.signal

    def scan_pairs(self, pairs: Iterable[str]) -> List[TradeSignal]:
        pair_list = [self._normalize_pair(item) for item in pairs]
        pair_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

        for pair in pair_list:
            try:
                pair_frames[pair] = self._fetch_frames(pair)
            except Exception as exc:
                logger.warning("[%s] filtered at data: market data error | %s", pair, exc)

        universe = set(pair_frames.keys())
        candidates: List[SignalCandidate] = []
        for pair in pair_list:
            frames = pair_frames.get(pair)
            if frames is None:
                continue
            htf, ltf, trigger = frames

            reference_pair = self._resolve_smt_reference_pair(pair, universe) if self.enable_smt_confirmation else None
            reference_trigger_frame = None
            if reference_pair is not None:
                ref_frames = pair_frames.get(reference_pair)
                if ref_frames is not None:
                    reference_trigger_frame = ref_frames[2]

            evaluation = self.evaluate_snapshot(
                pair,
                htf,
                ltf,
                trigger_frame=trigger,
                reference_pair=reference_pair,
                reference_trigger_frame=reference_trigger_frame,
                emit_logs=True,
            )
            if evaluation.signal is not None:
                candidates.append(SignalCandidate(pair=pair, signal=evaluation.signal, frame=ltf))

        kept, dropped = self.correlation_cap.filter(candidates)
        for drop in dropped:
            self._log_rejection(
                drop.pair,
                "correlation",
                drop.reason,
                kept_pair=drop.kept_pair,
                correlation=f"{drop.correlation:.4f}",
            )

        kept, exposure_drops = self._apply_currency_exposure_cap(kept)
        for drop in exposure_drops:
            self._log_rejection(drop.pair, drop.stage, drop.reason, **drop.context)

        kept, portfolio_drops = self._apply_portfolio_exposure_cap(kept)
        for drop in portfolio_drops:
            self._log_rejection(drop.pair, drop.stage, drop.reason, **drop.context)

        released: list[TradeSignal] = []
        for candidate in kept:
            allowed, drop = self.gate_signal_release(candidate.signal, commit=True)
            if not allowed and drop is not None:
                self._log_rejection(drop.pair, drop.stage, drop.reason, **drop.context)
                continue
            released.append(candidate.signal)
        
        # Trade Gate v2 check BEFORE release
        if self._trade_gate is not None:
            filtered: list[TradeSignal] = []
            for signal in released:
                result = self._trade_gate.check_trade(
                    pair=signal.symbol,
                    side=signal.side,
                    regime_output=None,  # Will auto-classify
                    universe=universe,
                    current_score=signal.score,
                )
                if not result.allowed:
                    self._log_rejection(
                        signal.symbol,
                        "trade_gate_v2",
                        result.reason,
                        regime_state=result.regime_state,
                        session_state=result.session_state,
                    )
                    continue
                filtered.append(signal)
            released = filtered
        
        return released
