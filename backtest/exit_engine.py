from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from backtest.risk import atr_value_at
from core.signal_engine import TradeSignal


def _pip_size(pair: str) -> float:
    normalized = pair.upper().replace("/", "")
    quote = normalized[3:6] if len(normalized) >= 6 else ""
    return 0.01 if quote == "JPY" else 0.0001


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class PartialRTarget:
    r_multiple: float
    fraction: float

    def sanitized(self) -> "PartialRTarget":
        return PartialRTarget(
            r_multiple=max(0.0, float(self.r_multiple)),
            fraction=_clamp(float(self.fraction), 0.0, 1.0),
        )


@dataclass(frozen=True)
class ExitProfile:
    target_rr: float
    partial_targets: tuple[PartialRTarget, ...]
    break_even_r: float
    trailing_enabled: bool
    trailing_start_r: float
    trailing_lookback_bars: int
    time_stop_bars: int

    def sanitized(self) -> "ExitProfile":
        targets = [target.sanitized() for target in self.partial_targets]
        targets = [target for target in targets if target.r_multiple > 0 and target.fraction > 0]
        targets.sort(key=lambda item: item.r_multiple)

        normalized_targets: list[PartialRTarget] = []
        total_fraction = sum(item.fraction for item in targets)
        if total_fraction > 1.0 and total_fraction > 0:
            for target in targets:
                normalized_targets.append(
                    PartialRTarget(
                        r_multiple=target.r_multiple,
                        fraction=target.fraction / total_fraction,
                    )
                )
        else:
            normalized_targets = targets

        return ExitProfile(
            target_rr=max(0.1, float(self.target_rr)),
            partial_targets=tuple(normalized_targets),
            break_even_r=max(0.0, float(self.break_even_r)),
            trailing_enabled=bool(self.trailing_enabled),
            trailing_start_r=max(0.0, float(self.trailing_start_r)),
            trailing_lookback_bars=max(1, int(self.trailing_lookback_bars)),
            time_stop_bars=max(0, int(self.time_stop_bars)),
        )


@dataclass(frozen=True)
class AdaptiveExitSettings:
    enabled: bool = False
    profile_preset: str = "default"
    use_regime_profiles: bool = True
    regime_profiles: dict[str, ExitProfile] = field(default_factory=dict)
    profile_overrides: dict[str, dict[str, object]] | None = None
    atr_trailing_enabled: bool = False
    atr_trailing_period: int = 14
    atr_trailing_multiplier: float = 1.5
    liquidity_trailing_enabled: bool = False
    liquidity_lookback_bars: int = 8
    liquidity_buffer_pips: float = 1.0
    volatility_rr_enabled: bool = False
    volatility_rr_floor: float = 0.85
    volatility_rr_cap: float = 1.25

    def sanitized(self) -> "AdaptiveExitSettings":
        profile_preset = str(self.profile_preset or "default").strip().lower()
        profiles = self.regime_profiles or exit_profiles_for_preset(profile_preset)
        parsed_profiles = {key.lower(): value.sanitized() for key, value in profiles.items()}
        if self.profile_overrides:
            parsed_profiles = _apply_profile_overrides(parsed_profiles, self.profile_overrides)
        return AdaptiveExitSettings(
            enabled=bool(self.enabled),
            profile_preset=profile_preset,
            use_regime_profiles=bool(self.use_regime_profiles),
            regime_profiles=parsed_profiles,
            profile_overrides=self.profile_overrides,
            atr_trailing_enabled=bool(self.atr_trailing_enabled),
            atr_trailing_period=max(2, int(self.atr_trailing_period)),
            atr_trailing_multiplier=max(0.1, float(self.atr_trailing_multiplier)),
            liquidity_trailing_enabled=bool(self.liquidity_trailing_enabled),
            liquidity_lookback_bars=max(2, int(self.liquidity_lookback_bars)),
            liquidity_buffer_pips=max(0.0, float(self.liquidity_buffer_pips)),
            volatility_rr_enabled=bool(self.volatility_rr_enabled),
            volatility_rr_floor=_clamp(float(self.volatility_rr_floor), 0.1, 5.0),
            volatility_rr_cap=max(
                _clamp(float(self.volatility_rr_floor), 0.1, 5.0),
                _clamp(float(self.volatility_rr_cap), 0.1, 5.0),
            ),
        )


@dataclass(frozen=True)
class ExitPartialTarget:
    price: float
    fraction: float
    label: str


@dataclass(frozen=True)
class ExitPlan:
    mode: str
    profile: str
    take_profit: float
    partial_targets: tuple[ExitPartialTarget, ...]
    break_even_r: float
    trailing_enabled: bool
    trailing_start_r: float
    trailing_lookback_bars: int
    time_stop_bars: int
    atr_trailing_enabled: bool
    atr_trailing_period: int
    atr_trailing_multiplier: float
    liquidity_trailing_enabled: bool
    liquidity_lookback_bars: int
    liquidity_buffer_pips: float
    target_rr: float


def default_regime_profiles() -> dict[str, ExitProfile]:
    return {
        "trend": ExitProfile(
            target_rr=2.8,
            partial_targets=(PartialRTarget(1.0, 0.40), PartialRTarget(2.0, 0.30)),
            break_even_r=1.0,
            trailing_enabled=True,
            trailing_start_r=1.4,
            trailing_lookback_bars=10,
            time_stop_bars=72,
        ).sanitized(),
        "expansion": ExitProfile(
            target_rr=3.2,
            partial_targets=(PartialRTarget(1.0, 0.35), PartialRTarget(2.2, 0.30)),
            break_even_r=1.1,
            trailing_enabled=True,
            trailing_start_r=1.5,
            trailing_lookback_bars=12,
            time_stop_bars=84,
        ).sanitized(),
        "range": ExitProfile(
            target_rr=1.5,
            partial_targets=(PartialRTarget(0.8, 0.60),),
            break_even_r=0.7,
            trailing_enabled=True,
            trailing_start_r=1.0,
            trailing_lookback_bars=5,
            time_stop_bars=28,
        ).sanitized(),
        "contraction": ExitProfile(
            target_rr=1.3,
            partial_targets=(PartialRTarget(0.7, 0.65),),
            break_even_r=0.6,
            trailing_enabled=True,
            trailing_start_r=0.9,
            trailing_lookback_bars=4,
            time_stop_bars=22,
        ).sanitized(),
        "neutral": ExitProfile(
            target_rr=2.0,
            partial_targets=(PartialRTarget(1.0, 0.50),),
            break_even_r=1.0,
            trailing_enabled=True,
            trailing_start_r=1.5,
            trailing_lookback_bars=6,
            time_stop_bars=48,
        ).sanitized(),
    }


def m15_vol_liq_v1_profiles() -> dict[str, ExitProfile]:
    balanced_trend = ExitProfile(
        target_rr=2.4,
        partial_targets=(PartialRTarget(1.0, 0.35), PartialRTarget(1.8, 0.25)),
        break_even_r=1.1,
        trailing_enabled=True,
        trailing_start_r=1.5,
        trailing_lookback_bars=8,
        time_stop_bars=60,
    ).sanitized()
    balanced_range = ExitProfile(
        target_rr=1.35,
        partial_targets=(PartialRTarget(0.8, 0.55),),
        break_even_r=0.8,
        trailing_enabled=True,
        trailing_start_r=1.1,
        trailing_lookback_bars=5,
        time_stop_bars=24,
    ).sanitized()
    runner_expansion = ExitProfile(
        target_rr=3.4,
        partial_targets=(PartialRTarget(1.2, 0.30), PartialRTarget(2.2, 0.20)),
        break_even_r=1.35,
        trailing_enabled=True,
        trailing_start_r=1.8,
        trailing_lookback_bars=12,
        time_stop_bars=84,
    ).sanitized()
    return {
        "trend": balanced_trend,
        "expansion": runner_expansion,
        "range": balanced_range,
        "contraction": balanced_range,
        "neutral": balanced_range,
    }


def exit_profiles_for_preset(preset: str) -> dict[str, ExitProfile]:
    key = str(preset or "default").strip().lower()
    if key == "m15_vol_liq_v1":
        return m15_vol_liq_v1_profiles()
    return default_regime_profiles()


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_partial_targets(raw_targets: Any, fallback: tuple[PartialRTarget, ...]) -> tuple[PartialRTarget, ...]:
    if not isinstance(raw_targets, (list, tuple)):
        return fallback

    parsed: list[PartialRTarget] = []
    for raw in raw_targets:
        if isinstance(raw, dict):
            r_multiple = _as_float(raw.get("r"), 0.0)
            fraction = _as_float(raw.get("fraction"), 0.0)
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            r_multiple = _as_float(raw[0], 0.0)
            fraction = _as_float(raw[1], 0.0)
        else:
            continue
        target = PartialRTarget(r_multiple=r_multiple, fraction=fraction).sanitized()
        if target.r_multiple > 0 and target.fraction > 0:
            parsed.append(target)

    if not parsed:
        return fallback
    parsed.sort(key=lambda item: item.r_multiple)
    return tuple(parsed)


def _build_profile_from_mapping(base: ExitProfile, raw: Mapping[str, object]) -> ExitProfile:
    return ExitProfile(
        target_rr=max(0.1, _as_float(raw.get("target_rr"), base.target_rr)),
        partial_targets=_parse_partial_targets(raw.get("partials"), base.partial_targets),
        break_even_r=max(0.0, _as_float(raw.get("break_even_r"), base.break_even_r)),
        trailing_enabled=bool(raw.get("trailing_enabled", base.trailing_enabled)),
        trailing_start_r=max(0.0, _as_float(raw.get("trailing_start_r"), base.trailing_start_r)),
        trailing_lookback_bars=max(1, _as_int(raw.get("trailing_lookback_bars"), base.trailing_lookback_bars)),
        time_stop_bars=max(0, _as_int(raw.get("time_stop_bars"), base.time_stop_bars)),
    ).sanitized()


def _apply_profile_overrides(
    profiles: dict[str, ExitProfile],
    overrides: Mapping[str, dict[str, object]],
) -> dict[str, ExitProfile]:
    result = dict(profiles)
    for key, raw in overrides.items():
        if not isinstance(raw, Mapping):
            continue
        regime = str(key).strip().lower()
        if not regime:
            continue
        base = result.get(regime, result.get("neutral", ExitProfile(
            target_rr=2.0,
            partial_targets=(PartialRTarget(1.0, 0.5),),
            break_even_r=1.0,
            trailing_enabled=True,
            trailing_start_r=1.5,
            trailing_lookback_bars=6,
            time_stop_bars=48,
        )))
        result[regime] = _build_profile_from_mapping(base, raw)
    return result


class AdaptiveExitEngine:
    def __init__(self, settings: AdaptiveExitSettings | None = None) -> None:
        self.settings = (settings or AdaptiveExitSettings()).sanitized()
        self._profiles = self.settings.regime_profiles or default_regime_profiles()

    def _select_profile(self, regime_label: str) -> tuple[str, ExitProfile]:
        regime = regime_label.strip().lower()
        if not regime:
            regime = "neutral"

        if regime in self._profiles:
            return regime, self._profiles[regime]
        if regime in {"trend", "expansion"} and "trend" in self._profiles:
            return "trend", self._profiles["trend"]
        if regime in {"range", "contraction"} and "range" in self._profiles:
            return "range", self._profiles["range"]
        return "neutral", self._profiles.get("neutral", next(iter(self._profiles.values())))

    def build_plan(
        self,
        *,
        pair: str,
        signal: TradeSignal,
        entry: float,
        stop_loss: float,
        take_profit: float,
        risk: float,
        volatility_ratio: float = 1.0,
    ) -> ExitPlan:
        if not self.settings.enabled or risk <= 0:
            partial_targets: list[ExitPartialTarget] = []
            if signal.partial_take_profit is not None and signal.partial_take_fraction > 0:
                partial_targets.append(
                    ExitPartialTarget(
                        price=float(signal.partial_take_profit),
                        fraction=_clamp(signal.partial_take_fraction, 0.0, 0.95),
                        label="legacy_tp1",
                    )
                )
            target_rr = abs(take_profit - entry) / max(risk, 1e-9)
            return ExitPlan(
                mode="legacy",
                profile="legacy",
                take_profit=float(take_profit),
                partial_targets=tuple(partial_targets),
                break_even_r=max(0.0, float(signal.break_even_r)),
                trailing_enabled=bool(signal.trailing_enabled),
                trailing_start_r=max(0.0, float(signal.trailing_start_r)),
                trailing_lookback_bars=max(1, int(signal.trailing_lookback_bars)),
                time_stop_bars=max(0, int(signal.time_stop_bars)),
                atr_trailing_enabled=False,
                atr_trailing_period=max(2, int(self.settings.atr_trailing_period)),
                atr_trailing_multiplier=max(0.1, float(self.settings.atr_trailing_multiplier)),
                liquidity_trailing_enabled=False,
                liquidity_lookback_bars=max(2, int(self.settings.liquidity_lookback_bars)),
                liquidity_buffer_pips=max(0.0, float(self.settings.liquidity_buffer_pips)),
                target_rr=float(target_rr),
            )

        profile_name, profile = self._select_profile(signal.regime_label if self.settings.use_regime_profiles else "neutral")

        target_rr = profile.target_rr
        if self.settings.volatility_rr_enabled:
            rr_adjust = _clamp(volatility_ratio, self.settings.volatility_rr_floor, self.settings.volatility_rr_cap)
            target_rr = max(0.1, target_rr * rr_adjust)

        if signal.side == "BUY":
            adaptive_take_profit = entry + risk * target_rr
        else:
            adaptive_take_profit = entry - risk * target_rr

        partial_targets: list[ExitPartialTarget] = []
        for index, partial in enumerate(profile.partial_targets, start=1):
            if signal.side == "BUY":
                price = entry + risk * partial.r_multiple
                if price >= adaptive_take_profit:
                    continue
            else:
                price = entry - risk * partial.r_multiple
                if price <= adaptive_take_profit:
                    continue
            partial_targets.append(
                ExitPartialTarget(
                    price=float(price),
                    fraction=_clamp(partial.fraction, 0.0, 1.0),
                    label=f"profile_tp{index}",
                )
            )

        return ExitPlan(
            mode="adaptive",
            profile=profile_name,
            take_profit=float(adaptive_take_profit),
            partial_targets=tuple(partial_targets),
            break_even_r=max(0.0, float(profile.break_even_r)),
            trailing_enabled=bool(profile.trailing_enabled),
            trailing_start_r=max(0.0, float(profile.trailing_start_r)),
            trailing_lookback_bars=max(1, int(profile.trailing_lookback_bars)),
            time_stop_bars=max(0, int(profile.time_stop_bars)),
            atr_trailing_enabled=bool(self.settings.atr_trailing_enabled),
            atr_trailing_period=max(2, int(self.settings.atr_trailing_period)),
            atr_trailing_multiplier=max(0.1, float(self.settings.atr_trailing_multiplier)),
            liquidity_trailing_enabled=bool(self.settings.liquidity_trailing_enabled),
            liquidity_lookback_bars=max(2, int(self.settings.liquidity_lookback_bars)),
            liquidity_buffer_pips=max(0.0, float(self.settings.liquidity_buffer_pips)),
            target_rr=float(target_rr),
        )

    def trailing_stop_candidate(
        self,
        *,
        pair: str,
        side: str,
        frame: pd.DataFrame,
        fill_index: int,
        index: int,
        current_stop: float,
        lookback_bars: int,
        atr_enabled: bool,
        atr_period: int,
        atr_multiplier: float,
        liquidity_enabled: bool,
        liquidity_lookback_bars: int,
        liquidity_buffer_pips: float,
    ) -> tuple[float, bool, bool]:
        start = max(fill_index, index - max(1, lookback_bars) + 1)
        trailing_slice = frame.iloc[start : index + 1]
        if trailing_slice.empty:
            return current_stop, False, False

        if side == "BUY":
            candidates: list[tuple[str, float]] = [("structural", float(trailing_slice["low"].min()))]
        else:
            candidates = [("structural", float(trailing_slice["high"].max()))]

        if atr_enabled:
            atr_val = atr_value_at(frame, index, period=atr_period)
            if atr_val is not None and atr_val > 0:
                close = float(frame.iloc[index]["close"])
                if side == "BUY":
                    candidates.append(("atr", close - atr_val * atr_multiplier))
                else:
                    candidates.append(("atr", close + atr_val * atr_multiplier))

        if liquidity_enabled:
            liq_start = max(fill_index, index - max(2, liquidity_lookback_bars) + 1)
            liq_slice = frame.iloc[liq_start : index + 1]
            if not liq_slice.empty:
                buffer = _pip_size(pair) * max(0.0, float(liquidity_buffer_pips))
                if side == "BUY":
                    lows = sorted(float(value) for value in liq_slice["low"].tolist())
                    anchor = lows[1] if len(lows) > 1 else lows[0]
                    candidates.append(("liquidity", anchor - buffer))
                else:
                    highs = sorted((float(value) for value in liq_slice["high"].tolist()), reverse=True)
                    anchor = highs[1] if len(highs) > 1 else highs[0]
                    candidates.append(("liquidity", anchor + buffer))

        if side == "BUY":
            source, candidate = max(candidates, key=lambda item: item[1])
            if candidate > current_stop:
                return candidate, source == "atr", source == "liquidity"
            return current_stop, False, False

        source, candidate = min(candidates, key=lambda item: item[1])
        if candidate < current_stop:
            return candidate, source == "atr", source == "liquidity"
        return current_stop, False, False
