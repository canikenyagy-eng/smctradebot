from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class LivePartialTarget:
    r_multiple: float
    fraction: float

    def sanitized(self) -> "LivePartialTarget":
        return LivePartialTarget(
            r_multiple=max(0.0, float(self.r_multiple)),
            fraction=_clamp(float(self.fraction), 0.0, 1.0),
        )


@dataclass(frozen=True)
class LiveExitProfile:
    target_rr: float
    partial_targets: tuple[LivePartialTarget, ...]
    break_even_r: float
    trailing_enabled: bool
    trailing_start_r: float
    trailing_lookback_bars: int
    time_stop_bars: int

    def sanitized(self) -> "LiveExitProfile":
        partials = tuple(
            item.sanitized()
            for item in sorted(self.partial_targets, key=lambda target: target.r_multiple)
            if item.r_multiple > 0 and item.fraction > 0
        )
        return LiveExitProfile(
            target_rr=max(0.1, float(self.target_rr)),
            partial_targets=partials,
            break_even_r=max(0.0, float(self.break_even_r)),
            trailing_enabled=bool(self.trailing_enabled),
            trailing_start_r=max(0.0, float(self.trailing_start_r)),
            trailing_lookback_bars=max(1, int(self.trailing_lookback_bars)),
            time_stop_bars=max(0, int(self.time_stop_bars)),
        )


@dataclass(frozen=True)
class LiveProfileSettings:
    enabled: bool = False
    preset: str = "default"
    use_regime_profiles: bool = True
    volatility_rr_enabled: bool = False
    volatility_rr_floor: float = 0.85
    volatility_rr_cap: float = 1.25
    liquidity_trailing_enabled: bool = False
    liquidity_lookback_bars: int = 8
    liquidity_buffer_pips: float = 1.0

    def sanitized(self) -> "LiveProfileSettings":
        floor = _clamp(float(self.volatility_rr_floor), 0.1, 5.0)
        cap = max(floor, _clamp(float(self.volatility_rr_cap), 0.1, 5.0))
        return LiveProfileSettings(
            enabled=bool(self.enabled),
            preset=str(self.preset or "default").strip().lower(),
            use_regime_profiles=bool(self.use_regime_profiles),
            volatility_rr_enabled=bool(self.volatility_rr_enabled),
            volatility_rr_floor=floor,
            volatility_rr_cap=cap,
            liquidity_trailing_enabled=bool(self.liquidity_trailing_enabled),
            liquidity_lookback_bars=max(2, int(self.liquidity_lookback_bars)),
            liquidity_buffer_pips=max(0.0, float(self.liquidity_buffer_pips)),
        )


@dataclass(frozen=True)
class LiveProfilePlan:
    enabled: bool
    preset: str
    regime_profile: str
    target_rr: float
    base_target_rr: float
    volatility_ratio: float
    partial_targets: tuple[LivePartialTarget, ...]
    break_even_r: float
    trailing_enabled: bool
    trailing_start_r: float
    trailing_lookback_bars: int
    time_stop_bars: int
    liquidity_trailing_enabled: bool
    liquidity_lookback_bars: int
    liquidity_buffer_pips: float

    def first_partial(self) -> LivePartialTarget | None:
        return self.partial_targets[0] if self.partial_targets else None

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "preset": self.preset,
            "regime_profile": self.regime_profile,
            "target_rr": round(self.target_rr, 4),
            "base_target_rr": round(self.base_target_rr, 4),
            "volatility_ratio": round(self.volatility_ratio, 4),
            "partials": [
                {"r": round(item.r_multiple, 4), "fraction": round(item.fraction, 4)}
                for item in self.partial_targets
            ],
            "break_even_r": round(self.break_even_r, 4),
            "trailing_enabled": self.trailing_enabled,
            "trailing_start_r": round(self.trailing_start_r, 4),
            "trailing_lookback_bars": self.trailing_lookback_bars,
            "time_stop_bars": self.time_stop_bars,
            "liquidity_trailing_enabled": self.liquidity_trailing_enabled,
            "liquidity_lookback_bars": self.liquidity_lookback_bars,
            "liquidity_buffer_pips": round(self.liquidity_buffer_pips, 4),
        }


def m15_vol_liq_v1_profiles() -> dict[str, LiveExitProfile]:
    balanced_trend = LiveExitProfile(
        target_rr=2.4,
        partial_targets=(LivePartialTarget(1.0, 0.35), LivePartialTarget(1.8, 0.25)),
        break_even_r=1.1,
        trailing_enabled=True,
        trailing_start_r=1.5,
        trailing_lookback_bars=8,
        time_stop_bars=60,
    ).sanitized()
    balanced_range = LiveExitProfile(
        target_rr=1.35,
        partial_targets=(LivePartialTarget(0.8, 0.55),),
        break_even_r=0.8,
        trailing_enabled=True,
        trailing_start_r=1.1,
        trailing_lookback_bars=5,
        time_stop_bars=24,
    ).sanitized()
    runner_expansion = LiveExitProfile(
        target_rr=3.4,
        partial_targets=(LivePartialTarget(1.2, 0.30), LivePartialTarget(2.2, 0.20)),
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


def live_profiles_for_preset(preset: str) -> dict[str, LiveExitProfile]:
    if str(preset or "default").strip().lower() == "m15_vol_liq_v1":
        return m15_vol_liq_v1_profiles()
    return {"neutral": LiveExitProfile(2.0, (LivePartialTarget(1.0, 0.50),), 1.0, True, 1.5, 6, 48).sanitized()}


class LiveProfileSelector:
    def __init__(self, settings: LiveProfileSettings | None = None) -> None:
        self.settings = (settings or LiveProfileSettings()).sanitized()
        self._profiles = live_profiles_for_preset(self.settings.preset)

    def _select_profile(self, regime_label: str) -> tuple[str, LiveExitProfile]:
        regime = (regime_label or "neutral").strip().lower()
        if not self.settings.use_regime_profiles:
            regime = "neutral"
        if regime in self._profiles:
            return regime, self._profiles[regime]
        if regime in {"trend", "expansion"} and "trend" in self._profiles:
            return "trend", self._profiles["trend"]
        if regime in {"range", "contraction"} and "range" in self._profiles:
            return "range", self._profiles["range"]
        return "neutral", self._profiles.get("neutral", next(iter(self._profiles.values())))

    def build_plan(self, *, regime_label: str, fallback_rr: float, volatility_ratio: float = 1.0) -> LiveProfilePlan:
        if not self.settings.enabled:
            fallback = max(0.1, float(fallback_rr))
            return LiveProfilePlan(
                enabled=False,
                preset="disabled",
                regime_profile="fallback",
                target_rr=fallback,
                base_target_rr=fallback,
                volatility_ratio=1.0,
                partial_targets=(),
                break_even_r=0.0,
                trailing_enabled=False,
                trailing_start_r=0.0,
                trailing_lookback_bars=1,
                time_stop_bars=0,
                liquidity_trailing_enabled=False,
                liquidity_lookback_bars=2,
                liquidity_buffer_pips=0.0,
            )

        profile_name, profile = self._select_profile(regime_label)
        rr = profile.target_rr
        ratio = 1.0
        if self.settings.volatility_rr_enabled:
            ratio = _clamp(float(volatility_ratio), self.settings.volatility_rr_floor, self.settings.volatility_rr_cap)
            rr = max(0.1, rr * ratio)

        return LiveProfilePlan(
            enabled=True,
            preset=self.settings.preset,
            regime_profile=profile_name,
            target_rr=rr,
            base_target_rr=profile.target_rr,
            volatility_ratio=ratio,
            partial_targets=profile.partial_targets,
            break_even_r=profile.break_even_r,
            trailing_enabled=profile.trailing_enabled,
            trailing_start_r=profile.trailing_start_r,
            trailing_lookback_bars=profile.trailing_lookback_bars,
            time_stop_bars=profile.time_stop_bars,
            liquidity_trailing_enabled=self.settings.liquidity_trailing_enabled,
            liquidity_lookback_bars=self.settings.liquidity_lookback_bars,
            liquidity_buffer_pips=self.settings.liquidity_buffer_pips,
        )
