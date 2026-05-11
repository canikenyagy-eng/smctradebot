from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from smc.structure import identify_swings, infer_trend


@dataclass(frozen=True)
class SMTDivergence:
    target_pair: str
    reference_pair: str
    direction: str
    kind: str
    target_level: float | None
    reference_level: float | None
    target_prev_level: float | None
    reference_prev_level: float | None
    target_trend: str
    reference_trend: str
    strength: float
    detected_at: datetime | None
    description: str


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for SMT detection: {', '.join(sorted(missing))}")


def _last_two(values: pd.Series) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    return float(values.iloc[-2]), float(values.iloc[-1])


def _extreme_levels(frame: pd.DataFrame, column: str, highest: bool, lookback: int) -> tuple[float | None, float | None]:
    scoped = frame.tail(lookback)
    if len(scoped) < 2:
        return None, None

    values = scoped[column].astype(float)
    selected = values.nlargest(2) if highest else values.nsmallest(2)
    if len(selected) < 2:
        return None, None

    selected = selected.sort_index()
    return float(selected.iloc[0]), float(selected.iloc[1])


def _levels_with_fallback(
    frame: pd.DataFrame,
    window: int,
    lookback: int,
) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None], str]:
    scoped = frame.tail(lookback)
    swings = identify_swings(scoped, window=window)
    highs = swings.loc[swings["swing_high"], "high"]
    lows = swings.loc[swings["swing_low"], "low"]
    if len(highs) >= 2 and len(lows) >= 2:
        return _last_two(highs), _last_two(lows), infer_trend(swings)

    high_levels = _last_two(highs)
    low_levels = _last_two(lows)
    if high_levels == (None, None):
        high_levels = _extreme_levels(scoped, "high", highest=True, lookback=lookback)
    if low_levels == (None, None):
        low_levels = _extreme_levels(scoped, "low", highest=False, lookback=lookback)

    if scoped.empty:
        trend = "neutral"
    else:
        first_close = float(scoped["close"].iloc[0])
        last_close = float(scoped["close"].iloc[-1])
        if last_close > first_close:
            trend = "bullish"
        elif last_close < first_close:
            trend = "bearish"
        else:
            trend = "neutral"

    return high_levels, low_levels, trend


def _range_scale(frame: pd.DataFrame, lookback: int) -> float:
    scoped = frame.tail(lookback)
    if scoped.empty:
        return 1.0
    return float((scoped["high"].astype(float) - scoped["low"].astype(float)).mean() or 1.0)


def detect_smt_divergence(
    target_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    *,
    target_pair: str = "TARGET",
    reference_pair: str = "REFERENCE",
    swing_window: int = 3,
    lookback: int = 80,
) -> SMTDivergence | None:
    if target_frame.empty or reference_frame.empty:
        return None

    _validate_frame(target_frame)
    _validate_frame(reference_frame)

    target_highs, target_lows, target_trend = _levels_with_fallback(target_frame, swing_window, lookback)
    reference_highs, reference_lows, reference_trend = _levels_with_fallback(reference_frame, swing_window, lookback)

    target_prev_high, target_last_high = target_highs
    target_prev_low, target_last_low = target_lows
    ref_prev_high, ref_last_high = reference_highs
    ref_prev_low, ref_last_low = reference_lows

    target_scale = _range_scale(target_frame, lookback)
    reference_scale = _range_scale(reference_frame, lookback)
    scale = max(target_scale, reference_scale, 1e-9)
    detected_at = target_frame.index[-1].to_pydatetime() if hasattr(target_frame.index[-1], "to_pydatetime") else None

    if (
        target_prev_high is not None
        and target_last_high is not None
        and ref_prev_high is not None
        and ref_last_high is not None
        and target_last_high > target_prev_high
        and ref_last_high <= ref_prev_high
    ):
        magnitude = (target_last_high - target_prev_high) / scale
        peer_resilience = max(0.0, (ref_prev_high - ref_last_high) / scale)
        strength = min(100.0, 55.0 + magnitude * 35.0 + peer_resilience * 20.0)
        return SMTDivergence(
            target_pair=target_pair,
            reference_pair=reference_pair,
            direction="bearish",
            kind="bearish_smt",
            target_level=target_last_high,
            reference_level=ref_last_high,
            target_prev_level=target_prev_high,
            reference_prev_level=ref_prev_high,
            target_trend=target_trend.upper(),
            reference_trend=reference_trend.upper(),
            strength=round(strength, 2),
            detected_at=detected_at,
            description=f"{target_pair} made a higher high while {reference_pair} failed to confirm",
        )

    if (
        target_prev_low is not None
        and target_last_low is not None
        and ref_prev_low is not None
        and ref_last_low is not None
        and target_last_low < target_prev_low
        and ref_last_low >= ref_prev_low
    ):
        magnitude = (target_prev_low - target_last_low) / scale
        peer_resilience = max(0.0, (ref_last_low - ref_prev_low) / scale)
        strength = min(100.0, 55.0 + magnitude * 35.0 + peer_resilience * 20.0)
        return SMTDivergence(
            target_pair=target_pair,
            reference_pair=reference_pair,
            direction="bullish",
            kind="bullish_smt",
            target_level=target_last_low,
            reference_level=ref_last_low,
            target_prev_level=target_prev_low,
            reference_prev_level=ref_prev_low,
            target_trend=target_trend.upper(),
            reference_trend=reference_trend.upper(),
            strength=round(strength, 2),
            detected_at=detected_at,
            description=f"{target_pair} made a lower low while {reference_pair} held above the prior low",
        )

    return None
