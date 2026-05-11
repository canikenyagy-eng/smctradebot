from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smc.structure import identify_swings


@dataclass(frozen=True)
class LiquidityContext:
    equal_highs: bool
    equal_lows: bool
    equal_high_level: float | None
    equal_low_level: float | None
    sweep: bool
    sweep_direction: str | None
    displacement: bool
    displacement_direction: str | None


def _equal_level(last_a: float, last_b: float, tolerance: float) -> bool:
    reference = max(abs(last_a), abs(last_b), 1e-9)
    return abs(last_a - last_b) / reference <= tolerance


def _detect_equal_highs_lows(
    frame: pd.DataFrame,
    swing_window: int = 3,
    tolerance: float = 0.0006,
) -> tuple[bool, bool, float | None, float | None]:
    swings = identify_swings(frame, window=swing_window)

    highs = swings.loc[swings["swing_high"], "high"]
    lows = swings.loc[swings["swing_low"], "low"]

    equal_highs = False
    equal_lows = False
    high_level = None
    low_level = None

    if len(highs) >= 2:
        high_prev = float(highs.iloc[-2])
        high_last = float(highs.iloc[-1])
        equal_highs = _equal_level(high_prev, high_last, tolerance)
        if equal_highs:
            high_level = (high_prev + high_last) / 2.0

    if len(lows) >= 2:
        low_prev = float(lows.iloc[-2])
        low_last = float(lows.iloc[-1])
        equal_lows = _equal_level(low_prev, low_last, tolerance)
        if equal_lows:
            low_level = (low_prev + low_last) / 2.0

    return equal_highs, equal_lows, high_level, low_level


def _detect_sweep(
    frame: pd.DataFrame,
    equal_high_level: float | None,
    equal_low_level: float | None,
) -> tuple[bool, str | None]:
    if len(frame) < 2:
        return False, None

    candle = frame.iloc[-1]
    sweep = False
    direction = None

    if equal_high_level is not None:
        if float(candle["high"]) > equal_high_level and float(candle["close"]) < equal_high_level:
            sweep = True
            direction = "bearish"

    if equal_low_level is not None:
        if float(candle["low"]) < equal_low_level and float(candle["close"]) > equal_low_level:
            sweep = True
            direction = "bullish"

    return sweep, direction


def _detect_displacement(
    frame: pd.DataFrame,
    body_multiplier: float = 1.8,
    range_multiplier: float = 1.5,
    lookback: int = 20,
) -> tuple[bool, str | None]:
    if len(frame) < lookback + 2:
        return False, None

    scoped = frame.tail(lookback + 1)
    historical = scoped.iloc[:-1]
    candle = scoped.iloc[-1]

    avg_body = (historical["close"] - historical["open"]).abs().mean()
    avg_range = (historical["high"] - historical["low"]).abs().mean()

    body = abs(float(candle["close"]) - float(candle["open"]))
    full_range = float(candle["high"]) - float(candle["low"])

    if avg_body <= 0 or avg_range <= 0:
        return False, None

    displacement = body >= body_multiplier * float(avg_body) and full_range >= range_multiplier * float(avg_range)
    if not displacement:
        return False, None

    direction = "bullish" if float(candle["close"]) > float(candle["open"]) else "bearish"
    return True, direction


def analyze_liquidity(
    frame: pd.DataFrame,
    swing_window: int = 3,
    tolerance: float = 0.0006,
) -> LiquidityContext:
    equal_highs, equal_lows, high_level, low_level = _detect_equal_highs_lows(
        frame,
        swing_window=swing_window,
        tolerance=tolerance,
    )
    sweep, sweep_direction = _detect_sweep(frame, high_level, low_level)
    displacement, displacement_direction = _detect_displacement(frame)

    return LiquidityContext(
        equal_highs=equal_highs,
        equal_lows=equal_lows,
        equal_high_level=high_level,
        equal_low_level=low_level,
        sweep=sweep,
        sweep_direction=sweep_direction,
        displacement=displacement,
        displacement_direction=displacement_direction,
    )
