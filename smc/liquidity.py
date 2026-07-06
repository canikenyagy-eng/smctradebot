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
    equal_level_tolerance_price: float = 0.0
    equal_level_tolerance_mode: str = "relative"


def pip_size(pair: str | None) -> float:
    clean = str(pair or "").upper().replace("/", "")
    return 0.01 if clean.endswith("JPY") else 0.0001


def _avg_range(frame: pd.DataFrame, lookback: int = 20) -> float:
    scoped = frame.tail(max(1, lookback))
    if scoped.empty:
        return 0.0
    return float((scoped["high"].astype(float) - scoped["low"].astype(float)).mean() or 0.0)


def _equal_level_relative(last_a: float, last_b: float, tolerance: float) -> bool:
    reference = max(abs(last_a), abs(last_b), 1e-9)
    return abs(last_a - last_b) / reference <= tolerance


def _equal_level_price(last_a: float, last_b: float, tolerance_price: float) -> bool:
    return abs(last_a - last_b) <= max(0.0, float(tolerance_price))


def _resolve_equal_level_tolerance(
    frame: pd.DataFrame,
    *,
    pair: str | None,
    tolerance_pips: float | None,
    atr_tolerance_factor: float,
    atr_lookback: int,
) -> tuple[float | None, str]:
    if pair is None and tolerance_pips is None and atr_tolerance_factor <= 0:
        return None, "relative"

    candidates: list[float] = []
    if tolerance_pips is not None:
        candidates.append(max(0.0, float(tolerance_pips)) * pip_size(pair))
    if atr_tolerance_factor > 0:
        candidates.append(_avg_range(frame, lookback=atr_lookback) * max(0.0, float(atr_tolerance_factor)))
    if not candidates:
        return None, "relative"
    return max(candidates), "pip_atr"


def _detect_equal_highs_lows(
    frame: pd.DataFrame,
    swing_window: int = 3,
    tolerance: float = 0.0006,
    *,
    pair: str | None = None,
    tolerance_pips: float | None = None,
    atr_tolerance_factor: float = 0.0,
    atr_lookback: int = 20,
) -> tuple[bool, bool, float | None, float | None, float, str]:
    swings = identify_swings(frame, window=swing_window)
    price_tolerance, tolerance_mode = _resolve_equal_level_tolerance(
        frame,
        pair=pair,
        tolerance_pips=tolerance_pips,
        atr_tolerance_factor=atr_tolerance_factor,
        atr_lookback=atr_lookback,
    )

    highs = swings.loc[swings["swing_high"], "high"]
    lows = swings.loc[swings["swing_low"], "low"]

    equal_highs = False
    equal_lows = False
    high_level = None
    low_level = None

    if len(highs) >= 2:
        high_prev = float(highs.iloc[-2])
        high_last = float(highs.iloc[-1])
        equal_highs = (
            _equal_level_price(high_prev, high_last, price_tolerance)
            if price_tolerance is not None
            else _equal_level_relative(high_prev, high_last, tolerance)
        )
        if equal_highs:
            high_level = (high_prev + high_last) / 2.0

    if len(lows) >= 2:
        low_prev = float(lows.iloc[-2])
        low_last = float(lows.iloc[-1])
        equal_lows = (
            _equal_level_price(low_prev, low_last, price_tolerance)
            if price_tolerance is not None
            else _equal_level_relative(low_prev, low_last, tolerance)
        )
        if equal_lows:
            low_level = (low_prev + low_last) / 2.0

    return equal_highs, equal_lows, high_level, low_level, float(price_tolerance or 0.0), tolerance_mode


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
    *,
    pair: str | None = None,
    tolerance_pips: float | None = None,
    atr_tolerance_factor: float = 0.0,
    atr_lookback: int = 20,
) -> LiquidityContext:
    equal_highs, equal_lows, high_level, low_level, tolerance_price, tolerance_mode = _detect_equal_highs_lows(
        frame,
        swing_window=swing_window,
        tolerance=tolerance,
        pair=pair,
        tolerance_pips=tolerance_pips,
        atr_tolerance_factor=atr_tolerance_factor,
        atr_lookback=atr_lookback,
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
        equal_level_tolerance_price=round(tolerance_price, 10),
        equal_level_tolerance_mode=tolerance_mode,
    )
