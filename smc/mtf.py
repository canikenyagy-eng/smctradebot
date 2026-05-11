from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smc.structure import identify_swings, infer_trend


@dataclass(frozen=True)
class MTFContext:
    bias: str
    zone: str
    midpoint: float
    range_high: float
    range_low: float


def analyze_htf_bias(htf_frame: pd.DataFrame, swing_window: int = 3) -> str:
    swings = identify_swings(htf_frame, window=swing_window)
    return infer_trend(swings)


def premium_discount_context(
    htf_frame: pd.DataFrame,
    current_price: float,
    swing_window: int = 3,
    range_lookback: int = 120,
) -> MTFContext:
    if htf_frame.empty:
        raise ValueError("HTF frame is empty")

    bias = analyze_htf_bias(htf_frame, swing_window=swing_window)
    scoped = htf_frame.tail(range_lookback)

    range_high = float(scoped["high"].max())
    range_low = float(scoped["low"].min())
    midpoint = (range_high + range_low) / 2.0

    if current_price < midpoint:
        zone = "discount"
    elif current_price > midpoint:
        zone = "premium"
    else:
        zone = "equilibrium"

    return MTFContext(
        bias=bias,
        zone=zone,
        midpoint=midpoint,
        range_high=range_high,
        range_low=range_low,
    )


def zone_supports_direction(zone: str, target_direction: str) -> float:
    """
    Return compatibility score (0.0-1.0) between zone and target direction.
    This is a FEATURE only - no decision logic.
    
    Args:
        zone: premium/discount (zone type)
        target_direction: bullish/bearish (trade direction)
    
    Returns:
        compatibility score: 1.0 = compatible, 0.0 = incompatible
    """
    zone_u = zone.upper()
    direction_u = target_direction.upper()
    
    # Feature: zone compatibility with direction
    if direction_u == "BULLISH":
        return 1.0 if zone_u == "DISCOUNT" else 0.0
    if direction_u == "BEARISH":
        return 1.0 if zone_u == "PREMIUM" else 0.0
    return 0.0  # unknown direction
