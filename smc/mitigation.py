from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

from smc.zones import PriceZone, assess_zone_lifecycle_as_of


@dataclass(frozen=True)
class MitigationState:
    zone: PriceZone
    touched: bool
    mitigated: bool
    reaction_direction: str
    reaction_strength: float
    touch_depth: float
    bars_since_touch: int | None
    touch_time: datetime | None
    summary: str


def _zone_touch_strength(zone: PriceZone, candle: pd.Series) -> tuple[float, float, str]:
    high = float(candle["high"])
    low = float(candle["low"])
    close = float(candle["close"])
    open_ = float(candle["open"])
    width = max(zone.width, 1e-9)

    if zone.direction == "bullish":
        depth = max(0.0, zone.upper - min(low, zone.upper)) / width
        close_position = (close - zone.lower) / width
        if low <= zone.upper and close >= zone.upper:
            return min(1.0, depth), min(100.0, 70.0 + (close - zone.upper) / width * 25.0), "bullish"
        if low <= zone.upper and close >= zone.midpoint:
            return min(1.0, depth), min(100.0, 50.0 + max(0.0, close_position) * 25.0), "bullish"
        if low <= zone.upper:
            return min(1.0, depth), min(100.0, 35.0 + max(0.0, close_position) * 15.0), "neutral"
        return 0.0, 0.0, "neutral"

    depth = max(0.0, max(high, zone.lower) - zone.lower) / width
    close_position = (zone.upper - close) / width
    if high >= zone.lower and close <= zone.lower:
        return min(1.0, depth), min(100.0, 70.0 + (zone.lower - close) / width * 25.0), "bearish"
    if high >= zone.lower and close <= zone.midpoint:
        return min(1.0, depth), min(100.0, 50.0 + max(0.0, close_position) * 25.0), "bearish"
    if high >= zone.lower:
        return min(1.0, depth), min(100.0, 35.0 + max(0.0, close_position) * 15.0), "neutral"
    return 0.0, 0.0, "neutral"


def evaluate_mitigation(
    frame: pd.DataFrame,
    zone: PriceZone,
    *,
    lookback: int = 50,
) -> MitigationState:
    if frame.empty:
        return MitigationState(
            zone=zone,
            touched=False,
            mitigated=False,
            reaction_direction="neutral",
            reaction_strength=0.0,
            touch_depth=0.0,
            bars_since_touch=None,
            touch_time=None,
            summary="Empty frame",
        )

    lifecycle = assess_zone_lifecycle_as_of(frame, zone, as_of_index=len(frame) - 1)
    if lifecycle.touch_count == 0 or lifecycle.last_touch_index is None:
        return MitigationState(
            zone=lifecycle,
            touched=False,
            mitigated=False,
            reaction_direction="neutral",
            reaction_strength=0.0,
            touch_depth=0.0,
            bars_since_touch=None,
            touch_time=None,
            summary="Zone untouched",
        )

    touch_index = lifecycle.last_touch_index
    start = max(0, touch_index - max(1, lookback))
    touch_frame = frame.iloc[start : touch_index + 1]
    candle = frame.iloc[touch_index]
    touch_depth, reaction_strength, reaction_direction = _zone_touch_strength(lifecycle, candle)

    if lifecycle.direction == "bullish":
        mitigated = float(candle["low"]) <= lifecycle.upper and float(candle["close"]) >= lifecycle.upper
    else:
        mitigated = float(candle["high"]) >= lifecycle.lower and float(candle["close"]) <= lifecycle.lower

    bars_since_touch = len(frame) - 1 - touch_index
    touch_time = frame.index[touch_index].to_pydatetime() if hasattr(frame.index[touch_index], "to_pydatetime") else None

    summary = "Mitigated" if mitigated else "Touched without full mitigation"
    if touch_frame.shape[0] > 1 and not mitigated:
        summary = f"{summary} over {touch_frame.shape[0]} bars"

    return MitigationState(
        zone=lifecycle,
        touched=True,
        mitigated=mitigated,
        reaction_direction=reaction_direction,
        reaction_strength=round(reaction_strength, 2),
        touch_depth=round(touch_depth, 4),
        bars_since_touch=bars_since_touch,
        touch_time=touch_time,
        summary=summary,
    )


def evaluate_mitigation_set(
    frame: pd.DataFrame,
    zones: Iterable[PriceZone],
    *,
    lookback: int = 50,
) -> list[MitigationState]:
    return [evaluate_mitigation(frame, zone, lookback=lookback) for zone in zones]
