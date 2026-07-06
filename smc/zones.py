from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class PriceZone:
    kind: str
    direction: str
    lower: float
    upper: float
    created_at: datetime | None = None
    created_index: int | None = None
    source_index: int | None = None
    strength: float = 0.0
    touch_count: int = 0
    fill_ratio: float = 0.0
    is_fresh: bool = True
    invalidated: bool = False
    last_touch_at: datetime | None = None
    last_touch_index: int | None = None

    def __post_init__(self) -> None:
        lower = float(min(self.lower, self.upper))
        upper = float(max(self.lower, self.upper))
        object.__setattr__(self, "kind", self.kind.strip().lower())
        object.__setattr__(self, "direction", self.direction.strip().lower())
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "strength", float(self.strength))
        object.__setattr__(self, "fill_ratio", max(0.0, min(1.0, float(self.fill_ratio))))

    @property
    def width(self) -> float:
        return max(0.0, self.upper - self.lower)

    @property
    def midpoint(self) -> float:
        return (self.upper + self.lower) / 2.0

    def contains(self, price: float) -> bool:
        return self.lower <= float(price) <= self.upper

    def intersects(self, high: float, low: float) -> bool:
        return float(low) <= self.upper and float(high) >= self.lower


def _resolved_start_index(frame: pd.DataFrame, start_index: int | None) -> int:
    if frame.empty:
        return -1

    if start_index is None:
        return -1

    if start_index < 0:
        return -1

    return min(start_index, len(frame) - 1)


def assess_zone_lifecycle_as_of(
    frame: pd.DataFrame,
    zone: PriceZone,
    *,
    start_index: int | None = None,
    as_of_index: int | None = None,
) -> PriceZone:
    if frame.empty:
        return zone

    start_pos = _resolved_start_index(frame, start_index if start_index is not None else zone.created_index)
    end_pos = len(frame) - 1 if as_of_index is None else min(max(int(as_of_index), start_pos), len(frame) - 1)
    future = frame.iloc[start_pos + 1 : end_pos + 1]
    if future.empty:
        return zone

    touch_count = 0
    last_touch_index: int | None = None
    last_touch_at: datetime | None = None
    deepest_penetration = 0.0

    for pos, (timestamp, row) in enumerate(future.iterrows(), start=start_pos + 1):
        high = float(row["high"])
        low = float(row["low"])
        if not zone.intersects(high, low):
            continue

        touch_count += 1
        last_touch_index = pos
        last_touch_at = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else None

        if zone.direction == "bullish":
            penetration = max(0.0, zone.upper - min(low, zone.upper)) / max(zone.width, 1e-9)
        else:
            penetration = max(0.0, max(high, zone.lower) - zone.lower) / max(zone.width, 1e-9)
        deepest_penetration = max(deepest_penetration, min(1.0, penetration))

    return replace(
        zone,
        touch_count=touch_count,
        fill_ratio=round(deepest_penetration, 4),
        is_fresh=touch_count == 0,
        invalidated=deepest_penetration >= 1.0,
        last_touch_index=last_touch_index,
        last_touch_at=last_touch_at,
    )


def assess_zone_lifecycle(
    frame: pd.DataFrame,
    zone: PriceZone,
    *,
    start_index: int | None = None,
) -> PriceZone:
    return assess_zone_lifecycle_as_of(
        frame,
        zone,
        start_index=start_index,
        as_of_index=len(frame) - 1 if not frame.empty else None,
    )
