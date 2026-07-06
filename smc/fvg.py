from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smc.zones import PriceZone, assess_zone_lifecycle_as_of


@dataclass(frozen=True)
class FVGContext:
    zones: list[PriceZone]

    @property
    def bullish(self) -> list[PriceZone]:
        return [zone for zone in self.zones if zone.direction == "bullish"]

    @property
    def bearish(self) -> list[PriceZone]:
        return [zone for zone in self.zones if zone.direction == "bearish"]

    @property
    def active(self) -> list[PriceZone]:
        return [zone for zone in self.zones if zone.is_fresh and not zone.invalidated]


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for FVG detection: {', '.join(sorted(missing))}")


def _avg_range(frame: pd.DataFrame, end_index: int, lookback: int = 20) -> float:
    start = max(0, end_index - lookback)
    scoped = frame.iloc[start:end_index]
    if scoped.empty:
        return 0.0
    return float((scoped["high"].astype(float) - scoped["low"].astype(float)).mean())


def _gap_strength(gap_size: float, avg_range: float, middle_body: float) -> float:
    if avg_range <= 0:
        return 0.0
    return min(1.0, (gap_size / avg_range) * 0.7 + (middle_body / avg_range) * 0.3)


def detect_fvg_zones(
    frame: pd.DataFrame,
    *,
    lookback: int = 150,
    min_gap_ratio: float = 0.15,
    min_body_ratio: float = 0.35,
) -> list[PriceZone]:
    if frame.empty or len(frame) < 3:
        return []

    _validate_frame(frame)
    start = max(1, len(frame) - lookback - 1)
    zones: list[PriceZone] = []

    for mid_index in range(start, len(frame) - 1):
        prev_index = mid_index - 1
        next_index = mid_index + 1

        prev_row = frame.iloc[prev_index]
        mid_row = frame.iloc[mid_index]
        next_row = frame.iloc[next_index]

        prev_high = float(prev_row["high"])
        prev_low = float(prev_row["low"])
        mid_open = float(mid_row["open"])
        mid_close = float(mid_row["close"])
        next_high = float(next_row["high"])
        next_low = float(next_row["low"])

        avg_range = _avg_range(frame, mid_index, lookback=20)
        gap_size = 0.0

        bullish_gap = next_low > prev_high and mid_close >= mid_open
        if bullish_gap:
            gap_size = next_low - prev_high
            if avg_range <= 0 or gap_size / avg_range < min_gap_ratio:
                continue
            if abs(mid_close - mid_open) / max(avg_range, 1e-9) < min_body_ratio:
                continue

            zone = PriceZone(
                kind="fvg",
                direction="bullish",
                lower=prev_high,
                upper=next_low,
                created_at=frame.index[next_index].to_pydatetime(),
                created_index=next_index,
                source_index=mid_index,
                strength=_gap_strength(gap_size, avg_range, abs(mid_close - mid_open)),
            )
            zones.append(assess_zone_lifecycle_as_of(frame, zone, start_index=next_index, as_of_index=len(frame) - 1))
            continue

        bearish_gap = next_high < prev_low and mid_close <= mid_open
        if bearish_gap:
            gap_size = prev_low - next_high
            if avg_range <= 0 or gap_size / avg_range < min_gap_ratio:
                continue
            if abs(mid_close - mid_open) / max(avg_range, 1e-9) < min_body_ratio:
                continue

            zone = PriceZone(
                kind="fvg",
                direction="bearish",
                lower=next_high,
                upper=prev_low,
                created_at=frame.index[next_index].to_pydatetime(),
                created_index=next_index,
                source_index=mid_index,
                strength=_gap_strength(gap_size, avg_range, abs(mid_close - mid_open)),
            )
            zones.append(assess_zone_lifecycle_as_of(frame, zone, start_index=next_index, as_of_index=len(frame) - 1))

    return sorted(zones, key=lambda item: (item.created_index or -1, item.strength), reverse=True)


def latest_fvg(frame: pd.DataFrame, direction: str | None = None) -> PriceZone | None:
    zones = detect_fvg_zones(frame)
    if direction is not None:
        direction_u = direction.strip().lower()
        zones = [zone for zone in zones if zone.direction == direction_u]
    return zones[0] if zones else None
