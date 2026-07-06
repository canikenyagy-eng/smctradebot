from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smc.zones import PriceZone, assess_zone_lifecycle_as_of


@dataclass(frozen=True)
class OrderBlockContext:
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
        raise ValueError(f"Missing columns for order block detection: {', '.join(sorted(missing))}")


def _avg_body(frame: pd.DataFrame, end_index: int, lookback: int = 20) -> float:
    start = max(0, end_index - lookback)
    scoped = frame.iloc[start:end_index]
    if scoped.empty:
        return 0.0
    return float((scoped["close"].astype(float) - scoped["open"].astype(float)).abs().mean())


def _avg_range(frame: pd.DataFrame, end_index: int, lookback: int = 20) -> float:
    start = max(0, end_index - lookback)
    scoped = frame.iloc[start:end_index]
    if scoped.empty:
        return 0.0
    return float((scoped["high"].astype(float) - scoped["low"].astype(float)).mean())


def _find_order_block_candle(frame: pd.DataFrame, impulse_index: int, lookback: int = 6) -> int | None:
    start = max(0, impulse_index - lookback)
    impulse_direction = 1 if float(frame.iloc[impulse_index]["close"]) >= float(frame.iloc[impulse_index]["open"]) else -1

    for idx in range(impulse_index - 1, start - 1, -1):
        candle = frame.iloc[idx]
        bullish_candle = float(candle["close"]) > float(candle["open"])
        bearish_candle = float(candle["close"]) < float(candle["open"])

        if impulse_direction > 0 and bearish_candle:
            return idx
        if impulse_direction < 0 and bullish_candle:
            return idx

    return None


def _impulse_strength(body: float, avg_body: float, rng: float, avg_range: float) -> float:
    if avg_body <= 0 or avg_range <= 0:
        return 0.0
    body_component = body / avg_body
    range_component = rng / avg_range
    return min(1.0, 0.6 * (body_component / 3.0) + 0.4 * (range_component / 3.0))


def detect_order_blocks(
    frame: pd.DataFrame,
    *,
    lookback: int = 150,
    displacement_lookback: int = 20,
    block_lookback: int = 6,
    body_multiplier: float = 1.6,
    range_multiplier: float = 1.4,
    min_strength: float = 0.35,
) -> list[PriceZone]:
    if frame.empty or len(frame) < 3:
        return []

    _validate_frame(frame)
    start = max(1, len(frame) - lookback)
    zones: list[PriceZone] = []

    for impulse_index in range(start, len(frame)):
        impulse_row = frame.iloc[impulse_index]
        body = abs(float(impulse_row["close"]) - float(impulse_row["open"]))
        rng = float(impulse_row["high"]) - float(impulse_row["low"])
        avg_body = _avg_body(frame, impulse_index, lookback=displacement_lookback)
        avg_range = _avg_range(frame, impulse_index, lookback=displacement_lookback)

        if avg_body <= 0 or avg_range <= 0:
            continue
        if body < body_multiplier * avg_body or rng < range_multiplier * avg_range:
            continue

        block_index = _find_order_block_candle(frame, impulse_index, lookback=block_lookback)
        if block_index is None:
            continue

        block_row = frame.iloc[block_index]
        bullish_impulse = float(impulse_row["close"]) >= float(impulse_row["open"])

        if bullish_impulse:
            if float(block_row["close"]) >= float(block_row["open"]):
                continue
            lower = float(block_row["low"])
            upper = float(block_row["open"])
            zone_direction = "bullish"
        else:
            if float(block_row["close"]) <= float(block_row["open"]):
                continue
            lower = float(block_row["open"])
            upper = float(block_row["high"])
            zone_direction = "bearish"

        strength = _impulse_strength(body, avg_body, rng, avg_range)
        if strength < min_strength:
            continue

        zone = PriceZone(
            kind="order_block",
            direction=zone_direction,
            lower=lower,
            upper=upper,
            created_at=frame.index[impulse_index].to_pydatetime(),
            created_index=impulse_index,
            source_index=block_index,
            strength=strength,
        )
        zones.append(assess_zone_lifecycle_as_of(frame, zone, start_index=impulse_index, as_of_index=len(frame) - 1))

    return sorted(zones, key=lambda item: (item.created_index or -1, item.strength), reverse=True)


def latest_order_block(frame: pd.DataFrame, direction: str | None = None) -> PriceZone | None:
    zones = detect_order_blocks(frame)
    if direction is not None:
        direction_u = direction.strip().lower()
        zones = [zone for zone in zones if zone.direction == direction_u]
    return zones[0] if zones else None
