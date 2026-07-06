from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class StructureState:
    trend: str
    event: str | None
    direction: str | None
    last_swing_high: float | None
    last_swing_low: float | None
    last_swing_high_index: int | None = None
    last_swing_low_index: int | None = None
    last_swing_high_confirmed_at_index: int | None = None
    last_swing_low_confirmed_at_index: int | None = None
    event_confirmed_at_index: int | None = None


def identify_swings(frame: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame

    if len(frame) < window * 2 + 3:
        out = frame.copy()
        out["swing_high"] = False
        out["swing_low"] = False
        out["swing_high_confirmed_at_index"] = pd.NA
        out["swing_low_confirmed_at_index"] = pd.NA
        return out

    out = frame.copy()
    out["swing_high"] = False
    out["swing_low"] = False
    out["swing_high_confirmed_at_index"] = pd.NA
    out["swing_low_confirmed_at_index"] = pd.NA

    for idx in range(window, len(out) - window):
        high_slice = out["high"].iloc[idx - window : idx + window + 1]
        low_slice = out["low"].iloc[idx - window : idx + window + 1]

        if out["high"].iloc[idx] == high_slice.max():
            out.iloc[idx, out.columns.get_loc("swing_high")] = True
            out.iloc[idx, out.columns.get_loc("swing_high_confirmed_at_index")] = idx + window

        if out["low"].iloc[idx] == low_slice.min():
            out.iloc[idx, out.columns.get_loc("swing_low")] = True
            out.iloc[idx, out.columns.get_loc("swing_low_confirmed_at_index")] = idx + window

    return out


def _last_two(values: pd.Series) -> Tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    return float(values.iloc[-2]), float(values.iloc[-1])


def infer_trend(swings_frame: pd.DataFrame) -> str:
    highs = swings_frame.loc[swings_frame["swing_high"], "high"]
    lows = swings_frame.loc[swings_frame["swing_low"], "low"]

    prev_high, last_high = _last_two(highs)
    prev_low, last_low = _last_two(lows)

    if None in {prev_high, last_high, prev_low, last_low}:
        return "neutral"

    if last_high > prev_high and last_low > prev_low:
        return "bullish"

    if last_high < prev_high and last_low < prev_low:
        return "bearish"

    return "neutral"


def detect_bos_choch(frame: pd.DataFrame, window: int = 3) -> StructureState:
    swings_frame = identify_swings(frame, window=window)
    trend = infer_trend(swings_frame)

    high_mask = swings_frame["swing_high"].astype(bool)
    low_mask = swings_frame["swing_low"].astype(bool)
    recent_highs = swings_frame.loc[high_mask, "high"]
    recent_lows = swings_frame.loc[low_mask, "low"]

    if recent_highs.empty or recent_lows.empty:
        return StructureState(
            trend=trend,
            event=None,
            direction=None,
            last_swing_high=None,
            last_swing_low=None,
        )

    high_positions = [int(pos) for pos, is_swing in enumerate(high_mask) if bool(is_swing)]
    low_positions = [int(pos) for pos, is_swing in enumerate(low_mask) if bool(is_swing)]
    last_high_index = high_positions[-1] if high_positions else None
    last_low_index = low_positions[-1] if low_positions else None
    last_swing_high = float(recent_highs.iloc[-1])
    last_swing_low = float(recent_lows.iloc[-1])
    last_close = float(frame["close"].iloc[-1])
    last_high_confirmed = (
        int(swings_frame["swing_high_confirmed_at_index"].iloc[last_high_index])
        if last_high_index is not None and pd.notna(swings_frame["swing_high_confirmed_at_index"].iloc[last_high_index])
        else None
    )
    last_low_confirmed = (
        int(swings_frame["swing_low_confirmed_at_index"].iloc[last_low_index])
        if last_low_index is not None and pd.notna(swings_frame["swing_low_confirmed_at_index"].iloc[last_low_index])
        else None
    )
    event_confirmed = len(frame) - 1

    if last_close > last_swing_high:
        event = "BOS" if trend in {"bullish", "neutral"} else "CHoCH"
        return StructureState(
            trend=trend,
            event=event,
            direction="bullish",
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            last_swing_high_index=last_high_index,
            last_swing_low_index=last_low_index,
            last_swing_high_confirmed_at_index=last_high_confirmed,
            last_swing_low_confirmed_at_index=last_low_confirmed,
            event_confirmed_at_index=event_confirmed,
        )

    if last_close < last_swing_low:
        event = "BOS" if trend in {"bearish", "neutral"} else "CHoCH"
        return StructureState(
            trend=trend,
            event=event,
            direction="bearish",
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            last_swing_high_index=last_high_index,
            last_swing_low_index=last_low_index,
            last_swing_high_confirmed_at_index=last_high_confirmed,
            last_swing_low_confirmed_at_index=last_low_confirmed,
            event_confirmed_at_index=event_confirmed,
        )

    return StructureState(
        trend=trend,
        event=None,
        direction=None,
        last_swing_high=last_swing_high,
        last_swing_low=last_swing_low,
        last_swing_high_index=last_high_index,
        last_swing_low_index=last_low_index,
        last_swing_high_confirmed_at_index=last_high_confirmed,
        last_swing_low_confirmed_at_index=last_low_confirmed,
        event_confirmed_at_index=None,
    )
