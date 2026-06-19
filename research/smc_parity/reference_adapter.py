from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from research.smc_parity.event_schema import AdapterStatus, SMCEvent, normalize_direction, timestamp_at
from smc.fvg import detect_fvg_zones
from smc.liquidity import analyze_liquidity
from smc.order_block import detect_order_blocks
from smc.structure import detect_bos_choch, identify_swings


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {', '.join(missing)}")
    out = frame.copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out[["open", "high", "low", "close", "volume"]].dropna()


def _value(row: pd.Series, key: str) -> Any:
    if key not in row:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    return value


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _pip_size(pair: str) -> float:
    clean = pair.upper().replace("/", "")
    return 0.01 if clean.endswith("JPY") else 0.0001


def _level_bucket(pair: str, level: float, tolerance_pips: float = 2.0) -> int:
    bucket_size = max(_pip_size(pair) * tolerance_pips, 1e-9)
    return round(float(level) / bucket_size)


def _reference_liquidity_direction(raw: Any) -> str:
    direction = normalize_direction(raw)
    if direction == "bullish":
        return "bearish"
    if direction == "bearish":
        return "bullish"
    return "neutral"


@dataclass(frozen=True)
class AdapterSettings:
    swing_window: int = 3
    max_structure_scan_bars: int = 900
    max_liquidity_scan_bars: int = 900
    fvg_lookback: int = 600
    ob_lookback: int = 600


class InternalSMCAdapter:
    name = "internal"

    def __init__(self, settings: AdapterSettings | None = None) -> None:
        self.settings = settings or AdapterSettings()

    def status(self) -> AdapterStatus:
        return AdapterStatus(name=self.name, available=True, version="local")

    def build_events(self, frame: pd.DataFrame, *, pair: str, timeframe: str) -> list[SMCEvent]:
        ohlcv = _clean_ohlcv(frame)
        events: list[SMCEvent] = []
        events.extend(self._fvg_events(ohlcv, pair=pair, timeframe=timeframe))
        events.extend(self._order_block_events(ohlcv, pair=pair, timeframe=timeframe))
        events.extend(self._swing_events(ohlcv, pair=pair, timeframe=timeframe))
        events.extend(self._structure_events(ohlcv, pair=pair, timeframe=timeframe))
        events.extend(self._liquidity_events(ohlcv, pair=pair, timeframe=timeframe))
        return sorted(events, key=lambda item: (item.known_at_index, item.event_type, item.direction))

    def _base_meta(self, pair: str, timeframe: str, **extra: Any) -> dict[str, Any]:
        return {"pair": pair.upper().replace("/", ""), "timeframe": timeframe.upper(), **extra}

    def _fvg_events(self, frame: pd.DataFrame, *, pair: str, timeframe: str) -> list[SMCEvent]:
        zones = detect_fvg_zones(frame, lookback=min(len(frame), self.settings.fvg_lookback))
        events: list[SMCEvent] = []
        for zone in zones:
            idx = int(zone.created_index or 0)
            events.append(
                SMCEvent(
                    source=self.name,
                    event_type="FVG",
                    direction=zone.direction,
                    timestamp=zone.created_at or timestamp_at(frame.index, idx),
                    confirmation_timestamp=zone.created_at or timestamp_at(frame.index, idx),
                    index=idx,
                    confirmation_index=idx,
                    top=zone.upper,
                    bottom=zone.lower,
                    strength=round(zone.strength * 100.0, 2),
                    mitigated_index=zone.last_touch_index,
                    live_safety="DELAYED_LIVE_SAFE",
                    metadata=self._base_meta(
                        pair,
                        timeframe,
                        source_index=zone.source_index,
                        touch_count=zone.touch_count,
                        fill_ratio=zone.fill_ratio,
                        is_fresh=zone.is_fresh,
                        invalidated=zone.invalidated,
                    ),
                )
            )
        return events

    def _order_block_events(self, frame: pd.DataFrame, *, pair: str, timeframe: str) -> list[SMCEvent]:
        zones = detect_order_blocks(frame, lookback=min(len(frame), self.settings.ob_lookback))
        events: list[SMCEvent] = []
        for zone in zones:
            idx = int(zone.created_index or 0)
            events.append(
                SMCEvent(
                    source=self.name,
                    event_type="ORDER_BLOCK",
                    direction=zone.direction,
                    timestamp=zone.created_at or timestamp_at(frame.index, idx),
                    confirmation_timestamp=zone.created_at or timestamp_at(frame.index, idx),
                    index=idx,
                    confirmation_index=idx,
                    top=zone.upper,
                    bottom=zone.lower,
                    strength=round(zone.strength * 100.0, 2),
                    mitigated_index=zone.last_touch_index,
                    live_safety="DELAYED_LIVE_SAFE",
                    metadata=self._base_meta(
                        pair,
                        timeframe,
                        source_index=zone.source_index,
                        touch_count=zone.touch_count,
                        fill_ratio=zone.fill_ratio,
                        is_fresh=zone.is_fresh,
                        invalidated=zone.invalidated,
                    ),
                )
            )
        return events

    def _swing_events(self, frame: pd.DataFrame, *, pair: str, timeframe: str) -> list[SMCEvent]:
        window = max(1, int(self.settings.swing_window))
        swings = identify_swings(frame, window=window)
        events: list[SMCEvent] = []
        for idx, row in enumerate(swings.itertuples(index=False)):
            is_high = bool(getattr(row, "swing_high", False))
            is_low = bool(getattr(row, "swing_low", False))
            if not is_high and not is_low:
                continue
            event_type = "SWING_HIGH" if is_high else "SWING_LOW"
            level = float(getattr(row, "high" if is_high else "low"))
            confirmation_index = min(len(frame) - 1, idx + window)
            events.append(
                SMCEvent(
                    source=self.name,
                    event_type=event_type,
                    direction="bearish" if is_high else "bullish",
                    timestamp=timestamp_at(frame.index, idx),
                    confirmation_timestamp=timestamp_at(frame.index, confirmation_index),
                    index=idx,
                    confirmation_index=confirmation_index,
                    level=level,
                    live_safety="DELAYED_LIVE_SAFE",
                    metadata=self._base_meta(pair, timeframe, swing_window=window),
                )
            )
        return events

    def _structure_events(self, frame: pd.DataFrame, *, pair: str, timeframe: str) -> list[SMCEvent]:
        max_scan = min(len(frame), max(120, self.settings.max_structure_scan_bars))
        start = max(20, len(frame) - max_scan)
        seen: set[tuple[str, str, int, float | None]] = set()
        events: list[SMCEvent] = []

        for idx in range(start, len(frame)):
            state = detect_bos_choch(frame.iloc[: idx + 1], window=self.settings.swing_window)
            if state.event is None or state.direction is None:
                continue
            level = state.last_swing_high if state.direction == "bullish" else state.last_swing_low
            signature = (state.event, state.direction, idx, round(float(level or 0.0), 8) if level is not None else None)
            if signature in seen:
                continue
            seen.add(signature)
            close_price = float(frame["close"].iloc[idx])
            break_distance_pips = (
                abs(close_price - float(level)) / _pip_size(pair)
                if level is not None
                else None
            )
            events.append(
                SMCEvent(
                    source=self.name,
                    event_type=state.event,
                    direction=state.direction,
                    timestamp=timestamp_at(frame.index, idx),
                    confirmation_timestamp=timestamp_at(frame.index, idx),
                    index=idx,
                    confirmation_index=idx,
                    level=level,
                    live_safety="DELAYED_LIVE_SAFE",
                    metadata=self._base_meta(
                        pair,
                        timeframe,
                        trend=state.trend,
                        break_price=close_price,
                        break_distance_pips=round(break_distance_pips, 4) if break_distance_pips is not None else None,
                    ),
                )
            )
        return events

    def _liquidity_events(self, frame: pd.DataFrame, *, pair: str, timeframe: str) -> list[SMCEvent]:
        max_scan = min(len(frame), max(120, self.settings.max_liquidity_scan_bars))
        start = max(30, len(frame) - max_scan)
        events: list[SMCEvent] = []
        pool_seen: set[tuple[str, int]] = set()

        for idx in range(start, len(frame)):
            context = analyze_liquidity(frame.iloc[: idx + 1], swing_window=self.settings.swing_window)
            if context.equal_highs and context.equal_high_level is not None:
                key = ("EQUAL_HIGHS", _level_bucket(pair, context.equal_high_level))
                if key not in pool_seen:
                    pool_seen.add(key)
                    events.append(
                        SMCEvent(
                            source=self.name,
                            event_type="LIQUIDITY_POOL",
                            direction="bearish",
                            timestamp=timestamp_at(frame.index, idx),
                            confirmation_timestamp=timestamp_at(frame.index, idx),
                            index=idx,
                            confirmation_index=idx,
                            level=context.equal_high_level,
                            live_safety="DELAYED_LIVE_SAFE",
                            metadata=self._base_meta(pair, timeframe, pool_type="equal_highs"),
                        )
                    )
            if context.equal_lows and context.equal_low_level is not None:
                key = ("EQUAL_LOWS", _level_bucket(pair, context.equal_low_level))
                if key not in pool_seen:
                    pool_seen.add(key)
                    events.append(
                        SMCEvent(
                            source=self.name,
                            event_type="LIQUIDITY_POOL",
                            direction="bullish",
                            timestamp=timestamp_at(frame.index, idx),
                            confirmation_timestamp=timestamp_at(frame.index, idx),
                            index=idx,
                            confirmation_index=idx,
                            level=context.equal_low_level,
                            live_safety="DELAYED_LIVE_SAFE",
                            metadata=self._base_meta(pair, timeframe, pool_type="equal_lows"),
                        )
                    )
            if context.sweep and context.sweep_direction is not None:
                level = context.equal_low_level if context.sweep_direction == "bullish" else context.equal_high_level
                events.append(
                    SMCEvent(
                        source=self.name,
                        event_type="LIQUIDITY_SWEEP",
                        direction=context.sweep_direction,
                        timestamp=timestamp_at(frame.index, idx),
                        confirmation_timestamp=timestamp_at(frame.index, idx),
                        index=idx,
                        confirmation_index=idx,
                        level=level,
                        live_safety="LIVE_SAFE",
                        metadata=self._base_meta(
                            pair,
                            timeframe,
                            displacement=context.displacement,
                            displacement_direction=context.displacement_direction,
                        ),
                    )
                )
        return events


class SmartMoneyConceptsReferenceAdapter:
    name = "smartmoneyconcepts"

    def __init__(self, settings: AdapterSettings | None = None) -> None:
        self.settings = settings or AdapterSettings()
        self._smc_module: Any | None = None
        self._status = self._load_status()

    def _load_status(self) -> AdapterStatus:
        try:
            os.environ.setdefault("SMC_CREDIT", "0")
            from smartmoneyconcepts import smc as smc_module  # type: ignore
        except Exception as exc:
            return AdapterStatus(name=self.name, available=False, reason=str(exc))
        self._smc_module = smc_module
        version = getattr(smc_module, "__version__", None)
        return AdapterStatus(name=self.name, available=True, version=str(version) if version else None)

    def status(self) -> AdapterStatus:
        return self._status

    def build_events(self, frame: pd.DataFrame, *, pair: str, timeframe: str) -> list[SMCEvent]:
        if self._smc_module is None:
            raise RuntimeError(self._status.reason or "smartmoneyconcepts is not available")

        ohlcv = _clean_ohlcv(frame).reset_index(drop=True)
        source_index = _clean_ohlcv(frame).index
        events: list[SMCEvent] = []

        swings = self._safe_call("swing_highs_lows", ohlcv, swing_length=self.settings.swing_window)
        events.extend(self._external_fvg_events(ohlcv, source_index, pair=pair, timeframe=timeframe))
        if isinstance(swings, pd.DataFrame):
            events.extend(self._external_swing_events(swings, source_index, pair=pair, timeframe=timeframe))
            events.extend(self._external_structure_events(ohlcv, swings, source_index, pair=pair, timeframe=timeframe))
            events.extend(self._external_ob_events(ohlcv, swings, source_index, pair=pair, timeframe=timeframe))
            events.extend(self._external_liquidity_events(ohlcv, swings, source_index, pair=pair, timeframe=timeframe))
        return sorted(events, key=lambda item: (item.known_at_index, item.event_type, item.direction))

    def _safe_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        func = getattr(self._smc_module, method)
        return func(*args, **kwargs)

    def _base_meta(self, pair: str, timeframe: str, **extra: Any) -> dict[str, Any]:
        return {"pair": pair.upper().replace("/", ""), "timeframe": timeframe.upper(), **extra}

    def _external_fvg_events(
        self,
        ohlcv: pd.DataFrame,
        frame_index: pd.Index,
        *,
        pair: str,
        timeframe: str,
    ) -> list[SMCEvent]:
        result = self._safe_call("fvg", ohlcv, join_consecutive=False)
        events: list[SMCEvent] = []
        if not isinstance(result, pd.DataFrame):
            return events
        for idx, row in result.iterrows():
            raw = _value(row, "FVG")
            if raw is None:
                continue
            confirmation_index = min(len(frame_index) - 1, int(idx) + 1)
            events.append(
                SMCEvent(
                    source=self.name,
                    event_type="FVG",
                    direction=normalize_direction(raw),
                    timestamp=timestamp_at(frame_index, int(idx)),
                    confirmation_timestamp=timestamp_at(frame_index, confirmation_index),
                    index=int(idx),
                    confirmation_index=confirmation_index,
                    top=_float_or_none(_value(row, "Top")),
                    bottom=_float_or_none(_value(row, "Bottom")),
                    mitigated_index=_int_or_none(_value(row, "MitigatedIndex")),
                    live_safety="DELAYED_LIVE_SAFE",
                    metadata=self._base_meta(pair, timeframe, source_marks_middle_candle=True),
                )
            )
        return events

    def _external_swing_events(
        self,
        swings: pd.DataFrame,
        frame_index: pd.Index,
        *,
        pair: str,
        timeframe: str,
    ) -> list[SMCEvent]:
        events: list[SMCEvent] = []
        delay = max(1, self.settings.swing_window)
        for idx, row in swings.iterrows():
            raw = _value(row, "HighLow")
            if raw is None:
                continue
            is_high = float(raw) > 0
            confirmation_index = min(len(frame_index) - 1, int(idx) + delay)
            events.append(
                SMCEvent(
                    source=self.name,
                    event_type="SWING_HIGH" if is_high else "SWING_LOW",
                    direction="bearish" if is_high else "bullish",
                    timestamp=timestamp_at(frame_index, int(idx)),
                    confirmation_timestamp=timestamp_at(frame_index, confirmation_index),
                    index=int(idx),
                    confirmation_index=confirmation_index,
                    level=_float_or_none(_value(row, "Level")),
                    live_safety="DELAYED_LIVE_SAFE",
                    metadata=self._base_meta(pair, timeframe, swing_window=self.settings.swing_window),
                )
            )
        return events

    def _external_structure_events(
        self,
        ohlcv: pd.DataFrame,
        swings: pd.DataFrame,
        frame_index: pd.Index,
        *,
        pair: str,
        timeframe: str,
    ) -> list[SMCEvent]:
        result = self._safe_call("bos_choch", ohlcv, swings, close_break=True)
        events: list[SMCEvent] = []
        if not isinstance(result, pd.DataFrame):
            return events
        for idx, row in result.iterrows():
            for event_type in ("BOS", "CHOCH"):
                raw = _value(row, event_type)
                if raw is None:
                    continue
                broken_index = _int_or_none(_value(row, "BrokenIndex"))
                confirmation_index = min(len(frame_index) - 1, broken_index or int(idx))
                level = _float_or_none(_value(row, "Level"))
                break_price = float(ohlcv["close"].iloc[confirmation_index]) if confirmation_index < len(ohlcv) else None
                break_distance_pips = (
                    abs(break_price - level) / _pip_size(pair)
                    if break_price is not None and level is not None
                    else None
                )
                events.append(
                    SMCEvent(
                        source=self.name,
                        event_type=event_type,
                        direction=normalize_direction(raw),
                        timestamp=timestamp_at(frame_index, int(idx)),
                        confirmation_timestamp=timestamp_at(frame_index, confirmation_index),
                        index=int(idx),
                        confirmation_index=confirmation_index,
                        level=level,
                        live_safety="DELAYED_LIVE_SAFE",
                        metadata=self._base_meta(
                            pair,
                            timeframe,
                            broken_index=broken_index,
                            break_price=break_price,
                            break_distance_pips=round(break_distance_pips, 4) if break_distance_pips is not None else None,
                        ),
                    )
                )
        return events

    def _external_ob_events(
        self,
        ohlcv: pd.DataFrame,
        swings: pd.DataFrame,
        frame_index: pd.Index,
        *,
        pair: str,
        timeframe: str,
    ) -> list[SMCEvent]:
        result = self._safe_call("ob", ohlcv, swings, close_mitigation=False)
        events: list[SMCEvent] = []
        if not isinstance(result, pd.DataFrame):
            return events
        for idx, row in result.iterrows():
            raw = _value(row, "OB")
            if raw is None:
                continue
            mitigated_index = _int_or_none(_value(row, "MitigatedIndex"))
            events.append(
                SMCEvent(
                    source=self.name,
                    event_type="ORDER_BLOCK",
                    direction=normalize_direction(raw),
                    timestamp=timestamp_at(frame_index, int(idx)),
                    confirmation_timestamp=timestamp_at(frame_index, int(idx)),
                    index=int(idx),
                    confirmation_index=int(idx),
                    top=_float_or_none(_value(row, "Top")),
                    bottom=_float_or_none(_value(row, "Bottom")),
                    strength=_float_or_none(_value(row, "Percentage")),
                    mitigated_index=mitigated_index,
                    live_safety="RESEARCH_ONLY" if mitigated_index is not None else "DELAYED_LIVE_SAFE",
                    metadata=self._base_meta(pair, timeframe, ob_volume=_float_or_none(_value(row, "OBVolume"))),
                )
            )
        return events

    def _external_liquidity_events(
        self,
        ohlcv: pd.DataFrame,
        swings: pd.DataFrame,
        frame_index: pd.Index,
        *,
        pair: str,
        timeframe: str,
    ) -> list[SMCEvent]:
        result = self._safe_call("liquidity", ohlcv, swings, range_percent=0.01)
        events: list[SMCEvent] = []
        if not isinstance(result, pd.DataFrame):
            return events
        for idx, row in result.iterrows():
            raw = _value(row, "Liquidity")
            if raw is None:
                continue
            swept_index = _int_or_none(_value(row, "Swept"))
            direction = _reference_liquidity_direction(raw)
            level = _float_or_none(_value(row, "Level"))
            events.append(
                SMCEvent(
                    source=self.name,
                    event_type="LIQUIDITY_POOL",
                    direction=direction,
                    timestamp=timestamp_at(frame_index, int(idx)),
                    confirmation_timestamp=timestamp_at(frame_index, int(idx)),
                    index=int(idx),
                    confirmation_index=int(idx),
                    level=level,
                    swept_index=swept_index,
                    live_safety="RESEARCH_ONLY" if swept_index is not None else "DELAYED_LIVE_SAFE",
                    metadata=self._base_meta(
                        pair,
                        timeframe,
                        end_index=_int_or_none(_value(row, "End")),
                        reference_liquidity_raw=float(raw),
                        direction_semantics="expected_post_sweep_reversal",
                    ),
                )
            )
            if swept_index is not None:
                events.append(
                    SMCEvent(
                        source=self.name,
                        event_type="LIQUIDITY_SWEEP",
                        direction=direction,
                        timestamp=timestamp_at(frame_index, swept_index),
                        confirmation_timestamp=timestamp_at(frame_index, swept_index),
                        index=swept_index,
                        confirmation_index=swept_index,
                        level=level,
                        live_safety="LIVE_SAFE",
                        metadata=self._base_meta(
                            pair,
                            timeframe,
                            pool_index=int(idx),
                            end_index=_int_or_none(_value(row, "End")),
                            derived_from_reference_swept_index=True,
                        ),
                    )
                )
        return events


def build_reference_adapter(provider: str, settings: AdapterSettings | None = None) -> SmartMoneyConceptsReferenceAdapter:
    normalized = provider.strip().lower()
    if normalized not in {"smartmoneyconcepts", "smart-money-concepts", "joshyattridge"}:
        raise ValueError(f"Unsupported SMC reference provider: {provider}")
    return SmartMoneyConceptsReferenceAdapter(settings=settings)
