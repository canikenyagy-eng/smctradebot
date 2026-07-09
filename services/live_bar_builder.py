"""Build local OHLCV bars from live quote events."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping

import pandas as pd

from services.itick_websocket_shadow import parse_provider_time

logger = logging.getLogger(__name__)


TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
}


@dataclass(frozen=True)
class LiveBarBuilderSettings:
    enabled: bool = False
    source: str = "itick_websocket"
    timeframes: tuple[str, ...] = ("M5", "M15", "H1")
    bars_dir: Path | str = Path("data/live_bars/itick")
    log_path: Path | str = Path("logs/live_bars_itick.jsonl")
    max_bars_per_timeframe: int = 1000
    flush_interval_seconds: float = 2.0
    max_quote_age_seconds: float = 5.0

    def normalized(self) -> "LiveBarBuilderSettings":
        timeframes = tuple(
            dict.fromkeys(
                timeframe.strip().upper()
                for timeframe in self.timeframes
                if timeframe.strip().upper() in TIMEFRAME_SECONDS
            )
        )
        return LiveBarBuilderSettings(
            enabled=bool(self.enabled),
            source=self.source.strip() or "itick_websocket",
            timeframes=timeframes or ("M5", "M15", "H1"),
            bars_dir=Path(self.bars_dir),
            log_path=Path(self.log_path),
            max_bars_per_timeframe=max(100, int(self.max_bars_per_timeframe)),
            flush_interval_seconds=max(0.25, float(self.flush_interval_seconds)),
            max_quote_age_seconds=max(1.0, float(self.max_quote_age_seconds)),
        )


@dataclass
class MutableBar:
    pair: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_count: int
    first_provider_time: datetime
    last_provider_time: datetime
    last_observed_at: datetime
    complete: bool = False

    def update(self, *, price: float, provider_time: datetime, observed_at: datetime) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += 1.0
        self.quote_count += 1
        self.last_provider_time = provider_time
        self.last_observed_at = observed_at

    def to_row(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "timeframe": self.timeframe,
            "timestamp": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_count": self.quote_count,
            "first_provider_time": self.first_provider_time.isoformat(),
            "last_provider_time": self.last_provider_time.isoformat(),
            "last_observed_at": self.last_observed_at.isoformat(),
            "complete": self.complete,
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clean_pair(value: object) -> str:
    return str(value or "").upper().replace("/", "").strip()


def parse_time(value: object | None) -> datetime | None:
    return parse_provider_time(value)


def as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def floor_time(timestamp: datetime, timeframe: str) -> datetime:
    seconds = TIMEFRAME_SECONDS[timeframe]
    epoch = int(timestamp.timestamp())
    floored = epoch - (epoch % seconds)
    return datetime.fromtimestamp(floored, timezone.utc)


def end_time(start_time: datetime, timeframe: str) -> datetime:
    return datetime.fromtimestamp(start_time.timestamp() + TIMEFRAME_SECONDS[timeframe], timezone.utc)


def read_jsonl(path: Path | str) -> list[dict[str, object]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, object]] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


class LiveBarBuilder:
    """Aggregates live quote messages into local OHLCV bars."""

    def __init__(self, settings: LiveBarBuilderSettings) -> None:
        self.settings = settings.normalized()
        self._current: dict[tuple[str, str], MutableBar] = {}
        self._closed: defaultdict[tuple[str, str], deque[MutableBar]] = defaultdict(
            lambda: deque(maxlen=self.settings.max_bars_per_timeframe)
        )
        self._last_flush = 0.0
        self._ignored_quotes = 0

    def on_quote(self, quote: Mapping[str, object]) -> None:
        if not self.settings.enabled:
            return
        pair = clean_pair(quote.get("pair"))
        price = as_float(quote.get("last_price"))
        provider_time = parse_time(quote.get("provider_time"))
        observed_at = parse_time(quote.get("observed_at")) or utc_now()
        if len(pair) != 6 or price is None or provider_time is None:
            self._ignored_quotes += 1
            return

        quote_age = max(0.0, (observed_at - provider_time).total_seconds())
        if quote_age > self.settings.max_quote_age_seconds:
            self._ignored_quotes += 1
            self._write_event(
                {
                    "event": "ignored_stale_quote",
                    "pair": pair,
                    "provider_time": provider_time.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    "quote_age_seconds": round(quote_age, 6),
                    "max_quote_age_seconds": self.settings.max_quote_age_seconds,
                }
            )
            return

        for timeframe in self.settings.timeframes:
            self._apply_quote(pair=pair, timeframe=timeframe, price=price, provider_time=provider_time, observed_at=observed_at)

        if time.monotonic() - self._last_flush >= self.settings.flush_interval_seconds:
            self.flush()

    def flush(self) -> None:
        if not self.settings.enabled:
            return
        self._last_flush = time.monotonic()
        for key in sorted(set(self._closed) | set(self._current)):
            pair, timeframe = key
            rows = [bar.to_row() for bar in self._closed.get(key, [])]
            current = self._current.get(key)
            if current is not None:
                rows.append(current.to_row())
                self._write_event({"event": "bar_update", **current.to_row()})
            self._write_state(pair=pair, timeframe=timeframe, rows=rows)

    def _apply_quote(
        self,
        *,
        pair: str,
        timeframe: str,
        price: float,
        provider_time: datetime,
        observed_at: datetime,
    ) -> None:
        key = (pair, timeframe)
        start_time = floor_time(provider_time, timeframe)
        current = self._current.get(key)

        if current is not None and start_time < current.start_time:
            self._ignored_quotes += 1
            return

        if current is None:
            self._current[key] = self._new_bar(pair, timeframe, start_time, price, provider_time, observed_at)
            return

        if start_time > current.start_time:
            current.complete = True
            self._closed[key].append(current)
            self._write_event({"event": "bar_closed", **current.to_row()})
            self._current[key] = self._new_bar(pair, timeframe, start_time, price, provider_time, observed_at)
            return

        current.update(price=price, provider_time=provider_time, observed_at=observed_at)

    def _new_bar(
        self,
        pair: str,
        timeframe: str,
        start_time: datetime,
        price: float,
        provider_time: datetime,
        observed_at: datetime,
    ) -> MutableBar:
        return MutableBar(
            pair=pair,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time(start_time, timeframe),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1.0,
            quote_count=1,
            first_provider_time=provider_time,
            last_provider_time=provider_time,
            last_observed_at=observed_at,
            complete=False,
        )

    def _write_state(self, *, pair: str, timeframe: str, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        path = Path(self.settings.bars_dir) / f"{pair}_{timeframe}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(tmp_path, index=False)
        tmp_path.replace(path)

    def _write_event(self, payload: dict[str, object]) -> None:
        path = Path(self.settings.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "type": "live_bar_builder",
            "version": 1,
            "source": self.settings.source,
            "observed_at": utc_now().isoformat(),
            **payload,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


class LiveBarBuilderReporter:
    def __init__(
        self,
        *,
        log_path: Path | str,
        recent_minutes: int = 1440,
        max_bar_age_seconds: float = 30.0,
    ) -> None:
        self.log_path = Path(log_path)
        self.recent_minutes = max(1, int(recent_minutes))
        self.max_bar_age_seconds = max(1.0, float(max_bar_age_seconds))

    def build_report(self) -> dict[str, object]:
        now = utc_now()
        cutoff = now.timestamp() - self.recent_minutes * 60
        rows = []
        for row in read_jsonl(self.log_path):
            if str(row.get("type")) != "live_bar_builder":
                continue
            observed_at = parse_time(row.get("observed_at"))
            if observed_at is None or observed_at.timestamp() < cutoff:
                continue
            rows.append(row)

        updates = [row for row in rows if row.get("event") in {"bar_update", "bar_closed"}]
        return {
            "type": "live_bar_builder_summary",
            "version": 1,
            "generated_at": now.isoformat(),
            "settings": {
                "log_path": str(self.log_path),
                "recent_minutes": self.recent_minutes,
                "max_bar_age_seconds": self.max_bar_age_seconds,
            },
            "overall": self._stats(updates),
            "by_pair_timeframe": self._group(updates),
            "latest": self._latest(updates),
        }

    def _group(self, rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
        grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            key = f"{row.get('pair', 'unknown')}_{row.get('timeframe', 'unknown')}"
            grouped[key].append(row)
        return {key: self._stats(group_rows) for key, group_rows in sorted(grouped.items())}

    def _stats(self, rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
        row_list = list(rows)
        ages = []
        quote_counts = []
        for row in row_list:
            last_provider_time = parse_time(row.get("last_provider_time"))
            observed_at = parse_time(row.get("observed_at"))
            if last_provider_time is not None and observed_at is not None:
                ages.append(max(0.0, (observed_at - last_provider_time).total_seconds()))
            quote_count = as_float(row.get("quote_count"))
            if quote_count is not None:
                quote_counts.append(quote_count)
        stale = [age for age in ages if age > self.max_bar_age_seconds]
        return {
            "updates": len(row_list),
            "closed": sum(1 for row in row_list if row.get("event") == "bar_closed"),
            "stale_updates": len(stale),
            "avg_bar_age_seconds": round(mean(ages), 6) if ages else 0.0,
            "max_bar_age_seconds": round(max(ages), 6) if ages else 0.0,
            "avg_quote_count": round(mean(quote_counts), 3) if quote_counts else 0.0,
            "alert": bool(not row_list or stale),
        }

    def _latest(self, rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
        latest: dict[str, Mapping[str, object]] = {}
        for row in rows:
            key = f"{row.get('pair', 'unknown')}_{row.get('timeframe', 'unknown')}"
            observed_at = parse_time(row.get("observed_at"))
            previous = latest.get(key)
            previous_at = parse_time(previous.get("observed_at")) if previous else None
            if observed_at is not None and (previous_at is None or observed_at > previous_at):
                latest[key] = row
        return {
            key: {
                "event": row.get("event"),
                "timestamp": row.get("timestamp"),
                "end_time": row.get("end_time"),
                "close": row.get("close"),
                "quote_count": row.get("quote_count"),
                "last_provider_time": row.get("last_provider_time"),
                "last_observed_at": row.get("last_observed_at"),
                "complete": row.get("complete"),
            }
            for key, row in sorted(latest.items())
        }
