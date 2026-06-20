"""Shadow diagnostics for comparing primary and candidate market data feeds."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from core.signal_engine import TradeSignal
from data.market_data import MarketDataClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketDataShadowSettings:
    enabled: bool = False
    primary_source: str = "yahoo"
    candidate_source: str = "itick"
    timeframes: tuple[str, ...] = ("M5", "M15", "H1")
    log_path: Path | str = Path("logs/market_data_shadow.jsonl")
    max_close_diff_pips: float = 2.0
    max_staleness_seconds: int = 120
    compare_signals: bool = True


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: object | None) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _pip_size(pair: str) -> float:
    return 0.01 if pair.upper().replace("/", "").endswith("JPY") else 0.0001


def _signal_key(signal: TradeSignal) -> str:
    return "|".join(
        [
            signal.symbol,
            signal.side,
            str(round(float(signal.entry), 5)),
            str(round(float(signal.stop_loss), 5)),
            str(round(float(signal.take_profit), 5)),
        ]
    )


class MarketDataShadowLogger:
    """Writes JSONL diagnostics for feed and signal parity checks."""

    def __init__(
        self,
        settings: MarketDataShadowSettings,
        primary_client: MarketDataClient,
        candidate_client: MarketDataClient,
    ) -> None:
        self.settings = settings
        self.primary_client = primary_client
        self.candidate_client = candidate_client
        self.log_path = Path(settings.log_path)

    def _write(self, payload: dict[str, object]) -> None:
        if not self.settings.enabled:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def compare_market_data(self, pairs: Sequence[str]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if not self.settings.enabled:
            return rows

        observed_at = _utc_now()
        for pair in pairs:
            pip = _pip_size(pair)
            for timeframe in self.settings.timeframes:
                row: dict[str, object] = {
                    "type": "market_data_shadow",
                    "observed_at": observed_at.isoformat(),
                    "pair": pair,
                    "timeframe": timeframe,
                    "primary_source": self.settings.primary_source,
                    "candidate_source": self.settings.candidate_source,
                }
                try:
                    primary_started = time.monotonic()
                    primary = self.primary_client.fetch_ohlcv(pair, timeframe)
                    primary_latency = time.monotonic() - primary_started

                    candidate_started = time.monotonic()
                    candidate = self.candidate_client.fetch_ohlcv(pair, timeframe)
                    candidate_latency = time.monotonic() - candidate_started

                    primary_last_time = primary.index[-1]
                    candidate_last_time = candidate.index[-1]
                    primary_close = float(primary["close"].iloc[-1])
                    candidate_close = float(candidate["close"].iloc[-1])
                    close_diff_pips = abs(primary_close - candidate_close) / pip
                    time_diff_seconds = abs((pd.Timestamp(candidate_last_time) - pd.Timestamp(primary_last_time)).total_seconds())
                    candidate_staleness_seconds = max(0.0, (observed_at - pd.Timestamp(candidate_last_time).to_pydatetime()).total_seconds())

                    row.update(
                        {
                            "ok": True,
                            "primary_latency_seconds": round(primary_latency, 6),
                            "candidate_latency_seconds": round(candidate_latency, 6),
                            "primary_last_time": _to_iso(primary_last_time),
                            "candidate_last_time": _to_iso(candidate_last_time),
                            "primary_close": primary_close,
                            "candidate_close": candidate_close,
                            "close_diff_pips": round(close_diff_pips, 4),
                            "time_diff_seconds": round(time_diff_seconds, 3),
                            "candidate_staleness_seconds": round(candidate_staleness_seconds, 3),
                            "close_diff_alert": close_diff_pips > self.settings.max_close_diff_pips,
                            "staleness_alert": candidate_staleness_seconds > self.settings.max_staleness_seconds,
                            "primary_rows": int(len(primary)),
                            "candidate_rows": int(len(candidate)),
                        }
                    )
                except Exception as exc:
                    row.update({"ok": False, "error": str(exc)})
                    logger.warning("Market data shadow failed for %s %s: %s", pair, timeframe, exc)
                rows.append(row)
                self._write(row)
        return rows

    def compare_signals(
        self,
        primary_signals: Iterable[TradeSignal],
        candidate_signals: Iterable[TradeSignal],
    ) -> dict[str, object]:
        if not self.settings.enabled or not self.settings.compare_signals:
            return {}

        primary_list = list(primary_signals)
        candidate_list = list(candidate_signals)
        primary_keys = {_signal_key(signal): signal for signal in primary_list}
        candidate_keys = {_signal_key(signal): signal for signal in candidate_list}
        matched = sorted(set(primary_keys) & set(candidate_keys))
        primary_only = sorted(set(primary_keys) - set(candidate_keys))
        candidate_only = sorted(set(candidate_keys) - set(primary_keys))
        payload: dict[str, object] = {
            "type": "signal_shadow",
            "observed_at": _utc_now().isoformat(),
            "primary_source": self.settings.primary_source,
            "candidate_source": self.settings.candidate_source,
            "primary_count": len(primary_list),
            "candidate_count": len(candidate_list),
            "matched_count": len(matched),
            "primary_only_count": len(primary_only),
            "candidate_only_count": len(candidate_only),
            "primary_only": primary_only[:20],
            "candidate_only": candidate_only[:20],
        }
        self._write(payload)
        return payload
