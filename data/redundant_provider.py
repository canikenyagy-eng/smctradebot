"""Redundant market data provider with freshness-aware failover."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.itick_provider import ItickConfig, ItickMarketDataProvider
from data.live_bar_provider import LiveBarMarketDataProvider, LiveBarProviderConfig
from data.market_data_base import MarketDataProvider, TIMEFRAME_MAP, register_provider
from data.mt5_provider import MT5Config, MT5MarketDataProvider
from data.yahoo_provider import YahooMarketDataProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedundantProviderConfig:
    primary_source: str = "live_bars"
    backup_sources: tuple[str, ...] = ()
    require_fresh: bool = True
    max_candle_age_seconds: float = 1800.0
    fail_closed: bool = True
    log_path: Path | str = Path("logs/market_data_redundancy.jsonl")
    itick_config: ItickConfig | None = None
    live_bar_config: LiveBarProviderConfig | None = None
    mt5_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RedundantProviderConfig":
        data = payload or {}
        return cls(
            primary_source=str(data.get("primary_source", "live_bars")).strip().lower() or "live_bars",
            backup_sources=_parse_sources(data.get("backup_sources")),
            require_fresh=_parse_bool(data.get("require_fresh"), default=True),
            max_candle_age_seconds=max(60.0, float(data.get("max_candle_age_seconds", 1800))),
            fail_closed=_parse_bool(data.get("fail_closed"), default=True),
            log_path=Path(str(data.get("log_path", "logs/market_data_redundancy.jsonl")).strip()),
            itick_config=ItickConfig.from_dict(data.get("itick_config") if isinstance(data.get("itick_config"), dict) else None),
            live_bar_config=LiveBarProviderConfig.from_dict(
                data.get("live_bar_config") if isinstance(data.get("live_bar_config"), dict) else None
            ),
            mt5_config=data.get("mt5_config") if isinstance(data.get("mt5_config"), dict) else None,
        )


def _parse_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_sources(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    sources = []
    for item in raw_items:
        source = str(item).strip().lower()
        if source and source not in sources:
            sources.append(source)
    return tuple(sources)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timeframe_seconds(timeframe: str) -> int:
    mapping = {
        "M1": 60,
        "M5": 300,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H4": 14400,
        "D1": 86400,
    }
    return mapping.get(timeframe.upper(), 300)


class RedundantMarketDataProvider(MarketDataProvider):
    """Try configured providers in order and fail closed if none are fresh."""

    def __init__(self, config: RedundantProviderConfig | None = None, history_limit: int = 500) -> None:
        super().__init__(history_limit)
        self.config = config or RedundantProviderConfig.from_dict(None)
        self._providers: dict[str, MarketDataProvider] = {}
        self._initialized = True

    def _source_order(self) -> tuple[str, ...]:
        sources = [self.config.primary_source, *self.config.backup_sources]
        ordered = []
        for source in sources:
            normalized = source.strip().lower()
            if normalized and normalized != "redundant" and normalized not in ordered:
                ordered.append(normalized)
        return tuple(ordered)

    def _get_provider(self, source: str) -> MarketDataProvider:
        provider = self._providers.get(source)
        if provider is not None:
            return provider

        if source == "live_bars":
            provider = LiveBarMarketDataProvider(config=self.config.live_bar_config, history_limit=self.history_limit)
        elif source == "itick":
            provider = ItickMarketDataProvider(config=self.config.itick_config, history_limit=self.history_limit)
        elif source == "yahoo":
            provider = YahooMarketDataProvider(history_limit=self.history_limit)
        elif source == "mt5":
            mt5_payload = self.config.mt5_config or {}
            provider = MT5MarketDataProvider(
                config=MT5Config(
                    login=int(mt5_payload.get("login", 0) or 0),
                    password=str(mt5_payload.get("password", "")),
                    server=str(mt5_payload.get("server", "")),
                    path=str(mt5_payload.get("path", "")),
                ),
                history_limit=self.history_limit,
            )
        else:
            raise ValueError(f"Unsupported redundant market data source: {source}")

        self._providers[source] = provider
        return provider

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        tf_key = timeframe.upper()
        if tf_key not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        attempts: list[dict[str, object]] = []
        last_frame: pd.DataFrame | None = None
        last_error: Exception | None = None
        for source in self._source_order():
            started_at = time.monotonic()
            try:
                provider = self._get_provider(source)
                frame = provider.fetch_ohlcv(symbol, tf_key, limit)
                frame = self._validate_dataframe(frame, source=source)
                freshness = self._freshness(frame, tf_key)
                if self.config.require_fresh and not freshness["fresh"]:
                    raise ConnectionError(
                        f"{source} stale for {symbol} {tf_key}: "
                        f"age={freshness['age_seconds']:.1f}s max={freshness['max_age_seconds']:.1f}s"
                    )
                attempts.append(
                    self._attempt_payload(
                        source=source,
                        ok=True,
                        selected=True,
                        started_at=started_at,
                        freshness=freshness,
                        rows=len(frame),
                    )
                )
                self._write_event(symbol=symbol, timeframe=tf_key, selected_source=source, attempts=attempts)
                return frame.tail(limit or self.history_limit).copy()
            except Exception as exc:
                last_error = exc
                attempts.append(
                    self._attempt_payload(
                        source=source,
                        ok=False,
                        selected=False,
                        started_at=started_at,
                        error=exc,
                    )
                )
                logger.warning("Redundant market data source %s failed for %s %s: %s", source, symbol, tf_key, exc)
                if isinstance(exc, ConnectionError):
                    last_frame = None

        self._write_event(symbol=symbol, timeframe=tf_key, selected_source=None, attempts=attempts)
        if self.config.fail_closed or last_frame is None:
            raise ConnectionError(
                f"No fresh redundant market data for {symbol} {tf_key}; "
                f"attempts={len(attempts)} last_error={last_error}"
            )
        return last_frame.tail(limit or self.history_limit).copy()

    def _freshness(self, frame: pd.DataFrame, timeframe: str) -> dict[str, object]:
        latest = pd.Timestamp(frame.index[-1])
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        else:
            latest = latest.tz_convert("UTC")
        seconds = _timeframe_seconds(timeframe)
        freshness_time = latest.to_pydatetime() + timedelta(seconds=seconds)
        age_seconds = max(0.0, (_utc_now() - freshness_time).total_seconds())
        max_age_seconds = max(float(self.config.max_candle_age_seconds), seconds * 2.0)
        return {
            "last_candle_time": latest.isoformat(),
            "freshness_time": freshness_time.isoformat(),
            "age_seconds": age_seconds,
            "max_age_seconds": max_age_seconds,
            "fresh": age_seconds <= max_age_seconds,
        }

    def _attempt_payload(
        self,
        *,
        source: str,
        ok: bool,
        selected: bool,
        started_at: float,
        freshness: dict[str, object] | None = None,
        rows: int | None = None,
        error: Exception | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": source,
            "ok": bool(ok),
            "selected": bool(selected),
            "latency_seconds": round(max(0.0, time.monotonic() - started_at), 6),
        }
        if rows is not None:
            payload["rows"] = rows
        if freshness:
            payload.update(freshness)
        if error is not None:
            payload["error_type"] = error.__class__.__name__
            payload["error"] = str(error)
        return payload

    def _write_event(
        self,
        *,
        symbol: str,
        timeframe: str,
        selected_source: str | None,
        attempts: list[dict[str, object]],
    ) -> None:
        path = Path(self.config.log_path)
        if not str(path):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "type": "market_data_redundancy",
            "version": 1,
            "observed_at": _utc_now().isoformat(),
            "symbol": symbol.upper().replace("/", ""),
            "timeframe": timeframe.upper(),
            "primary_source": self.config.primary_source,
            "backup_sources": list(self.config.backup_sources),
            "selected_source": selected_source,
            "attempts": attempts,
            "ok": selected_source is not None,
            "fail_closed": self.config.fail_closed,
            "require_fresh": self.config.require_fresh,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def health_check(self) -> bool:
        for source in self._source_order():
            try:
                if self._get_provider(source).health_check():
                    return True
            except Exception as exc:
                logger.debug("Redundant provider health check failed for %s: %s", source, exc)
        return False

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()
        self._providers.clear()
        super().close()


register_provider("redundant", RedundantMarketDataProvider)
