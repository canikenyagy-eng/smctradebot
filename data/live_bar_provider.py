"""Market data provider backed by locally built live OHLCV bars."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.itick_provider import ItickConfig, ItickMarketDataProvider
from data.market_data_base import MarketDataProvider, TIMEFRAME_MAP, register_provider
from data.yahoo_provider import YahooMarketDataProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveBarProviderConfig:
    bars_dir: Path | str = Path("data/live_bars/itick")
    fallback_source: str = "yahoo"
    include_current_bar: bool = False
    require_live_overlay: bool = False
    max_live_bar_age_seconds: float = 7200.0
    itick_config: ItickConfig | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LiveBarProviderConfig":
        data = payload or {}
        return cls(
            bars_dir=Path(str(data.get("bars_dir", "data/live_bars/itick")).strip()),
            fallback_source=str(data.get("fallback_source", "yahoo")).strip().lower(),
            include_current_bar=_parse_bool(data.get("include_current_bar"), default=False),
            require_live_overlay=_parse_bool(data.get("require_live_overlay"), default=False),
            max_live_bar_age_seconds=max(60.0, float(data.get("max_live_bar_age_seconds", 7200))),
            itick_config=ItickConfig.from_dict(data.get("itick_config") if isinstance(data.get("itick_config"), dict) else None),
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


def _clean_pair(pair: str) -> str:
    clean = pair.upper().replace("/", "").strip()
    if len(clean) != 6:
        raise ValueError(f"Unsupported forex symbol: {pair}")
    return clean


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LiveBarMarketDataProvider(MarketDataProvider):
    """Merges historical fallback candles with locally generated iTick live bars."""

    def __init__(self, config: LiveBarProviderConfig | None = None, history_limit: int = 500) -> None:
        super().__init__(history_limit)
        self.config = config or LiveBarProviderConfig.from_dict(None)
        self._fallback_provider = self._build_fallback_provider()
        self._initialized = True

    def _build_fallback_provider(self) -> MarketDataProvider | None:
        source = self.config.fallback_source
        if source in {"", "none", "disabled", "off"}:
            return None
        if source == "yahoo":
            return YahooMarketDataProvider(history_limit=self.history_limit)
        if source == "itick":
            return ItickMarketDataProvider(config=self.config.itick_config, history_limit=self.history_limit)
        raise ValueError(f"Unsupported LIVE_BAR_PROVIDER_FALLBACK_SOURCE: {source}")

    def _path(self, pair: str, timeframe: str) -> Path:
        return Path(self.config.bars_dir) / f"{_clean_pair(pair)}_{timeframe.upper()}.csv"

    def _read_live_bars(self, pair: str, timeframe: str) -> pd.DataFrame:
        path = self._path(pair, timeframe)
        if not path.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        raw = pd.read_csv(path)
        if raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if "timestamp" not in raw.columns:
            raise ValueError(f"Live bar file missing timestamp column: {path}")

        if not self.config.include_current_bar and "complete" in raw.columns:
            complete = raw["complete"].astype(str).str.lower().isin({"true", "1", "yes"})
            raw = raw[complete]

        if raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        frame = raw.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
        for column in ("open", "high", "low", "close", "volume"):
            if column not in frame.columns:
                raise ValueError(f"Live bar file missing {column} column: {path}")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[["open", "high", "low", "close", "volume"]].dropna()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame

    def _fetch_fallback(self, pair: str, timeframe: str, limit: int | None) -> pd.DataFrame:
        if self._fallback_provider is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        try:
            return self._fallback_provider.fetch_ohlcv(pair, timeframe, limit)
        except Exception as exc:
            if self.config.require_live_overlay:
                logger.warning("Live-bar fallback failed for %s %s: %s", pair, timeframe, exc)
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            raise

    def _validate_live_overlay(self, live: pd.DataFrame, pair: str, timeframe: str) -> None:
        if live.empty:
            if self.config.require_live_overlay:
                raise ConnectionError(f"No live bar overlay for {pair} {timeframe}")
            return

        latest = pd.Timestamp(live.index[-1]).to_pydatetime()
        timeframe_seconds = _timeframe_seconds(timeframe)
        freshness_time = latest if self.config.include_current_bar else latest + timedelta(seconds=timeframe_seconds)
        age_seconds = max(0.0, (_utc_now() - freshness_time).total_seconds())
        max_age = max(self.config.max_live_bar_age_seconds, timeframe_seconds * 2.0)
        if self.config.require_live_overlay and age_seconds > max_age:
            raise ConnectionError(
                f"Live bar overlay stale for {pair} {timeframe}: "
                f"age={age_seconds:.1f}s max={max_age:.1f}s"
            )

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        tf_key = timeframe.upper()
        if tf_key not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        max_rows = limit or self.history_limit
        fallback = self._fetch_fallback(symbol, tf_key, max_rows)
        live = self._read_live_bars(symbol, tf_key)
        self._validate_live_overlay(live, symbol, tf_key)

        frames = [frame for frame in (fallback, live) if frame is not None and not frame.empty]
        if not frames:
            raise ValueError(f"No live-bar market data for {symbol} {tf_key}")

        merged = pd.concat(frames)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged = self._validate_dataframe(merged, source="live_bars")
        self._log_data_integrity(merged, symbol, tf_key)
        return merged.tail(max_rows).copy()

    def health_check(self) -> bool:
        try:
            if self._fallback_provider is not None and not self._fallback_provider.health_check():
                return False
            return True
        except Exception as exc:
            logger.warning("Live-bar provider health check failed: %s", exc)
            return False

    def close(self) -> None:
        if self._fallback_provider is not None:
            self._fallback_provider.close()
        super().close()


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


register_provider("live_bars", LiveBarMarketDataProvider)
