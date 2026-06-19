"""
Market data client with provider selection and local OHLCV caching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

from data.provider_factory import get_default_manager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeframeConfig:
    interval: str
    period: str
    resample_rule: str | None = None


@dataclass(frozen=True)
class MarketDataCacheConfig:
    enabled: bool = True
    cache_dir: Path | str = Path("data/cache/ohlcv")
    ttl_hours: float = 12.0
    mode: str = "read_through"

    def sanitized(self) -> "MarketDataCacheConfig":
        mode = self.mode.strip().lower()
        if mode not in {"read_through", "cache_only", "refresh", "disabled"}:
            mode = "read_through"
        return MarketDataCacheConfig(
            enabled=self.enabled and mode != "disabled",
            cache_dir=Path(self.cache_dir),
            ttl_hours=max(0.0, float(self.ttl_hours)),
            mode=mode,
        )


TIMEFRAME_MAP: Dict[str, TimeframeConfig] = {
    "M1": TimeframeConfig(interval="1m", period="7d"),
    "M5": TimeframeConfig(interval="5m", period="60d"),
    "M15": TimeframeConfig(interval="15m", period="60d"),
    "M30": TimeframeConfig(interval="30m", period="60d"),
    "H1": TimeframeConfig(interval="60m", period="730d"),
    "H4": TimeframeConfig(interval="60m", period="730d", resample_rule="4h"),
    "D1": TimeframeConfig(interval="1d", period="10y"),
}


class MarketDataClient:
    """Backward-compatible market data client with cache-aware provider access."""

    def __init__(
        self,
        history_limit: int = 500,
        data_source: str = "yahoo",
        mt5_login: int | None = 0,
        mt5_password: str = "",
        mt5_server: str = "",
        mt5_path: str = "",
        cache_config: MarketDataCacheConfig | None = None,
    ) -> None:
        self.history_limit = history_limit
        self.data_source = data_source.strip().lower()
        self.cache_config = (cache_config or MarketDataCacheConfig()).sanitized()

        mt5_config = None
        if self.data_source == "mt5" and (mt5_login or 0) > 0:
            mt5_config = {
                "login": int(mt5_login or 0),
                "password": mt5_password,
                "server": mt5_server,
                "path": mt5_path,
            }

        self._manager = get_default_manager(
            data_source=self.data_source,
            mt5_config=mt5_config,
            history_limit=history_limit,
        )

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        clean_pair = pair.upper().replace("/", "")
        if len(clean_pair) != 6:
            raise ValueError(f"Unsupported forex symbol: {pair}")
        return f"{clean_pair}=X"

    @staticmethod
    def _clean_pair(pair: str) -> str:
        clean_pair = pair.upper().replace("/", "")
        if len(clean_pair) != 6:
            raise ValueError(f"Unsupported forex symbol: {pair}")
        return clean_pair

    @staticmethod
    def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        frame = frame.rename(columns=rename_map)
        expected = ["open", "high", "low", "close"]
        for col in expected:
            if col not in frame.columns:
                raise ValueError(f"Missing column '{col}' in downloaded data")

        if "volume" not in frame.columns:
            frame["volume"] = 0.0

        frame = frame[["open", "high", "low", "close", "volume"]].dropna()
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")

        return frame.sort_index()

    @staticmethod
    def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        if frame.empty:
            return frame

        return (
            frame.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    def _cache_path(self, pair: str, timeframe: str) -> Path:
        symbol = self._clean_pair(pair)
        return Path(self.cache_config.cache_dir) / f"{symbol}_{timeframe.upper()}.csv"

    @staticmethod
    def _coerce_end_time(end_time: object | None) -> pd.Timestamp | None:
        if end_time is None:
            return None
        text = str(end_time).strip()
        if not text:
            return None
        timestamp = pd.Timestamp(text)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.cache_config.ttl_hours <= 0:
            return True
        age_seconds = max(0.0, time.time() - path.stat().st_mtime)
        return age_seconds <= self.cache_config.ttl_hours * 3600.0

    def _read_cache(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if frame.empty:
            raise ValueError(f"Cached market data is empty: {path}")
        frame = self._standardize_frame(frame)
        if frame.empty:
            raise ValueError(f"Cached market data is invalid: {path}")
        return frame

    def _write_cache(self, path: Path, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(tmp_path, index_label="timestamp")
        tmp_path.replace(path)

    def _download_yahoo_ohlcv(self, pair: str, timeframe: str) -> pd.DataFrame:
        tf_cfg = TIMEFRAME_MAP[timeframe]
        ticker = self._normalize_pair(pair)
        raw = yf.download(
            tickers=ticker,
            interval=tf_cfg.interval,
            period=tf_cfg.period,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        frame = self._standardize_frame(raw)
        if tf_cfg.resample_rule:
            frame = self._resample(frame, tf_cfg.resample_rule)
        return frame

    def _fetch_provider_ohlcv(self, pair: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        if self.data_source == "mt5":
            try:
                return self._manager.fetch_ohlcv(pair, timeframe, limit)
            except Exception as exc:
                logger.warning("MT5 fetch failed for %s %s, falling back to Yahoo: %s", pair, timeframe, exc)
        return self._download_yahoo_ohlcv(pair, timeframe)

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        limit: int | None = None,
        end_time: object | None = None,
    ) -> pd.DataFrame:
        tf_key = timeframe.upper()
        if tf_key not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        cache = self.cache_config
        cache_path = self._cache_path(pair, tf_key)
        cached_frame: pd.DataFrame | None = None

        if cache.enabled and cache_path.exists():
            try:
                cached_frame = self._read_cache(cache_path)
            except Exception as exc:
                logger.warning("Failed to read OHLCV cache %s: %s", cache_path, exc)

        if cache.enabled and cache.mode == "cache_only":
            if cached_frame is None:
                raise ValueError(f"No cached market data for {pair} {timeframe}: {cache_path}")
            frame = cached_frame
        elif cache.enabled and cache.mode != "refresh" and cached_frame is not None and self._cache_is_fresh(cache_path):
            frame = cached_frame
        else:
            try:
                frame = self._fetch_provider_ohlcv(pair, tf_key, limit)
                frame = self._standardize_frame(frame)
                if cache.enabled and not frame.empty:
                    self._write_cache(cache_path, frame)
            except Exception as exc:
                if cached_frame is None:
                    raise
                logger.warning(
                    "Using stale OHLCV cache for %s %s after provider failure: %s",
                    pair,
                    timeframe,
                    exc,
                )
                frame = cached_frame

        if frame.empty:
            if cached_frame is not None:
                logger.warning("Using cached OHLCV for %s %s because fresh data was empty", pair, timeframe)
                frame = cached_frame
            else:
                raise ValueError(f"No market data for {pair} {timeframe}")

        cutoff = self._coerce_end_time(end_time)
        if cutoff is not None:
            frame = frame[frame.index <= cutoff]
            if frame.empty:
                raise ValueError(f"No market data for {pair} {timeframe} at or before {cutoff.isoformat()}")

        max_rows = limit or self.history_limit
        return frame.tail(max_rows).copy()

    def close(self) -> None:
        self._manager.close()
