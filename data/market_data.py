"""
Market data client with configurable providers.

This module provides backward-compatible access to market data
with support for different data providers (Yahoo, MetaTrader5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from data.market_data_base import MarketDataProvider
from data.provider_factory import MarketDataManager, get_default_manager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeframeConfig:
    interval: str
    period: str
    resample_rule: str | None = None


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
    """Market data client with configurable provider support.

    This class provides backward compatibility while supporting
    different data providers via configuration.
    """

    def __init__(
        self,
        history_limit: int = 500,
        data_source: str = "yahoo",
        mt5_login: int = 0,
        mt5_password: str = "",
        mt5_server: str = "",
    ) -> None:
        """Initialize market data client.

        Args:
            history_limit: Default maximum bars to return
            data_source: Data provider ("yahoo" or "mt5")
            mt5_login: MT5 login (if using mt5)
            mt5_password: MT5 password (if using mt5)
            mt5_server: MT5 server (if using mt5)
        """
        self.history_limit = history_limit
        self.data_source = data_source

        # Configure MT5 if needed
        mt5_config = None
        if data_source == "mt5" and mt5_login > 0:
            mt5_config = {
                "login": mt5_login,
                "password": mt5_password,
                "server": mt5_server,
            }

        # Create provider manager
        self._manager = get_default_manager(
            data_source=data_source,
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

    def fetch_ohlcv(self, pair: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        """Fetch OHLCV data for a trading pair.

        Args:
            pair: Trading pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "M5", "H1")
            limit: Maximum bars to return

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If timeframe is invalid or no data
        """
        # Use provider manager for MT5
        if self.data_source == "mt5":
            try:
                return self._manager.fetch_ohlcv(pair, timeframe, limit)
            except Exception as e:
                logger.warning(f"MT5 fetch failed: {e}, falling back to Yahoo")
                # Fall back to Yahoo

        # Default: use Yahoo (existing behavior)
        tf_key = timeframe.upper()
        if tf_key not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        tf_cfg = TIMEFRAME_MAP[tf_key]
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

        if frame.empty:
            raise ValueError(f"No market data for {pair} {timeframe}")

        max_rows = limit or self.history_limit
        return frame.tail(max_rows).copy()
