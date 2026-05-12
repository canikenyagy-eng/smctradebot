"""
Yahoo Finance market data provider.

This module provides market data from Yahoo Finance.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
import yfinance as yf

from data.market_data_base import (
    MarketDataProvider,
    TimeframeConfig,
    TIMEFRAME_MAP,
    register_provider,
)

logger = logging.getLogger(__name__)


class YahooMarketDataProvider(MarketDataProvider):
    """Market data provider using Yahoo Finance.

    This provider fetches OHLCV data from Yahoo Finance.
    """

    def __init__(self, history_limit: int = 500) -> None:
        """Initialize Yahoo provider.

        Args:
            history_limit: Default maximum number of bars to return
        """
        super().__init__(history_limit)
        self._initialized = True  # Yahoo doesn't need connection

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        """Convert SMC symbol to Yahoo format.

        Args:
            pair: Symbol pair (e.g., "EURUSD")

        Returns:
            Yahoo ticker (e.g., "EURUSD=X")
        """
        clean_pair = pair.upper().replace("/", "")
        if len(clean_pair) != 6:
            raise ValueError(f"Unsupported forex symbol: {pair}")
        return f"{clean_pair}=X"

    @staticmethod
    def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame to system format.

        Args:
            frame: Raw DataFrame from Yahoo

        Returns:
            Standardized DataFrame
        """
        if frame.empty:
            return frame

        # Handle MultiIndex columns
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        # Rename columns
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        frame = frame.rename(columns=rename_map)

        # Check required columns
        expected = ["open", "high", "low", "close"]
        for col in expected:
            if col not in frame.columns:
                raise ValueError(f"Missing column '{col}' in downloaded data")

        # Add volume if missing
        if "volume" not in frame.columns:
            frame["volume"] = 0.0

        # Select and clean columns
        frame = frame[["open", "high", "low", "close", "volume"]].dropna()

        # Handle timezone
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")

        return frame.sort_index()

    @staticmethod
    def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample DataFrame to different timeframe.

        Args:
            frame: Source DataFrame
            rule: Pandas resample rule

        Returns:
            Resampled DataFrame
        """
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

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo.

        Args:
            pair: Trading pair (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "M5", "H1")
            limit: Maximum number of bars to return

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index must be datetime (UTC)

        Raises:
            ValueError: If pair/timeframe is invalid
        """
        tf_key = timeframe.upper()
        if tf_key not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        tf_cfg = TIMEFRAME_MAP[tf_key]
        ticker = self._normalize_pair(pair)

        # Download from Yahoo
        raw = yf.download(
            tickers=ticker,
            interval=tf_cfg.interval,
            period=tf_cfg.period,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        frame = self._standardize_frame(raw)

        # Resample if needed
        if tf_cfg.resample_rule:
            frame = self._resample(frame, tf_cfg.resample_rule)

        if frame.empty:
            raise ValueError(f"No market data for {pair} {timeframe}")

        # Validate
        frame = self._validate_dataframe(frame, source="yahoo")
        self._log_data_integrity(frame, pair, timeframe)

        # Apply limit
        max_rows = limit or self.history_limit
        return frame.tail(max_rows).copy()

    def health_check(self) -> bool:
        """Check Yahoo provider health.

        Returns:
            Always True (Yahoo doesn't need connection check)
        """
        return True

    def close(self) -> None:
        """Clean up provider resources."""
        super().close()


# Auto-register provider
register_provider("yahoo", YahooMarketDataProvider)