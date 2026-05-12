"""
Abstract market data provider interface.

This module defines the base interface for market data providers.
All data providers must implement these methods to integrate with the trading system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class TimeframeConfig:
    """Configuration for a timeframe."""

    interval: str
    period: str
    resample_rule: str | None = None


# Standard timeframe mapping used across the system
TIMEFRAME_MAP: Dict[str, TimeframeConfig] = {
    "M1": TimeframeConfig(interval="1m", period="7d"),
    "M5": TimeframeConfig(interval="5m", period="60d"),
    "M15": TimeframeConfig(interval="15m", period="60d"),
    "M30": TimeframeConfig(interval="30m", period="60d"),
    "H1": TimeframeConfig(interval="60m", period="730d"),
    "H4": TimeframeConfig(interval="60m", period="730d", resample_rule="4h"),
    "D1": TimeframeConfig(interval="1d", period="10y"),
}


class MarketDataProvider(ABC):
    """Abstract base class for market data providers.

    All market data providers must implement these methods:
    - fetch_ohlcv: Fetch OHLCV bars for a symbol/timeframe
    - fetch_ticks: Fetch tick data (optional, for live mode)
    - health_check: Verify connection/system health
    """

    def __init__(self, history_limit: int = 500) -> None:
        """Initialize the provider.

        Args:
            history_limit: Default maximum number of bars to return
        """
        self.history_limit = history_limit
        self._initialized = False

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "M5", "H1")
            limit: Maximum number of bars to return (None for default)

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index must be datetime (UTC)

        Raises:
            ValueError: If symbol/timeframe is invalid
            ConnectionError: If provider connection fails
        """
        pass

    def fetch_ticks(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch tick data for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of ticks to return

        Returns:
            DataFrame with tick data

        Raises:
            NotImplementedError: If provider doesn't support ticks
            ConnectionError: If provider connection fails
        """
        raise NotImplementedError("Tick data not supported by this provider")

    @abstractmethod
    def health_check(self) -> bool:
        """Check provider health/connection status.

        Returns:
            True if provider is healthy, False otherwise
        """
        pass

    def _validate_dataframe(self, frame: pd.DataFrame, source: str) -> pd.DataFrame:
        """Validate and clean OHLCV DataFrame.

        Args:
            frame: DataFrame to validate
            source: Source name for error messages

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If validation fails
        """
        if frame.empty:
            raise ValueError(f"No data returned from {source}")

        # Ensure required columns exist
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in frame.columns:
                raise ValueError(f"Missing required column '{col}' from {source}")

        # Ensure volume column exists
        if "volume" not in frame.columns:
            frame = frame.copy()
            frame["volume"] = 0.0

        # Check for missing values in price columns
        if frame[required].isnull().any().any():
            missing_count = frame[required].isnull().sum().sum()
            raise ValueError(f"Found {missing_count} missing values in {source} data")

        # Ensure timestamp ordering
        if not frame.index.is_monotonic_increasing:
            frame = frame.sort_index()

        return frame

    def _log_data_integrity(self, frame: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Log data integrity summary.

        Args:
            frame: DataFrame to check
            symbol: Symbol name
            timeframe: Timeframe name
        """
        if frame.empty:
            return

        # Check for gaps (large time deltas)
        if len(frame) > 1:
            time_deltas = frame.index.to_series().diff().dropna()
            max_gap = time_deltas.max()
            if max_gap > pd.Timedelta(days=1):
                print(
                    f"WARNING: Large gap detected for {symbol} {timeframe}: {max_gap}"
                )

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    def close(self) -> None:
        """Clean up provider resources.

        Override this method to close connections, files, etc.
        """
        self._initialized = False


# Registry of available providers
PROVIDER_REGISTRY: Dict[str, type[MarketDataProvider]] = {}


def register_provider(name: str, provider_class: type[MarketDataProvider]) -> None:
    """Register a market data provider.

    Args:
        name: Provider identifier (e.g., "yahoo", "mt5")
        provider_class: Provider class implementing MarketDataProvider
    """
    PROVIDER_REGISTRY[name] = provider_class


def get_provider(name: str) -> type[MarketDataProvider] | None:
    """Get a provider class by name.

    Args:
        name: Provider identifier

    Returns:
        Provider class or None if not found
    """
    return PROVIDER_REGISTRY.get(name)


def list_providers() -> List[str]:
    """List all registered provider names.

    Returns:
        List of provider names
    """
    return list(PROVIDER_REGISTRY.keys())