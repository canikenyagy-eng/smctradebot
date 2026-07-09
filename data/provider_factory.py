"""
Market data provider factory.

This module provides a unified interface for accessing market data
from different providers (Yahoo, MetaTrader5, etc.)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.market_data_base import (
    MarketDataProvider,
    get_provider,
    list_providers,
)

# Import providers to register them
from data.itick_provider import ItickConfig, ItickMarketDataProvider
from data.live_bar_provider import LiveBarMarketDataProvider, LiveBarProviderConfig
from data.mt5_provider import MT5MarketDataProvider
from data.redundant_provider import RedundantMarketDataProvider, RedundantProviderConfig
from data.yahoo_provider import YahooMarketDataProvider

logger = logging.getLogger(__name__)


class MarketDataProviderError(Exception):
    """Error when market data provider fails."""
    pass


class ProviderNotAvailableError(Exception):
    """Error when requested provider is not available."""
    pass


class MarketDataManager:
    """Unified market data manager.

    This class selects and manages the active market data provider
    based on configuration.
    """

    _instance: Optional["MarketDataManager"] = None

    def __init__(
        self,
        provider_name: str = "yahoo",
        provider_config: Optional[dict] = None,
        history_limit: int = 500,
    ) -> None:
        """Initialize market data manager.

        Args:
            provider_name: Name of provider to use ("yahoo", "mt5")
            provider_config: Provider-specific configuration
            history_limit: Default history limit
        """
        self.provider_name = provider_name
        self.provider_config = provider_config or {}
        self.history_limit = history_limit
        self._provider: Optional[MarketDataProvider] = None

    def _create_provider(self) -> MarketDataProvider:
        """Create the provider instance.

        Returns:
            Provider instance

        Raises:
            ProviderNotAvailableError: If provider not found
        """
        provider_class = get_provider(self.provider_name)
        if provider_class is None:
            raise ProviderNotAvailableError(
                f"Provider '{self.provider_name}' not available. "
                f"Available: {list_providers()}"
            )

        # Create provider with config
        if self.provider_name == "mt5":
            from data.mt5_provider import MT5Config

            config = MT5Config(
                login=self.provider_config.get("login", 0),
                password=self.provider_config.get("password", ""),
                server=self.provider_config.get("server", ""),
                path=self.provider_config.get("path", ""),
            )
            return provider_class(config=config, history_limit=self.history_limit)
        if self.provider_name == "itick":
            return provider_class(
                config=ItickConfig.from_dict(self.provider_config),
                history_limit=self.history_limit,
            )
        if self.provider_name == "live_bars":
            return provider_class(
                config=LiveBarProviderConfig.from_dict(self.provider_config),
                history_limit=self.history_limit,
            )
        if self.provider_name == "redundant":
            return provider_class(
                config=RedundantProviderConfig.from_dict(self.provider_config),
                history_limit=self.history_limit,
            )
        else:
            return provider_class(history_limit=self.history_limit)

    def get_provider(self) -> MarketDataProvider:
        """Get the active provider.

        Returns:
            Market data provider

        Raises:
            ProviderNotAvailableError: If provider creation fails
        """
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Maximum bars

        Returns:
            OHLCV DataFrame
        """
        provider = self.get_provider()
        return provider.fetch_ohlcv(symbol, timeframe, limit)

    def fetch_ticks(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch tick data.

        Args:
            symbol: Trading symbol
            limit: Maximum ticks

        Returns:
            Tick DataFrame
        """
        provider = self.get_provider()
        return provider.fetch_ticks(symbol, limit)

    def health_check(self) -> bool:
        """Check provider health.

        Returns:
            True if provider is healthy
        """
        try:
            provider = self.get_provider()
            return provider.health_check()
        except Exception as e:
            logger.warning(f"Health check failed for {self.provider_name}: {e}")
            return False

    def close(self) -> None:
        """Close provider connections."""
        if self._provider is not None:
            self._provider.close()
            self._provider = None

    @classmethod
    def get_instance(
        cls,
        provider_name: str = "yahoo",
        provider_config: Optional[dict] = None,
        history_limit: int = 500,
    ) -> "MarketDataManager":
        """Get singleton instance.

        Args:
            provider_name: Name of provider to use
            provider_config: Provider-specific configuration
            history_limit: Default history limit

        Returns:
            MarketDataManager instance
        """
        if cls._instance is None:
            cls._instance = cls(
                provider_name=provider_name,
                provider_config=provider_config,
                history_limit=history_limit,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._provider is not None and self._provider.is_initialized


def get_default_manager(
    data_source: str = "yahoo",
    mt5_config: Optional[dict] = None,
    itick_config: Optional[dict] = None,
    live_bar_config: Optional[dict] = None,
    redundant_config: Optional[dict] = None,
    history_limit: int = 500,
) -> MarketDataManager:
    """Get default market data manager.

    Args:
        data_source: Data source name ("yahoo", "mt5", or "itick")
        mt5_config: MT5 configuration (if using mt5)
        itick_config: iTick configuration (if using itick)
        history_limit: Default history limit

    Returns:
        MarketDataManager instance
    """
    if data_source == "mt5":
        provider_config = mt5_config
    elif data_source == "live_bars":
        provider_config = live_bar_config
    elif data_source == "redundant":
        provider_config = redundant_config
    else:
        provider_config = itick_config
    return MarketDataManager(
        provider_name=data_source,
        provider_config=provider_config,
        history_limit=history_limit,
    )
