"""
MetaTrader 5 market data provider.

This module provides market data from MetaTrader 5 terminal using the official
MetaTrader5 Python package.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd

from data.market_data_base import (
    MarketDataProvider,
    TimeframeConfig,
    TIMEFRAME_MAP,
    register_provider,
)

logger = logging.getLogger(__name__)

# MT5 timeframe constants
MT5_TIMEFRAMES = {
    "M1": 1,           # M1 (1 minute)
    "M5": 5,           # M5 (5 minutes)
    "M15": 15,         # M15 (15 minutes)
    "M30": 30,         # M30 (30 minutes)
    "H1": 16392,       # H1 (1 hour) - MT5 constant
    "H4": 16396,       # H4 (4 hours) - MT5 constant
    "D1": 16408,      # D1 (1 day) - MT5 constant
}

# Reverse mapping: MT5 constant -> SMC timeframe
MT5_TO_TIMEFRAME = {v: k for k, v in MT5_TIMEFRAMES.items()}


@dataclass(frozen=True)
class MT5Config:
    """Configuration for MT5 connection."""

    path: str = "C:/Program Files/MetaTrader 5/terminal64.exe"
    login: int = 0
    password: str = ""
    server: str = ""
    timeout: int = 60000
    portable: bool = False


class MT5MarketDataProvider(MarketDataProvider):
    """Market data provider using MetaTrader 5.

    This provider fetches OHLCV data from a running MetaTrader 5 terminal.
    """

    def __init__(
        self,
        config: MT5Config | None = None,
        history_limit: int = 500,
        retry_attempts: int = 3,
    ) -> None:
        """Initialize MT5 provider.

        Args:
            config: MT5 connection configuration
            history_limit: Default maximum number of bars to return
            retry_attempts: Number of connection retry attempts
        """
        super().__init__(history_limit)
        self.config = config or MT5Config()
        self.retry_attempts = retry_attempts
        self._mt5 = None
        self._login_result = None

    def _init_mt5(self) -> None:
        """Initialize MetaTrader 5 Python API."""
        # Import here to allow graceful handling if not installed
        try:
            import MetaTrader5 as mt5

            self._mt5 = mt5
        except ImportError:
            raise ImportError(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )

    def _connect(self) -> bool:
        """Connect to MT5 terminal.

        Returns:
            True if connected successfully

        Raises:
            ConnectionError: If connection fails
        """
        if self._mt5 is None:
            self._init_mt5()

        # Initialize MT5
        if not self._mt5.initialize():
            error = self._mt5.last_error()
            logger.error(f"MT5 initialize failed: {error}")
            raise ConnectionError(f"Failed to initialize MT5: {error}")

        # Login if credentials provided
        if self.config.login > 0 and self.config.password and self.config.server:
            login_result = self._mt5.login(
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
                timeout=self.config.timeout,
            )
            if not login_result:
                error = self._mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                self._mt5.shutdown()
                raise ConnectionError(f"Failed to login to MT5: {error}")
            self._login_result = login_result

        logger.info("MT5 connected successfully")
        return True

    def _disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None
            logger.info("MT5 disconnected")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from MT5.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "M5", "H1")
            limit: Maximum number of bars to return

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index must be datetime (UTC)

        Raises:
            ValueError: If symbol/timeframe is invalid
            ConnectionError: If connection fails
        """
        tf_key = timeframe.upper()
        if tf_key not in MT5_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if not self._initialized:
            self._connect()
            self._initialized = True

        mt5_tf = MT5_TIMEFRAMES[tf_key]
        limit = limit or self.history_limit

        # Fetch bars from MT5
        bars = self._mt5.copy_rates_from_pos(
            symbol,
            mt5_tf,
            0,
            limit,
        )

        if bars is None or len(bars) == 0:
            error = self._mt5.last_error()
            logger.error(f"Failed to fetch {symbol} {timeframe}: {error}")
            raise ValueError(f"No data for {symbol} {timeframe}: {error}")

        # Convert to DataFrame
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")

        # Rename columns to standard format
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume",
            }
        )

        # Keep only required columns
        df = df[["open", "high", "low", "close", "volume"]]

        # Ensure UTC timezone
        df.index = df.index.tz_localize("UTC")

        # Validate data
        df = self._validate_dataframe(df, source="mt5")
        self._log_data_integrity(df, symbol, timeframe)

        return df.sort_index()

    def fetch_ticks(
        self,
        symbol: str,
        limit: int = 100,
        start_pos: int = 0,
    ) -> pd.DataFrame:
        """Fetch tick data from MT5.

        Args:
            symbol: Trading symbol
            limit: Maximum number of ticks to return
            start_pos: Starting position (0 = latest)

        Returns:
            DataFrame with tick data

        Raises:
            ConnectionError: If connection fails
        """
        if not self._initialized:
            self._connect()
            self._initialized = True

        ticks = self._mt5.copy_ticks_from(
            symbol,
            start_pos,
            limit,
            self._mt5.COPY_TICKS_ALL,
        )

        if ticks is None or len(ticks) == 0:
            error = self._mt5.last_error()
            raise ValueError(f"No ticks for {symbol}: {error}")

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        df.index = df.index.tz_localize("UTC")

        return df.sort_index()

    def fetch_ticks_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Fetch tick data for a time range.

        Args:
            symbol: Trading symbol
            start_time: Start time (UTC)
            end_time: End time (UTC)

        Returns:
            DataFrame with tick data

        Raises:
            ConnectionError: If connection fails
        """
        if not self._initialized:
            self._connect()
            self._initialized = True

        # Convert to MT5 time (seconds since 1970)
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())

        ticks = self._mt5.copy_ticks_range(
            symbol,
            start_ts,
            end_ts,
            self._mt5.COPY_TICKS_ALL,
        )

        if ticks is None or len(ticks) == 0:
            error = self._mt5.last_error()
            raise ValueError(f"No ticks for {symbol} in range: {error}")

        df = pd.DataFrame(ticks)

        # Normalize columns
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("timestamp")
        df.index = df.index.tz_localize("UTC")

        # Rename and select columns
        rename_map = {
            "bid": "bid",
            "ask": "ask",
            "last": "last",
            "volume": "volume",
            "time": "time",
        }
        df = df.rename(columns=rename_map)

        # Calculate spread
        df["spread"] = df["ask"] - df["bid"]
        df["mid"] = (df["bid"] + df["ask"]) / 2

        return df.sort_index()

    def fetch_recent_ticks(
        self,
        symbol: str,
        lookback_minutes: int = 60,
    ) -> pd.DataFrame:
        """Fetch recent ticks for a symbol.

        Args:
            symbol: Trading symbol
            lookback_minutes: How many minutes of data to fetch

        Returns:
            DataFrame with tick data

        Raises:
            ValueError: If no ticks available
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=lookback_minutes)
        return self.fetch_ticks_range(symbol, start, now)

    def get_symbol_info(self, symbol: str) -> Dict | None:
        """Get symbol information.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with symbol info or None
        """
        if not self._initialized:
            self._connect()
            self._initialized = True

        return self._mt5.symbol_info(symbol)

    def get_symbols(self) -> List[str]:
        """Get available symbols.

        Returns:
            List of available symbol names
        """
        if not self._initialized:
            self._connect()
            self._initialized = True

        symbols = self._mt5.symbols_get()
        return [s.name for s in symbols]

    def health_check(self) -> bool:
        """Check MT5 connection health.

        Returns:
            True if MT5 is connected and healthy
        """
        try:
            if self._mt5 is None:
                return False

            # Check if initialized
            if not self._mt5.initialize():
                return False

            # Try to get account info
            account = self._mt5.account_info()
            return account is not None
        except Exception as e:
            logger.warning(f"MT5 health check failed: {e}")
            return False

    def close(self) -> None:
        """Clean up MT5 connection."""
        self._disconnect()
        super().close()


# Auto-register provider
register_provider("mt5", MT5MarketDataProvider)


def create_mt5_provider(
    login: int = 0,
    password: str = "",
    server: str = "",
    path: str = "",
    history_limit: int = 500,
) -> MT5MarketDataProvider:
    """Create MT5 provider with configuration.

    Args:
        login: MT5 account login number
        password: MT5 account password
        server: MT5 broker server
        path: Path to MT5 terminal executable
        history_limit: Default history limit

    Returns:
        Configured MT5 provider
    """
    config = MT5Config(
        login=login,
        password=password,
        server=server,
        path=path or os.getenv("MT5_PATH", "C:/Program Files/MetaTrader 5/terminal64.exe"),
    )
    return MT5MarketDataProvider(config=config, history_limit=history_limit)


# Live polling support (optional)
def create_live_polling_provider(
    provider: MT5MarketDataProvider,
    symbols: list[str],
    timeframe: str = "M1",
    interval_seconds: int = 60,
    callback=None,
) -> "MT5LivePollingProvider":
    """Create a live polling provider for real-time data.

    Args:
        provider: MT5 provider instance
        symbols: List of symbols to poll
        timeframe: Timeframe for bars
        interval_seconds: Polling interval
        callback: Optional callback function for each update

    Returns:
        Live polling provider
    """
    return MT5LivePollingProvider(
        provider=provider,
        symbols=symbols,
        timeframe=timeframe,
        interval_seconds=interval_seconds,
        callback=callback,
    )


class MT5LivePollingProvider:
    """Live polling provider for real-time bar updates.

    This class periodically polls MT5 for new bars and calls a callback
    with the updated data.
    """

    def __init__(
        self,
        provider: MT5MarketDataProvider,
        symbols: list[str],
        timeframe: str = "M1",
        interval_seconds: int = 60,
        callback=None,
    ) -> None:
        """Initialize live polling provider.

        Args:
            provider: MT5 provider instance
            symbols: List of symbols to poll
            timeframe: Timeframe for bars
            interval_seconds: Polling interval
            callback: Callback function (symbol, dataframe)
        """
        self.provider = provider
        self.symbols = symbols
        self.timeframe = timeframe
        self.interval_seconds = interval_seconds
        self.callback = callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_bars: Dict[str, pd.DataFrame] = {}

    def _poll(self) -> None:
        """Poll for new data."""
        for symbol in self.symbols:
            try:
                # Get latest bars
                df = self.provider.fetch_ohlcv(symbol, self.timeframe, limit=10)

                # Check for new bars
                last_bars = self._last_bars.get(symbol)
                if last_bars is not None and not df.empty:
                    last_ts = df.index[-1]
                    last_ts_old = last_bars.index[-1]
                    if last_ts <= last_ts_old:
                        continue  # No new bars

                # Store and callback
                self._last_bars[symbol] = df
                if self.callback:
                    self.callback(symbol, df)
            except Exception as e:
                logger.warning(f"Poll error for {symbol}: {e}")

    def _run(self) -> None:
        """Run polling loop."""
        while self._running:
            self._poll()
            self._sleep(self.interval_seconds)

    def _sleep(self, seconds: int) -> None:
        """Sleep with interrupt support."""
        for _ in range(seconds * 10):
            if not self._running:
                break
            time.sleep(0.1)

    def start(self) -> None:
        """Start polling."""
        if self._running:
            return
        self._running = True
        self._thread = self._thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        logger.info(f"Live polling started for {self.symbols}")

    def stop(self) -> None:
        """Stop polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Live polling stopped")

    def get_latest(self, symbol: str) -> pd.DataFrame | None:
        """Get latest data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            DataFrame or None
        """
        return self._last_bars.get(symbol)