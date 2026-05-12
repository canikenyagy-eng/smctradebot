"""
Tick-Level Execution Engine using MT5 data.

This module provides realistic trade execution simulation using tick-level
market data from MetaTrader 5. It replaces candle-close execution
assumptions with tick-level market simulation.

Features:
- Market, Limit, and Stop order support
- Real-time spread derivation from bid/ask
- Stochastic slippage modeling
- Execution latency simulation
- Partial fill handling
- Trade state machine
- PnL realism with all costs
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ORDER STATE MACHINE
# =============================================================================

class OrderState(Enum):
    """Order lifecycle states."""
    PENDING = auto()      # Order created, waiting for trigger
    TRIGGERED = auto()    # Stop/limit condition met
    PARTIALLY_FILLED = auto()  # Partial fill received
    FILLED = auto()        # Fully filled
    CLOSED = auto()        # Trade closed (exit)
    REJECTED = auto()     # Order rejected
    CANCELLED = auto()    # Order cancelled


class OrderType(Enum):
    """Order types."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()


class TradeDirection(Enum):
    """Trade direction."""
    BUY = auto()
    SELL = auto()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TickExecutionSettings:
    """Configuration for tick execution engine."""

    # Enable tick-level execution
    enable_tick_execution: bool = False

    # Slippage modeling
    enable_realistic_slippage: bool = False
    max_slippage_pips: float = 1.0
    base_slippage_pips: float = 0.2

    # Partial fills
    enable_partial_fills: bool = False
    partial_fill_min_ratio: float = 0.25
    partial_fill_max_ratio: float = 1.0
    partial_fill_volume_threshold: float = 1000.0

    # Execution latency
    execution_latency_ticks: int = 0
    execution_latency_ms: int = 0

    # Random seed for determinism
    random_seed: int | None = None

    # Spread fallback
    fallback_spread_pips: float = 2.0

    # Tick data source
    tick_data_source: str = "mt5"

    # Debug options
    log_slippage_samples: bool = False
    log_fill_details: bool = False

    def sanitized(self) -> "TickExecutionSettings":
        """Return sanitized settings."""
        return TickExecutionSettings(
            enable_tick_execution=bool(self.enable_tick_execution),
            enable_realistic_slippage=bool(self.enable_realistic_slippage),
            max_slippage_pips=max(0.0, float(self.max_slippage_pips)),
            base_slippage_pips=max(0.0, float(self.base_slippage_pips)),
            enable_partial_fills=bool(self.enable_partial_fills),
            partial_fill_min_ratio=max(0.01, min(0.99, float(self.partial_fill_min_ratio))),
            partial_fill_max_ratio=max(0.01, min(1.0, float(self.partial_fill_max_ratio))),
            partial_fill_volume_threshold=max(0.0, float(self.partial_fill_volume_threshold)),
            execution_latency_ticks=max(0, int(self.execution_latency_ticks)),
            execution_latency_ms=max(0, int(self.execution_latency_ms)),
            random_seed=self.random_seed,
            fallback_spread_pips=max(0.0, float(self.fallback_spread_pips)),
            tick_data_source=self.tick_data_source.strip().lower(),
            log_slippage_samples=bool(self.log_slippage_samples),
            log_fill_details=bool(self.log_fill_details),
        )


@dataclass
class ExecutionMetrics:
    """Metrics for execution quality."""

    # Slippage
    total_slippage_pips: float = 0.0
    slippage_trades: int = 0
    avg_slippage_pips: float = 0.0
    max_slippage_pips: float = 0.0
    min_slippage_pips: float = 0.0

    # Spread costs
    total_spread_cost_pips: float = 0.0
    spread_trades: int = 0
    avg_spread_pips: float = 0.0

    # Fill quality
    total_fills: int = 0
    partial_fills: int = 0
    full_fills: int = 0
    fill_rate: float = 1.0
    partial_fill_rate: float = 0.0

    # Execution delay
    total_delay_ticks: int = 0
    delayed_trades: int = 0
    avg_delay_ticks: float = 0.0

    # PnL comparison
    baseline_pnl: float = 0.0
    realistic_pnl: float = 0.0
    pnl_degradation: float = 0.0
    pnl_degradation_pct: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_slippage_pips": self.total_slippage_pips,
            "slippage_trades": self.slippage_trades,
            "avg_slippage_pips": self.avg_slippage_pips,
            "max_slippage_pips": self.max_slippage_pips,
            "min_slippage_pips": self.min_slippage_pips,
            "total_spread_cost_pips": self.total_spread_cost_pips,
            "spread_trades": self.spread_trades,
            "avg_spread_pips": self.avg_spread_pips,
            "total_fills": self.total_fills,
            "partial_fills": self.partial_fills,
            "full_fills": self.full_fills,
            "fill_rate": self.fill_rate,
            "partial_fill_rate": self.partial_fill_rate,
            "total_delay_ticks": self.total_delay_ticks,
            "delayed_trades": self.delayed_trades,
            "avg_delay_ticks": self.avg_delay_ticks,
            "baseline_pnl": self.baseline_pnl,
            "realistic_pnl": self.realistic_pnl,
            "pnl_degradation": self.pnl_degradation,
            "pnl_degradation_pct": self.pnl_degradation_pct,
        }


# =============================================================================
# TICK DATA STRUCTURE
# =============================================================================

@dataclass
class TickData:
    """Single tick data point."""
    timestamp: datetime
    bid: float
    ask: float
    last: float  # Last traded price
    volume: float
    spread: float = 0.0  # ask - bid

    def __post_init__(self):
        if self.spread == 0.0:
            self.spread = self.ask - self.bid

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "volume": self.volume,
            "spread": self.spread,
            "mid": self.mid,
        }


class TickSeries:
    """Collection of tick data with helpers."""

    def __init__(self, ticks: List[TickData] | None = None):
        self._ticks: List[TickData] = ticks or []

    def append(self, tick: TickData) -> None:
        self._ticks.append(tick)

    def __len__(self) -> int:
        return len(self._ticks)

    def __getitem__(self, i: int) -> TickData:
        return self._ticks[i]

    @property
    def dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        if not self._ticks:
            return pd.DataFrame()

        data = [t.to_dict() for t in self._ticks]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df

    @property
    def bid_series(self) -> pd.Series:
        """Bid prices."""
        return pd.Series([t.bid for t in self._ticks], index=[t.timestamp for t in self._ticks])

    @property
    def ask_series(self) -> pd.Series:
        """Ask prices."""
        return pd.Series([t.ask for t in self._ticks], index=[t.timestamp for t in self._ticks])

    @property
    def spread_series(self) -> pd.Series:
        """Spread in price units."""
        return pd.Series([t.spread for t in self._ticks], index=[t.timestamp for t in self._ticks])

    @property
    def volume_series(self) -> pd.Series:
        """Volume."""
        return pd.Series([t.volume for t in self._ticks], index=[t.timestamp for t in self._ticks])

    def spread_pips(self, pair: str) -> pd.Series:
        """Spread in pips."""
        pip = 0.01 if pair.endswith("JPY") else 0.0001
        return self.spread_series / pip

    def is_continuous(self, max_gap: Optional[pd.Timedelta] = None) -> bool:
        """Check if ticks are continuous."""
        if len(self) < 2:
            return True

        max_gap = max_gap or pd.Timedelta(minutes=5)
        timestamps = pd.Series([t.timestamp for t in self._ticks])
        gaps = timestamps.diff().dropna()
        return gaps.max() <= max_gap

    @property
    def first_tick(self) -> Optional[TickData]:
        return self._ticks[0] if self._ticks else None

    @property
    def last_tick(self) -> Optional[TickData]:
        return self._ticks[-1] if self._ticks else None


# =============================================================================
# TICK ENGINE CORE
# =============================================================================

class TickExecutionEngine:
    """Tick-level execution engine."""

    def __init__(
        self,
        settings: TickExecutionSettings | None = None,
        ticks: Dict[str, TickSeries] | None = None,
    ):
        self.settings = settings or TickExecutionSettings()
        self.ticks: Dict[str, TickSeries] = ticks or {}
        self._rng = random.Random(self.settings.random_seed)

        # Active orders
        self._pending_orders: List[PendingOrder] = []
        self._active_trades: List[ActiveTrade] = []

        # Metrics
        self._metrics = ExecutionMetrics()

    @property
    def metrics(self) -> ExecutionMetrics:
        """Get execution metrics."""
        return self._metrics

    def load_ticks(self, symbol: str, df: pd.DataFrame) -> None:
        """Load tick data for a symbol.

        Args:
            symbol: Trading symbol
            df: DataFrame with columns [bid, ask, last, volume, timestamp]
        """
        if symbol not in self.ticks:
            self.ticks[symbol] = TickSeries()

        for _, row in df.iterrows():
            tick = TickData(
                timestamp=row.get("timestamp", row.name),
                bid=float(row.get("bid", 0)),
                ask=float(row.get("ask", 0)),
                last=float(row.get("last", 0)),
                volume=float(row.get("volume", 0)),
            )
            self.ticks[symbol].append(tick)

    def get_ticks(self, symbol: str) -> TickSeries:
        """Get ticks for a symbol."""
        return self.ticks.get(symbol, TickSeries())

    # -------------------------------------------------------------------------
    # ORDER EXECUTION
    # -------------------------------------------------------------------------

    def execute_market_order(
        self,
        symbol: str,
        direction: TradeDirection,
        size: float,
        signal_time: datetime,
        pair: str = "EURUSD",
    ) -> ExecutionResult:
        """Execute market order at best price from ticks.

        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            size: Position size
            signal_time: When signal was generated
            pair: Currency pair (for pip calculation)

        Returns:
            ExecutionResult with fill details
        """
        ticks = self.get_ticks(symbol)
        if len(ticks) == 0:
            return ExecutionResult(
                success=False,
                error="No tick data available",
                state=OrderState.REJECTED,
            )

        # Find execution tick (after signal time + latency)
        exec_tick = self._find_execution_tick(ticks, signal_time)
        if exec_tick is None:
            return ExecutionResult(
                success=False,
                error="No tick after signal time",
                state=OrderState.REJECTED,
            )

        # Get execution price
        if direction == TradeDirection.BUY:
            exec_price = exec_tick.ask  # Buy at ask
        else:
            exec_price = exec_tick.bid  # Sell at bid

        # Calculate spread cost
        spread_pips = self._calculate_spread_pips(exec_tick, pair)
        spread_cost = spread_pips * self._pip_size(pair)

        # Calculate slippage
        slippage_pips = self._calculate_slippage(
            direction=direction,
            entry_price=exec_price,
            ticks=ticks,
            pair=pair,
        )
        slippage_cost = slippage_pips * self._pip_size(pair)

        # Partial fill
        fill_ratio, filled_size = self._calculate_partial_fill(
            exec_tick=exec_tick,
            requested_size=size,
            direction=direction,
        )

        # Total cost
        total_cost = spread_cost + slippage_cost

        # Update metrics
        self._update_metrics(
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            fill_ratio=fill_ratio,
            delay_ticks=self._count_ticks_until(ticks, signal_time),
        )

        return ExecutionResult(
            success=True,
            state=OrderState.FILLED,
            symbol=symbol,
            direction=direction,
            requested_size=size,
            filled_size=filled_size,
            fill_ratio=fill_ratio,
            entry_price=exec_price,
            slippage_pips=slippage_pips,
            spread_pips=spread_pips,
            total_cost_pips=spread_pips + slippage_pips,
            execution_time=exec_tick.timestamp,
        )

    def execute_limit_order(
        self,
        symbol: str,
        direction: TradeDirection,
        size: float,
        limit_price: float,
        signal_time: datetime,
        pair: str = "EURUSD",
        tolerance_pips: float = 1.0,
    ) -> ExecutionResult:
        """Execute limit order when price touches level.

        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            size: Position size
            limit_price: Limit price level
            signal_time: When signal was generated
            pair: Currency pair
            tolerance_pips: Touch tolerance

        Returns:
            ExecutionResult
        """
        ticks = self.get_ticks(signal_time.symbol)
        if len(ticks) == 0:
            return ExecutionResult(
                success=False,
                error="No tick data for limit order",
                state=OrderState.REJECTED,
            )

        # Find tick where limit is touched
        exec_tick = self._find_limit_touch(
            ticks=ticks,
            limit_price=limit_price,
            direction=direction,
            tolerance_pips=tolerance_pips,
            pair=pair,
        )

        if exec_tick is None:
            return ExecutionResult(
                success=False,
                error="Limit not touched",
                state=OrderState.PENDING,
            )

        # Execute at touched price
        return self.execute_market_order(
            symbol=symbol,
            direction=direction,
            size=size,
            signal_time=exec_tick.timestamp,
            pair=pair,
        )

    def execute_stop_order(
        self,
        symbol: str,
        direction: TradeDirection,
        size: float,
        stop_price: float,
        signal_time: datetime,
        pair: str = "EURUSD",
    ) -> ExecutionResult:
        """Execute stop order when price crosses level.

        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            size: Position size
            stop_price: Stop trigger level
            signal_time: When signal was generated
            pair: Currency pair

        Returns:
            ExecutionResult
        """
        ticks = self.get_ticks(symbol)
        if len(ticks) == 0:
            return ExecutionResult(
                success=False,
                error="No tick data for stop order",
                state=OrderState.REJECTED,
            )

        # Find stop trigger
        exec_tick = self._find_stop_trigger(
            ticks=ticks,
            stop_price=stop_price,
            direction=direction,
        )

        if exec_tick is None:
            return ExecutionResult(
                success=False,
                error="Stop not triggered",
                state=OrderState.PENDING,
            )

        # Execute at market
        return self.execute_market_order(
            symbol=symbol,
            direction=direction,
            size=size,
            signal_time=exec_tick.timestamp,
            pair=pair,
        )

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _find_execution_tick(
        self,
        ticks: TickSeries,
        signal_time: datetime,
    ) -> Optional[TickData]:
        """Find first tick after signal time + latency."""
        latency = self.settings.execution_latency_ticks

        for i, tick in enumerate(ticks._ticks):
            if tick.timestamp >= signal_time:
                # Apply latency offset
                idx = i + latency
                if idx < len(ticks):
                    return ticks[idx]
                return ticks.last_tick  # Return last available

        return ticks.last_tick

    def _count_ticks_until(self, ticks: TickSeries, time: datetime) -> int:
        """Count ticks until a time."""
        count = 0
        for tick in ticks._ticks:
            if tick.timestamp < time:
                count += 1
            else:
                break
        return count

    def _calculate_spread_pips(self, tick: TickData, pair: str) -> float:
        """Calculate spread in pips."""
        spread_price = tick.spread
        if spread_price <= 0:
            return self.settings.fallback_spread_pips

        pip = self._pip_size(pair)
        if pip <= 0:
            return self.settings.fallback_spread_pips

        return max(0.0, spread_price / pip)

    def _calculate_slippage(
        self,
        direction: TradeDirection,
        entry_price: float,
        ticks: TickSeries,
        pair: str,
    ) -> float:
        """Calculate stochastic slippage."""
        if not self.settings.enable_realistic_slippage:
            return 0.0

        # Calculate volatility from recent ticks
        volatility = self._calculate_volatility(ticks)

        # Base slippage
        base = self.settings.base_slippage_pips

        # Random component scaled by volatility
        max_slip = self.settings.max_slippage_pips
        random_component = self._rng.uniform(0, max_slip) * volatility

        # Direction bias (small)
        direction_bias = 0.0
        if direction == TradeDirection.BUY:
            direction_bias = 0.1  # Slight upward bias for buys
        else:
            direction_bias = -0.1  # Slight downward bias for sells

        slippage = base + random_component + direction_bias
        slippage = max(0.0, min(max_slip, slippage))

        if self.settings.log_slippage_samples:
            logger.info(
                f"Slippage: {slippage:.3f} pips "
                f"(vol={volatility:.2f}, base={base}, max={max_slip})"
            )

        return slippage

    def _calculate_volatility(self, ticks: TickSeries) -> float:
        """Calculate volatility from tick variance."""
        if len(ticks) < 2:
            return 0.0

        # Use last 20 ticks for volatility
        recent = ticks._ticks[-min(20, len(ticks)):]
        mid_prices = [t.mid for t in recent]

        if len(mid_prices) < 2:
            return 0.0

        # Calculate variance
        returns = np.diff(mid_prices) / mid_prices[:-1]
        variance = float(np.var(returns)) if len(returns) > 0 else 0.0

        # Scale to 0-1 range
        volatility = min(1.0, variance * 1000)
        return volatility

    def _calculate_partial_fill(
        self,
        exec_tick: TickData,
        requested_size: float,
        direction: TradeDirection,
    ) -> tuple[float, float]:
        """Calculate partial fill ratio and size."""
        if not self.settings.enable_partial_fills:
            return 1.0, requested_size

        volume = exec_tick.volume

        # Check volume threshold
        if volume < self.settings.partial_fill_volume_threshold:
            return 1.0, requested_size

        # Random fill based on volume and settings
        min_ratio = self.settings.partial_fill_min_ratio
        max_ratio = self.settings.partial_fill_max_ratio

        # Determine fill ratio
        if self._rng.random() < 0.2:  # 20% chance of partial
            # Random between min and max
            fill_ratio = self._rng.uniform(min_ratio, max_ratio)
        else:
            fill_ratio = 1.0

        filled_size = requested_size * fill_ratio

        if self.settings.log_fill_details and fill_ratio < 1.0:
            logger.info(
                f"Partial fill: {fill_ratio:.0%} "
                f"(size={requested_size:.2f} -> {filled_size:.2f})"
            )

        return fill_ratio, filled_size

    def _find_limit_touch(
        self,
        ticks: TickSeries,
        limit_price: float,
        direction: TradeDirection,
        tolerance_pips: float,
        pair: str,
    ) -> Optional[TickData]:
        """Find tick where limit price is touched."""
        tolerance = tolerance_pips * self._pip_size(pair)

        for tick in ticks._ticks:
            if direction == TradeDirection.BUY:
                # For buy limit, price must go to or below limit
                if tick.ask <= limit_price + tolerance:
                    return tick
            else:
                # For sell limit, price must go to or above limit
                if tick.bid >= limit_price - tolerance:
                    return tick

        return None

    def _find_stop_trigger(
        self,
        ticks: TickSeries,
        stop_price: float,
        direction: TradeDirection,
    ) -> Optional[TickData]:
        """Find tick where stop is triggered."""
        for tick in ticks._ticks:
            if direction == TradeDirection.BUY:
                # Stop triggered when ask crosses above stop
                if tick.ask >= stop_price:
                    return tick
            else:
                # Stop triggered when bid crosses below stop
                if tick.bid <= stop_price:
                    return tick

        return None

    def _pip_size(self, pair: str) -> float:
        """Get pip size for pair."""
        pair = pair.upper().replace("/", "")
        if len(pair) != 6:
            return 0.0001
        quote = pair[3:6]
        return 0.01 if quote == "JPY" else 0.0001

    def _update_metrics(
        self,
        spread_pips: float,
        slippage_pips: float,
        fill_ratio: float,
        delay_ticks: int,
    ) -> None:
        """Update execution metrics."""
        m = self._metrics

        # Spread
        m.total_spread_cost_pips += spread_pips
        m.spread_trades += 1

        # Slippage
        if slippage_pips > 0:
            m.total_slippage_pips += slippage_pips
            m.slippage_trades += 1
            m.max_slippage_pips = max(m.max_slippage_pips, slippage_pips)
            m.min_slippage_pips = min(m.min_slippage_pips, slippage_pips)

        # Fills
        m.total_fills += 1
        if fill_ratio < 1.0:
            m.partial_fills += 1
        else:
            m.full_fills += 1

        # Delay
        if delay_ticks > 0:
            m.total_delay_ticks += delay_ticks
            m.delayed_trades += 1

        # Calculate averages
        if m.slippage_trades > 0:
            m.avg_slippage_pips = m.total_slippage_pips / m.slippage_trades

        if m.spread_trades > 0:
            m.avg_spread_pips = m.total_spread_cost_pips / m.spread_trades

        if m.total_fills > 0:
            m.fill_rate = m.full_fills / m.total_fills
            m.partial_fill_rate = m.partial_fills / m.total_fills

        if m.delayed_trades > 0:
            m.avg_delay_ticks = m.total_delay_ticks / m.delayed_trades

    def calculate_realistic_pnl(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        direction: TradeDirection,
        pair: str = "EURUSD",
    ) -> float:
        """Calculate realistic PnL with all costs."""
        pip = self._pip_size(pair)

        # Direction multiplier
        mult = 1.0 if direction == TradeDirection.BUY else -1.0

        # Baseline PnL
        baseline = (exit_price - entry_price) * size * mult

        # Add costs from metrics
        costs = (
            self._metrics.avg_slippage_pips * pip * size +
            self._metrics.avg_spread_pips * pip * size
        )

        return baseline - costs


@dataclass
class PendingOrder:
    """Pending order in the engine."""
    order_id: str
    order_type: OrderType
    symbol: str
    direction: TradeDirection
    size: float
    price: float  # Limit/stop price
    created_time: datetime
    state: OrderState = OrderState.PENDING


@dataclass
class ActiveTrade:
    """Active trade being tracked."""
    trade_id: str
    symbol: str
    direction: TradeDirection
    size: float
    filled_size: float
    entry_price: float
    entry_time: datetime
    state: OrderState = OrderState.FILLED


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    state: OrderState = OrderState.FILLED
    error: str = ""

    # Trade details
    symbol: str = ""
    direction: TradeDirection = TradeDirection.BUY
    requested_size: float = 0.0
    filled_size: float = 0.0
    fill_ratio: float = 1.0

    # Pricing
    entry_price: float = 0.0
    stop_price: float = 0.0
    exit_price: float = 0.0

    # Costs
    slippage_pips: float = 0.0
    spread_pips: float = 0.0
    total_cost_pips: float = 0.0

    # Timing
    signal_time: Optional[datetime] = None
    execution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "state": self.state.name,
            "error": self.error,
            "symbol": self.symbol,
            "direction": self.direction.name,
            "requested_size": self.requested_size,
            "filled_size": self.filled_size,
            "fill_ratio": self.fill_ratio,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "slippage_pips": self.slippage_pips,
            "spread_pips": self.spread_pips,
            "total_cost_pips": self.total_cost_pips,
        }


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

class ExecutionComparisonRunner:
    """Run baseline vs tick execution for comparison."""

    def __init__(
        self,
        tick_settings: TickExecutionSettings | None = None,
        baseline_settings: Optional[Any] = None,
    ):
        self.tick_settings = tick_settings or TickExecutionSettings()
        self.baseline_settings = baseline_settings

    def run_comparison(
        self,
        symbol: str,
        trades: List[Dict[str, Any]],
        tick_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Run baseline and tick execution comparison.

        Args:
            symbol: Trading symbol
            trades: List of trades with [direction, size, entry_price, exit_price, time]
            tick_data: Tick data DataFrame

        Returns:
            Comparison metrics
        """
        # Load tick data
        engine = TickExecutionEngine(settings=self.tick_settings)
        engine.load_ticks(symbol, tick_data)

        baseline_pnl = 0.0
        tick_pnl = 0.0

        pip = 0.0001  # Default pip

        for trade in trades:
            direction = (
                TradeDirection.BUY if trade.get("direction") == "BUY"
                else TradeDirection.SELL
            )
            size = trade.get("size", 1.0)
            entry = trade.get("entry_price", 0)
            exit = trade.get("exit_price", 0)

            # Baseline PnL (no costs)
            mult = 1.0 if direction == TradeDirection.BUY else -1.0
            baseline = (exit - entry) * size * mult
            baseline_pnl += baseline

            # Tick execution PnL (with costs)
            result = engine.execute_market_order(
                symbol=symbol,
                direction=direction,
                size=size,
                signal_time=trade.get("time", datetime.now()),
            )

            if result.success:
                # Calculate realistic PnL
                tick_pnl += engine.calculate_realistic_pnl(
                    entry_price=result.entry_price,
                    exit_price=exit,
                    size=result.filled_size,
                    direction=direction,
                )

        # Calculate degradation
        degradation = baseline_pnl - tick_pnl
        degradation_pct = (
            (degradation / abs(baseline_pnl) * 100)
            if baseline_pnl != 0
            else 0.0
        )

        return {
            "baseline_pnl": baseline_pnl,
            "tick_pnl": tick_pnl,
            "degradation": degradation,
            "degradation_pct": degradation_pct,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_tick_engine(
    enable_tick_execution: bool = False,
    enable_realistic_slippage: bool = False,
    enable_partial_fills: bool = False,
    max_slippage_pips: float = 1.0,
    execution_latency_ticks: int = 0,
    random_seed: int | None = None,
    tick_data_source: str = "mt5",
) -> TickExecutionSettings:
    """Create tick execution settings."""
    return TickExecutionSettings(
        enable_tick_execution=enable_tick_execution,
        enable_realistic_slippage=enable_realistic_slippage,
        enable_partial_fills=enable_partial_fills,
        max_slippage_pips=max_slippage_pips,
        execution_latency_ticks=execution_latency_ticks,
        random_seed=random_seed,
        tick_data_source=tick_data_source,
    ).sanitized()