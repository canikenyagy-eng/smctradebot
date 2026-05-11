"""
Execution Quality Adjustment Layer

Applies regime-based execution quality to backtest simulation:
- Slippage model adjustment
- Spread model adjustment
- Fill probability adjustment
- Execution delay simulation
- Effective execution price adjustment

This ONLY applies in BACKTEST mode - no live effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.regime_engine_v2 import RegimeOutput, RegimeType


# Execution quality per regime (0-1, higher = better)
REGIME_EXECUTION_QUALITY: dict[str, float] = {
    RegimeType.TREND_STRONG.value: 1.0,
    RegimeType.TREND_WEAK.value: 0.9,
    RegimeType.RANGE_TIGHT.value: 0.85,
    RegimeType.RANGE_WIDE.value: 0.6,
    RegimeType.EXPANSION.value: 0.7,
    RegimeType.TRANSITION.value: 0.4,
}

# Base spread per pair (pips)
PAIR_BASE_SPREAD: dict[str, float] = {
    "EURUSD": 0.1,
    "GBPUSD": 0.2,
    "USDJPY": 0.2,
    "USDCHF": 0.3,
    "AUDUSD": 0.3,
    "USDCAD": 0.3,
    "NZDUSD": 0.3,
    "EURGBP": 0.3,
    "EURJPY": 0.4,
    "GBPJPY": 0.4,
}

# Low liquidity hours (UTC)
LOW_LIQUIDITY_HOURS = {0, 1, 2, 3, 4, 5}


@dataclass(frozen=True)
class ExecutionQualitySettings:
    """Settings for execution quality model."""
    enabled: bool = False
    
    # Base slippage in pips
    base_slippage_pips: float = 0.5
    
    # Max slippage adjustment
    max_slippage_multiplier: float = 2.0
    
    # Fill probability base
    base_fill_probability: float = 0.95
    
    # Apply to limit orders specifically
    apply_to_limits: bool = True
    
    # Apply to market orders
    apply_to_market: bool = True
    
    # Execution delay (bars)
    execution_delay_bars: int = 0
    
    # Spread adjustment multiplier
    spread_multiplier: float = 1.5
    
    def sanitized(self) -> "ExecutionQualitySettings":
        return ExecutionQualitySettings(
            enabled=self.enabled,
            base_slippage_pips=max(0.0, float(self.base_slippage_pips)),
            max_slippage_multiplier=max(1.0, float(self.max_slippage_multiplier)),
            base_fill_probability=max(0.0, min(1.0, float(self.base_fill_probability))),
            apply_to_limits=self.apply_to_limits,
            apply_to_market=self.apply_to_market,
            execution_delay_bars=max(0, int(self.execution_delay_bars)),
            spread_multiplier=max(1.0, float(self.spread_multiplier)),
        )


@dataclass
class ExecutionQualityResult:
    """Result of execution quality calculation."""
    adjusted_slippage: float
    fill_probability: float
    execution_price_adjustment: float
    
    # Additional details
    base_slippage: float
    quality_multiplier: float
    regime: str
    
    # New fields
    spread_pips: float
    session_liquidity_factor: float
    execution_delay_bars: int


class ExecutionQualityLayer:
    """Regime-based execution quality model."""
    
    def __init__(
        self,
        settings: ExecutionQualitySettings | None = None,
    ):
        self.settings = (settings or ExecutionQualitySettings()).sanitized()
    
    def compute_quality(
        self,
        regime: str,
        order_type: str = "market",  # "market" or "limit"
        pair: str = "EURUSD",
    ) -> ExecutionQualityResult:
        """
        Compute execution quality for regime.
        
        Only applies in backtest mode.
        """
        if not self.settings.enabled:
            return ExecutionQualityResult(
                adjusted_slippage=self.settings.base_slippage_pips,
                fill_probability=self.settings.base_fill_probability,
                execution_price_adjustment=0.0,
                base_slippage=self.settings.base_slippage_pips,
                quality_multiplier=1.0,
                regime=regime,
                spread_pips=PAIR_BASE_SPREAD.get(pair, 0.2),
                session_liquidity_factor=1.0,
                execution_delay_bars=0,
            )
        
        # Get regime quality
        quality = REGIME_EXECUTION_QUALITY.get(regime, 0.5)
        
        # Adjust if order type should be skipped
        if order_type == "limit" and not self.settings.apply_to_limits:
            quality = 1.0
        if order_type == "market" and not self.settings.apply_to_market:
            quality = 1.0
        
        quality_multiplier = quality
        
        # Adjust slippage (lower quality = higher slippage)
        base_slippage = self.settings.base_slippage_pips
        adjusted_slippage = base_slippage * (2.0 - quality)
        adjusted_slippage = min(
            adjusted_slippage,
            base_slippage * self.settings.max_slippage_multiplier,
        )
        
        # Adjust fill probability
        base_fill = self.settings.base_fill_probability
        fill_probability = base_fill * quality
        
        # Price adjustment (pips against trader)
        execution_price_adjustment = adjusted_slippage * (1.0 - quality)
        
        # NEW: Spread calculation
        base_spread = PAIR_BASE_SPREAD.get(pair, 0.2)
        spread_pips = base_spread * self.settings.spread_multiplier * (2.0 - quality)
        
        # NEW: Session liquidity factor
        utc_hour = datetime.utcnow().hour
        session_liquidity_factor = 0.6 if utc_hour in LOW_LIQUIDITY_HOURS else 1.0
        
        # Execution delay
        delay_bars = self.settings.execution_delay_bars
        
        return ExecutionQualityResult(
            adjusted_slippage=round(adjusted_slippage, 4),
            fill_probability=round(fill_probability, 4),
            execution_price_adjustment=round(execution_price_adjustment, 4),
            base_slippage=base_slippage,
            quality_multiplier=round(quality_multiplier, 4),
            regime=regime,
            spread_pips=round(spread_pips, 4),
            session_liquidity_factor=round(session_liquidity_factor, 4),
            execution_delay_bars=delay_bars,
        )
    
    def get_slippage_pips(
        self,
        regime: str,
        order_type: str = "market",
    ) -> float:
        """Get adjusted slippage in pips."""
        return self.compute_quality(regime, order_type).adjusted_slippage
    
    def get_fill_probability(
        self,
        regime: str,
        order_type: str = "market",
    ) -> float:
        """Get fill probability."""
        return self.compute_quality(regime, order_type).fill_probability
    
    def get_settings(self) -> dict[str, Any]:
        """Get settings."""
        return {
            "enabled": self.settings.enabled,
            "base_slippage_pips": self.settings.base_slippage_pips,
            "max_slippage_multiplier": self.settings.max_slippage_multiplier,
            "base_fill_probability": self.settings.base_fill_probability,
            "regime_quality": REGIME_EXECUTION_QUALITY,
        }