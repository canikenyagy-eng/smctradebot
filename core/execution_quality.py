"""
Execution Quality Adjustment Layer

Applies regime-based execution quality to backtest simulation:
- Slippage model adjustment
- Fill probability adjustment
- Effective execution price adjustment

This ONLY applies in BACKTEST mode - no live effect.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    
    def sanitized(self) -> "ExecutionQualitySettings":
        return ExecutionQualitySettings(
            enabled=self.enabled,
            base_slippage_pips=max(0.0, float(self.base_slippage_pips)),
            max_slippage_multiplier=max(1.0, float(self.max_slippage_multiplier)),
            base_fill_probability=max(0.0, min(1.0, float(self.base_fill_probability))),
            apply_to_limits=self.apply_to_limits,
            apply_to_market=self.apply_to_market,
        )


@dataclass
class ExecutionQualityResult:
    """Result of execution quality calculation."""
    adjusted_slippage: float
    fill_probability: float
    execution_price_adjustment: float
    
    # Details
    base_slippage: float
    quality_multiplier: float
    regime: str


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
        execution_price_adjustment = adjusted_slippage * (
            1.0 - quality
        )
        
        return ExecutionQualityResult(
            adjusted_slippage=round(adjusted_slippage, 4),
            fill_probability=round(fill_probability, 4),
            execution_price_adjustment=round(execution_price_adjustment, 4),
            base_slippage=base_slippage,
            quality_multiplier=round(quality_multiplier, 4),
            regime=regime,
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