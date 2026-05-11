"""
Prop Risk Engine v2 - Institutional Risk Management Layer

This module computes dynamic risk per trade using:
- Base risk
- Regime multiplier
- Equity multiplier  
- Correlation multiplier

IMPORTANT: This is a risk layer only - no strategy logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskMode(Enum):
    """Risk calculation mode."""
    BASELINE = "baseline"
    PROP = "prop"
    INSTITUTIONAL = "institutional"


@dataclass(frozen=True)
class PropRiskSettings:
    """Settings forProp Risk Engine v2."""
    enabled: bool = False
    base_risk: float = 1.0
    max_risk_per_trade: float = 2.0
    
    # Drawdown thresholds (in R multiples)
    dd_threshold_low: float = 3.0
    dd_threshold_mid: float = 6.0
    dd_threshold_high: float = 10.0
    
    # Risk reduction factors
    dd_reduction_low: float = 1.0
    dd_reduction_mid: float = 0.5
    dd_reduction_high: float = 0.25
    dd_reduction_stop: float = 0.0
    
    # Consecutive loss reductions
    loss_2_reduction: float = 0.8
    loss_3_reduction: float = 0.6
    loss_4_pause: bool = True
    
    def sanitized(self) -> "PropRiskSettings":
        return PropRiskSettings(
            enabled=self.enabled,
            base_risk=max(0.1, float(self.base_risk)),
            max_risk_per_trade=max(0.1, float(self.max_risk_per_trade)),
            dd_threshold_low=max(0.0, float(self.dd_threshold_low)),
            dd_threshold_mid=max(0.0, float(self.dd_threshold_mid)),
            dd_threshold_high=max(0.0, float(self.dd_threshold_high)),
            dd_reduction_low=clamp(float(self.dd_reduction_low), 0.0, 1.0),
            dd_reduction_mid=clamp(float(self.dd_reduction_mid), 0.0, 1.0),
            dd_reduction_high=clamp(float(self.dd_reduction_high), 0.0, 1.0),
            dd_reduction_stop=clamp(float(self.dd_reduction_stop), 0.0, 1.0),
            loss_2_reduction=clamp(float(self.loss_2_reduction), 0.0, 1.0),
            loss_3_reduction=clamp(float(self.loss_3_reduction), 0.0, 1.0),
            loss_4_pause=self.loss_4_pause,
        )


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


@dataclass
class PropRiskState:
    """Runtime risk state."""
    settings: PropRiskSettings
    equity_r: float = 0.0
    peak_equity_r: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    paused: bool = False
    
    # Risk multipliers (accumulative)
    regime_multiplier: float = 1.0
    equity_multiplier: float = 1.0
    correlation_multiplier: float = 1.0
    
    def drawdown_r(self) -> float:
        """Current drawdown in R."""
        return max(0.0, self.peak_equity_r - self.equity_r)
    
    def base_risk_allowed(self) -> float:
        """Calculate base risk allowed."""
        if not self.settings.enabled:
            return self.settings.base_risk
        
        # Start with base risk
        risk = self.settings.base_risk
        
        # Apply all multipliers
        risk *= self.regime_multiplier
        risk *= self.equity_multiplier
        risk *= self.correlation_multiplier
        
        # Cap at max
        return min(risk, self.settings.max_risk_per_trade)
    
    def compute_final_risk(
        self,
        regime_multiplier: float,
        correlation_factor: float = 1.0,
    ) -> float:
        """
        Compute final risk for trade.
        
        Formula: Risk = BaseRisk × Regime × Equity × Correlation
        """
        if not self.settings.enabled:
            return self.settings.base_risk
        
        # Update regime multiplier
        self.regime_multiplier = max(0.0, min(1.0, regime_multiplier))
        
        # Update correlation
        self.correlation_multiplier = max(0.0, min(1.0, correlation_factor))
        
        # Apply equity multiplier
        self._update_equity_multiplier()
        
        # Final risk
        return self.base_risk_allowed()
    
    def _update_equity_multiplier(self) -> None:
        """Update equity multiplier based on drawdown."""
        if not self.settings.enabled:
            self.equity_multiplier = 1.0
            return
        
        dd = self.drawdown_r()
        settings = self.settings
        
        # DD thresholds
        if dd >= settings.dd_threshold_high:
            self.equity_multiplier = settings.dd_reduction_stop
            self.paused = True
        elif dd >= settings.dd_threshold_mid:
            self.equity_multiplier = settings.dd_reduction_high
        elif dd >= settings.dd_threshold_low:
            self.equity_multiplier = settings.dd_reduction_mid
        else:
            self.equity_multiplier = settings.dd_reduction_low
    
    def apply_consecutive_loss(self) -> None:
        """Apply consecutive loss penalty."""
        if not self.settings.enabled:
            return
        
        self.consecutive_losses += 1
        self.consecutive_wins = 0
        
        # Apply reduction
        if self.consecutive_losses >= 4 and self.settings.loss_4_pause:
            self.paused = True
            self.equity_multiplier = 0.0
        elif self.consecutive_losses == 3:
            self.equity_multiplier *= self.settings.loss_3_reduction
        elif self.consecutive_losses == 2:
            self.equity_multiplier *= self.settings.loss_2_reduction
    
    def apply_win(self) -> None:
        """Record win."""
        if not self.settings.enabled:
            return
        
        self.consecutive_wins += 1
        self.consecutive_losses = 0
    
    def can_trade(self) -> bool:
        """Check if trading allowed."""
        if not self.settings.enabled:
            return True
        if self.paused:
            return False
        if self.drawdown_r() >= self.settings.dd_threshold_high:
            return False
        return True
    
    def register_trade(self, r_pnl: float) -> None:
        """Register trade result."""
        if r_pnl > 0:
            self.apply_win()
        elif r_pnl < 0:
            self.apply_consecutive_loss()
        
        self.equity_r += r_pnl
        if self.equity_r > self.peak_equity_r:
            self.peak_equity_r = self.equity_r
    
    def get_risk_breakdown(self) -> dict[str, Any]:
        """Get risk calculation breakdown."""
        return {
            "base_risk": self.settings.base_risk,
            "regime_multiplier": self.regime_multiplier,
            "equity_multiplier": self.equity_multiplier,
            "correlation_multiplier": self.correlation_multiplier,
            "final_risk": self.base_risk_allowed(),
            "drawdown_r": self.drawdown_r(),
            "can_trade": self.can_trade(),
            "consecutive_losses": self.consecutive_losses,
            "paused": self.paused,
        }
    
    def reset(self) -> None:
        """Reset state."""
        self.equity_r = 0.0
        self.peak_equity_r = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.paused = False
        self.regime_multiplier = 1.0
        self.equity_multiplier = 1.0
        self.correlation_multiplier = 1.0


class PropRiskEngine:
    """Main risk engine for Prop trading."""
    
    def __init__(self, settings: PropRiskSettings | None = None):
        self.settings = (settings or PropRiskSettings()).sanitized()
        self.state = PropRiskState(settings=self.settings)
    
    def compute_risk(
        self,
        regime: str,  # From regime_engine_v2
        correlation_exposure: float = 1.0,
    ) -> float:
        """Compute final risk for trade."""
        from core.regime_engine_v2 import get_regime_multiplier
        
        regime_mult = get_regime_multiplier(regime)
        return self.state.compute_final_risk(regime_mult, correlation_exposure)
    
    def is_trade_allowed(self) -> bool:
        """Check if trade allowed."""
        return self.state.can_trade()
    
    def register_result(self, r_pnl: float) -> None:
        """Register trade result."""
        self.state.register_trade(r_pnl)
    
    def get_status(self) -> dict[str, Any]:
        """Get engine status."""
        return {
            "enabled": self.settings.enabled,
            "can_trade": self.is_trade_allowed(),
            "risk_breakdown": self.state.get_risk_breakdown(),
        }
    
    def get_risk_decision(self) -> dict[str, Any]:
        """
        Get structured risk decision for trade gate.
        
        Output format:
        {
            "risk_allowed": bool,
            "risk_size": float,
            "reason": "...",
            "multiplier_breakdown": {...}
        }
        """
        if not self.settings.enabled:
            return {
                "risk_allowed": True,
                "risk_size": self.settings.base_risk,
                "reason": "disabled",
                "multiplier_breakdown": {"mode": "disabled"},
            }
        
        if not self.state.can_trade():
            dd = self.state.drawdown_r()
            if dd >= self.settings.dd_threshold_high:
                reason = f"high_drawdown_{dd:.1f}R"
            elif self.state.paused:
                reason = "paused_loss_streak"
            else:
                reason = "drawdown_limit"
            return {
                "risk_allowed": False,
                "risk_size": 0.0,
                "reason": reason,
                "multiplier_breakdown": self.state.get_risk_breakdown(),
            }
        
        return {
            "risk_allowed": True,
            "risk_size": self.state.base_risk_allowed(),
            "reason": "",
            "multiplier_breakdown": self.state.get_risk_breakdown(),
        }
    
    def reset(self) -> None:
        """Reset state."""
        self.state.reset()