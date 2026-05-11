"""
Expectancy Engine v3 - Mathematical Expectancy Layer

Calculates expectancy per trade, regime, and signal type:
- E = (WinRate × AvgWin) - (LossRate × AvgLoss)

Features:
- Regime-based expectancy
- Volatility-adjusted expectancy
- Signal-type expectancy
- Risk-adjusted expectancy

IMPORTANT: This is READ-ONLY - does not modify SMC logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Default expectancy values per regime (from backtest analysis)
# These should be tuned based on actual backtest results
REGIME_EXPECTANCY: dict[str, dict[str, float]] = {
    "trend_strong": {
        "win_rate": 0.65,
        "avg_win": 1.5,  # R units
        "avg_loss": 0.8,  # R units
    },
    "trend_weak": {
        "win_rate": 0.55,
        "avg_win": 1.2,
        "avg_loss": 0.7,
    },
    "range_tight": {
        "win_rate": 0.60,
        "avg_win": 1.0,
        "avg_loss": 0.6,
    },
    "range_wide": {
        "win_rate": 0.45,
        "avg_win": 1.3,
        "avg_loss": 1.0,
    },
    "expansion": {
        "win_rate": 0.40,
        "avg_win": 2.0,
        "avg_loss": 1.2,
    },
    "transition": {
        "win_rate": 0.30,
        "avg_win": 1.0,
        "avg_loss": 1.0,
    },
}

# Default expectancy per signal type
SIGNAL_EXPECTANCY: dict[str, dict[str, float]] = {
    "liquidity_sweep": {
        "win_rate": 0.55,
        "avg_win": 1.4,
        "avg_loss": 0.9,
    },
    "fvg": {
        "win_rate": 0.50,
        "avg_win": 1.2,
        "avg_loss": 0.8,
    },
    "order_block": {
        "win_rate": 0.58,
        "avg_win": 1.3,
        "avg_loss": 0.75,
    },
    "breakout": {
        "win_rate": 0.52,
        "avg_win": 1.6,
        "avg_loss": 1.0,
    },
}


def compute_raw_expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Compute raw expectancy: E = (WR × AW) - (LR × AL)
    
    Where LR = 1 - WR
    """
    loss_rate = 1.0 - win_rate
    return (win_rate * avg_win) - (loss_rate * avg_loss)


def get_regime_expectancy(regime: str) -> dict[str, float]:
    """Get expectancy parameters for regime."""
    return REGIME_EXPECTANCY.get(
        regime, 
        REGIME_EXPECTANCY.get("transition", {})
    )


def get_signal_expectancy(signal_type: str) -> dict[str, float]:
    """Get expectancy parameters for signal type."""
    return SIGNAL_EXPECTANCY.get(
        signal_type, 
        {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0}
    )


@dataclass(frozen=True)
class ExpectancyResult:
    """Result of expectancy calculation."""
    expectancy: float
    risk_adjusted_expectancy: float
    regime: str
    valid_trade: bool
    confidence: float
    
    # Breakdown
    win_rate: float
    avg_win: float
    avg_loss: float
    signal_type: str
    volatility_factor: float


@dataclass
class ExpectancySettings:
    """Settings for expectancy engine."""
    enabled: bool = False
    
    # Regime expectancies (can be overridden)
    regime_params: dict[str, dict[str, float]] | None = None
    
    # Signal expectancies
    signal_params: dict[str, dict[str, float]] | None = None
    
    # Volatility scaling
    vol_scaling_enabled: bool = True
    max_vol_scaling: float = 1.5
    min_vol_scaling: float = 0.5
    
    # Drawdown sensitivity
    dd_threshold: float = 6.0  # R units
    dd_sensitivity: float = 0.5
    
    # Minimum expectancy threshold
    min_expectancy: float = 0.05
    
    # Valid trade threshold
    min_confidence: float = 0.3
    
    def sanitized(self) -> "ExpectancySettings":
        """Return sanitized settings."""
        return ExpectancySettings(
            enabled=self.enabled,
            regime_params=self.regime_params,
            signal_params=self.signal_params,
            vol_scaling_enabled=self.vol_scaling_enabled,
            max_vol_scaling=max(0.5, min(2.0, self.max_vol_scaling)),
            min_vol_scaling=max(0.1, min(1.0, self.min_vol_scaling)),
            dd_threshold=max(1.0, float(self.dd_threshold)),
            dd_sensitivity=max(0.0, min(1.0, self.dd_sensitivity)),
            min_expectancy=max(0.0, float(self.min_expectancy)),
            min_confidence=max(0.0, min(1.0, self.min_confidence)),
        )


class ExpectancyEngine:
    """Expectancy calculation engine."""
    
    def __init__(self, settings: ExpectancySettings | None = None):
        self.settings = (settings or ExpectancySettings()).sanitized()
        self._regime_params = self.settings.regime_params or REGIME_EXPECTANCY
        self._signal_params = self.settings.signal_params or SIGNAL_EXPECTANCY
    
    def compute_expectancy(
        self,
        regime: str,
        signal_type: str,
        current_volatility: float = 0.5,  # 0-1 normalized
        drawdown_r: float = 0.0,  # current drawdown in R
        score: int = 70,  # signal score
    ) -> ExpectancyResult:
        """
        Compute expectancy for trade.
        
        Args:
            regime: Current market regime
            signal_type: Signal trigger type (liquidity_sweep, fvg, order_block, breakout)
            current_volatility: Normalized volatility (0-1)
            drawdown_r: Current drawdown in R
            score: Signal score
            
        Returns:
            ExpectancyResult with expectancy values
        """
        if not self.settings.enabled:
            # Return neutral when disabled
            return ExpectancyResult(
                expectancy=0.0,
                risk_adjusted_expectancy=0.0,
                regime=regime,
                valid_trade=True,
                confidence=0.5,
                win_rate=0.5,
                avg_win=1.0,
                avg_loss=1.0,
                signal_type=signal_type,
                volatility_factor=1.0,
            )
        
        # Get base parameters
        regime_params = self._regime_params.get(
            regime, 
            self._regime_params.get("transition", {})
        )
        signal_params = self._signal_params.get(
            signal_type,
            {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0},
        )
        
        # Blend regime and signal parameters (50/50)
        win_rate = 0.5 * regime_params.get("win_rate", 0.5) + 0.5 * signal_params.get("win_rate", 0.5)
        avg_win = 0.5 * regime_params.get("avg_win", 1.0) + 0.5 * signal_params.get("avg_win", 1.0)
        avg_loss = 0.5 * regime_params.get("avg_loss", 1.0) + 0.5 * signal_params.get("avg_loss", 1.0)
        
        # Compute raw expectancy
        raw_expectancy = compute_raw_expectancy(win_rate, avg_win, avg_loss)
        
        # Compute volatility factor
        vol_factor = self._compute_volatility_factor(current_volatility)
        
        # Compute risk-adjusted expectancy
        risk_adj_expectancy = self._apply_risk_adjustment(
            raw_expectancy, 
            drawdown_r,
            vol_factor,
        )
        
        # Compute confidence
        confidence = self._compute_confidence(
            score, 
            regime,
            win_rate,
        )
        
        # Valid trade check
        valid_trade = (
            risk_adj_expectancy >= self.settings.min_expectancy and
            confidence >= self.settings.min_confidence and
            regime != "transition"
        )
        
        return ExpectancyResult(
            expectancy=round(raw_expectancy, 4),
            risk_adjusted_expectancy=round(risk_adj_expectancy, 4),
            regime=regime,
            valid_trade=valid_trade,
            confidence=round(confidence, 4),
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            signal_type=signal_type,
            volatility_factor=round(vol_factor, 4),
        )
    
    def _compute_volatility_factor(
        self, 
        volatility: float,
    ) -> float:
        """
        Compute volatility scaling factor.
        
        High volatility: reduce expectancy (harder to trade)
        Low volatility: normalize expectancy
        """
        if not self.settings.vol_scaling_enabled:
            return 1.0
        
        # Volatility 0.5 = 1.0 factor
        # Volatility > 0.5 = reduce
        # Volatility < 0.5 = increase slightly
        if volatility > 0.5:
            excess = volatility - 0.5
            factor = 1.0 - (excess * (self.settings.max_vol_scaling - 1.0))
        else:
            deficit = 0.5 - volatility
            factor = 1.0 + (deficit * (1.0 - self.settings.min_vol_scaling))
        
        return max(
            self.settings.min_vol_scaling,
            min(self.settings.max_vol_scaling, factor),
        )
    
    def _apply_risk_adjustment(
        self,
        expectancy: float,
        drawdown_r: float,
        vol_factor: float,
    ) -> float:
        """
        Apply risk adjustments to expectancy.
        
        - Drawdown sensitivity: reduce when in drawdown
        - Volatility factor already applied
        """
        # Base adjustment from drawdown
        if drawdown_r >= self.settings.dd_threshold:
            # Severe drawdown - reduce to near zero
            return expectancy * 0.1
        elif drawdown_r > 0:
            # Gradual reduction
            dd_factor = 1.0 - (
                (drawdown_r / self.settings.dd_threshold) 
                * self.settings.dd_sensitivity
            )
            return expectancy * vol_factor * dd_factor
        
        return expectancy * vol_factor
    
    def _compute_confidence(
        self,
        score: int,
        regime: str,
        win_rate: float,
    ) -> float:
        """
        Compute trade confidence (0-1).
        
        Based on:
        - Score threshold (min 70)
        - Regime (transition = 0)
        - Win rate
        """
        # Invalid regimes
        if regime == "transition":
            return 0.0
        
        # Score contribution (0-1)
        score_factor = max(0.0, min(1.0, (score - 50) / 50.0))
        
        # Regime contribution
        regime_factor = 1.0 if regime != "transition" else 0.0
        
        # Win rate contribution
        wr_factor = win_rate
        
        # Combined confidence
        confidence = 0.4 * score_factor + 0.3 * regime_factor + 0.3 * wr_factor
        
        return confidence
    
    def is_trade_allowed(
        self,
        regime: str,
        signal_type: str,
        current_volatility: float = 0.5,
        drawdown_r: float = 0.0,
        score: int = 70,
    ) -> bool:
        """Check if trade passes expectancy filter."""
        if not self.settings.enabled:
            return True
        
        result = self.compute_expectancy(
            regime=regime,
            signal_type=signal_type,
            current_volatility=current_volatility,
            drawdown_r=drawdown_r,
            score=score,
        )
        
        return result.valid_trade
    
    def get_settings(self) -> dict[str, Any]:
        """Get engine settings."""
        return {
            "enabled": self.settings.enabled,
            "min_expectancy": self.settings.min_expectancy,
            "min_confidence": self.settings.min_confidence,
            "vol_scaling_enabled": self.settings.vol_scaling_enabled,
            "dd_threshold": self.settings.dd_threshold,
        }


# Helper to get global expectancy
def quick_expectancy(
    regime: str,
    signal_type: str,
    volatility: float = 0.5,
    score: int = 70,
) -> ExpectancyResult:
    """Quick expectancy calculation with default settings."""
    engine = ExpectancyEngine(ExpectancySettings(enabled=True))
    return engine.compute_expectancy(
        regime=regime,
        signal_type=signal_type,
        current_volatility=volatility,
        drawdown_r=0.0,
        score=score,
    )