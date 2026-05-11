"""
Trade Gate v2 - Centralized Risk Gate

A trade is allowed ONLY if ALL conditions pass:
- Regime tradability score > threshold
- Risk engine allows trade
- Portfolio exposure within limits
- Equity state allows trading
- No transition regime active
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.regime_engine_v2 import RegimeOutput, classify_regime, is_trade_allowed
from core.prop_risk_engine_v2 import PropRiskEngine
from core.portfolio_risk_v2 import PortfolioRiskV2


@dataclass(frozen=True)
class TradeGateSettings:
    """Settings for trade gate."""
    enabled: bool = False
    
    # Minimum tradability threshold
    min_regime_tradability: int = 30
    
    # Require risk engine
    check_risk_engine: bool = True
    
    # Require portfolio check
    check_portfolio: bool = True
    
    # Block during transition
    block_transition: bool = True
    
    def sanitized(self) -> "TradeGateSettings":
        return TradeGateSettings(
            enabled=self.enabled,
            min_regime_tradability=max(0, min(100, int(self.min_regime_tradability))),
            check_risk_engine=self.check_risk_engine,
            check_portfolio=self.check_portfolio,
            block_transition=self.block_transition,
        )


@dataclass(frozen=True)
class TradeGateResult:
    """Result from trade gate check."""
    allowed: bool
    reason: str
    details: dict[str, Any]


class TradeGateV2:
    """Centralized trade permission gate."""
    
    def __init__(
        self,
        settings: TradeGateSettings | None = None,
        risk_engine: PropRiskEngine | None = None,
        portfolio: PortfolioRiskV2 | None = None,
    ):
        self.settings = (settings or TradeGateSettings()).sanitized()
        self.risk_engine = risk_engine
        self.portfolio = portfolio
    
    def check_trade(
        self,
        pair: str,
        side: str,
        regime_output: RegimeOutput | None = None,
        universe: set[str] | None = None,
    ) -> TradeGateResult:
        """
        Check if trade should be allowed.
        
        Returns TradeGateResult with allowed=True/False and reason.
        """
        if not self.settings.enabled:
            return TradeGateResult(
                allowed=True,
                reason="",
                details={"mode": "disabled"},
            )
        
        # Check 1: Regime tradability
        if regime_output is None:
            # Must have regime data
            regime_output = classify_regime(pd.DataFrame())  # Empty - will return transition
        
        # Block transition regime
        if self.settings.block_transition and regime_output.is_transition:
            return TradeGateResult(
                allowed=False,
                reason="transition_regime",
                details={"regime": regime_output.regime},
            )
        
        # Check minimum tradability
        if regime_output.tradability_score < self.settings.min_regime_tradability:
            return TradeGateResult(
                allowed=False,
                reason="low_tradability",
                details={
                    "score": regime_output.tradability_score,
                    "threshold": self.settings.min_regime_tradability,
                },
            )
        
        # Check 2: Risk engine
        if self.settings.check_risk_engine and self.risk_engine:
            if not self.risk_engine.is_trade_allowed():
                return TradeGateResult(
                    allowed=False,
                    reason="risk_engine_blocked",
                    details=self.risk_engine.get_status(),
                )
        
        # Check 3: Portfolio exposure
        if self.settings.check_portfolio and self.portfolio:
            allowed, reason = self.portfolio.check_trade(pair, side, universe or set())
            if not allowed:
                return TradeGateResult(
                    allowed=False,
                    reason=f"portfolio_{reason}",
                    details=self.portfolio.get_exposure_summary(),
                )
        
        # All checks passed
        return TradeGateResult(
            allowed=True,
            reason="",
            details={
                "regime": regime_output.regime,
                "tradability": regime_output.tradability_score,
                "confidence": regime_output.confidence,
            },
        )
    
    def can_trade(self) -> bool:
        """Quick check if any trading allowed."""
        if not self.settings.enabled:
            return True
        
        if self.risk_engine:
            return self.risk_engine.is_trade_allowed()
        
        return True
    
    def get_status(self) -> dict[str, Any]:
        """Get gate status."""
        return {
            "enabled": self.settings.enabled,
            "can_trade": self.can_trade(),
        }


# Import for empty DataFrame
import pandas as pd