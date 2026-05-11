"""
Trade Gate v2 - Centralized Risk Gate

A trade is allowed ONLY if ALL conditions pass:
- Regime tradability score > threshold
- Risk engine allows trade
- Portfolio exposure within limits
- Equity state allows trading
- No transition regime active
- Session liquidity check

This gate runs BEFORE trade execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


# Low liquidity sessions (UTC hour) - no trading during these
LOW_LIQUIDITY_SESSIONS = set([0, 1, 2, 3, 4, 5])


@dataclass(frozen=True)
class TradeGateSettings:
    """Settings for trade gate."""
    enabled: bool = False
    
    # Minimum tradability threshold
    min_regime_tradability: int = 30
    
    # Minimum score threshold (new - institutional requirement)
    min_score_threshold: float = 0.0  # Default 0.0 = allow all scores
    
    # Require score check
    check_score_threshold: bool = True
    
    # Require risk engine
    check_risk_engine: bool = True
    
    # Require portfolio check
    check_portfolio: bool = True
    
    # Block during transition
    block_transition: bool = True
    
    # Block low liquidity sessions
    check_session: bool = True
    
    # Block during expansion regime
    block_expansion: bool = False
    
    def sanitized(self) -> "TradeGateSettings":
        return TradeGateSettings(
            enabled=self.enabled,
            min_regime_tradability=max(0, min(100, int(self.min_regime_tradability))),
            min_score_threshold=max(0.0, min(100.0, float(self.min_score_threshold))),
            check_score_threshold=self.check_score_threshold,
            check_risk_engine=self.check_risk_engine,
            check_portfolio=self.check_portfolio,
            block_transition=self.block_transition,
            check_session=self.check_session,
            block_expansion=self.block_expansion,
        )


@dataclass(frozen=True)
class TradeGateResult:
    """Result from trade gate check."""
    allowed: bool
    reason: str
    
    # Full state snapshots
    risk_state: dict[str, Any]
    regime_state: dict[str, Any]
    portfolio_state: dict[str, Any]
    session_state: dict[str, Any]
    
    @property
    def details(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "risk_state": self.risk_state,
            "regime_state": self.regime_state,
            "portfolio_state": self.portfolio_state,
            "session_state": self.session_state,
        }


class TradeGateV2:
    """Centralized trade permission gate."""
    
    def __init__(
        self,
        settings: TradeGateSettings | None = None,
        risk_engine: "PropRiskEngine" | None = None,
        portfolio: "PortfolioRiskV2" | None = None,
    ):
        self.settings = (settings or TradeGateSettings()).sanitized()
        self.risk_engine = risk_engine
        self.portfolio = portfolio
    
    def _check_session(self) -> tuple[bool, str]:
        """Check if current session is good for trading."""
        if not self.settings.check_session:
            return True, ""
        
        # Get current UTC hour
        utc_hour = datetime.utcnow().hour
        
        if utc_hour in LOW_LIQUIDITY_SESSIONS:
            return False, f"low_liquidity_session_{utc_hour}"
        
        return True, ""
    
    def _get_states(self, regime_output) -> tuple[dict, dict, dict]:
        """Get current state snapshots."""
        # Regime state
        regime_state = {
            "regime": regime_output.regime,
            "tradability": regime_output.tradability_score,
            "confidence": regime_output.confidence,
            "volatility": regime_output.volatility_estimate,
            "liquidity": regime_output.liquidity_quality,
        }
        
        # Risk state
        if self.risk_engine:
            risk_state = self.risk_engine.state.get_risk_breakdown()
        else:
            risk_state = {"enabled": False}
        
        # Portfolio state
        if self.portfolio:
            portfolio_state = self.portfolio.get_exposure_summary()
        else:
            portfolio_state = {"enabled": False}
        
        # Session state
        allow_session, session_reason = self._check_session()
        session_state = {
            "utc_hour": datetime.utcnow().hour,
            "allowed": allow_session,
            "reason": session_reason,
        }
        
        return regime_state, risk_state, portfolio_state, session_state
    
    def check_trade(
        self,
        pair: str,
        side: str,
        regime_output: "RegimeOutput" | None = None,
        universe: set[str] | None = None,
        current_score: float = 0.0,
    ) -> TradeGateResult:
        """
        Check if trade should be allowed.
        
        Returns TradeGateResult with allowed=True/False and reason.
        All conditions must pass for trade to be allowed.
        
        Args:
            pair: Trading pair
            side: BUY or SELL
            regime_output: RegimeOutput from regime engine
            universe: Set of trading pairs
            current_score: Current signal score (0-100)
        """
        if not self.settings.enabled:
            # Build empty states when disabled
            dummy_regime = regime_output
            dummy_state = {"mode": "disabled"}
            return TradeGateResult(
                allowed=True,
                reason="",
                risk_state=dummy_state,
                regime_state=dummy_state,
                portfolio_state=dummy_state,
                session_state=dummy_state,
            )
        
        # Check 1: Regime - get states first for return
        if regime_output is None:
            from core.regime_engine_v2 import classify_regime
            regime_output = classify_regime(pd.DataFrame())
        
        regime_state, risk_state, portfolio_state, session_state = self._get_states(regime_output)
        
        # Block transition regime
        if self.settings.block_transition and regime_output.is_transition:
            return TradeGateResult(
                allowed=False,
                reason="transition_regime",
                risk_state=risk_state,
                regime_state=regime_state,
                portfolio_state=portfolio_state,
                session_state=session_state,
            )
        
        # Block expansion regime if configured
        if self.settings.block_expansion and regime_output.regime == "expansion":
            return TradeGateResult(
                allowed=False,
                reason="expansion_regime",
                risk_state=risk_state,
                regime_state=regime_state,
                portfolio_state=portfolio_state,
                session_state=session_state,
            )
        
        # Check minimum tradability
        if regime_output.tradability_score < self.settings.min_regime_tradability:
            return TradeGateResult(
                allowed=False,
                reason="low_tradability",
                risk_state=risk_state,
                regime_state=regime_state,
                portfolio_state=portfolio_state,
                session_state=session_state,
            )
        
        # === CHECK: Score threshold (institutional requirement) ===
        if self.settings.check_score_threshold:
            if current_score < self.settings.min_score_threshold:
                return TradeGateResult(
                    allowed=False,
                    reason="score_below_threshold",
                    risk_state=risk_state,
                    regime_state=regime_state,
                    portfolio_state=portfolio_state,
                    session_state=session_state,
                )
        
        # Check 2: Risk engine
        if self.settings.check_risk_engine and self.risk_engine:
            if not self.risk_engine.is_trade_allowed():
                return TradeGateResult(
                    allowed=False,
                    reason="risk_engine_blocked",
                    risk_state=risk_state,
                    regime_state=regime_state,
                    portfolio_state=portfolio_state,
                    session_state=session_state,
                )
        
        # Check 3: Portfolio exposure
        if self.settings.check_portfolio and self.portfolio:
            allowed, reason = self.portfolio.check_trade(pair, side, universe or set())
            if not allowed:
                return TradeGateResult(
                    allowed=False,
                    reason=f"portfolio_{reason}",
                    risk_state=risk_state,
                    regime_state=regime_state,
                    portfolio_state=portfolio_state,
                    session_state=session_state,
                )
        
        # Check 4: Session liquidity
        allow_session, session_reason = self._check_session()
        if not allow_session:
            return TradeGateResult(
                allowed=False,
                reason=session_reason,
                risk_state=risk_state,
                regime_state=regime_state,
                portfolio_state=portfolio_state,
                session_state=session_state,
            )
        
        # All checks passed - ALLOW trade
        return TradeGateResult(
            allowed=True,
            reason="",
            risk_state=risk_state,
            regime_state=regime_state,
            portfolio_state=portfolio_state,
            session_state=session_state,
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