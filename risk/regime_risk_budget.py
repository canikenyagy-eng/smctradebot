"""
Regime Risk Budget Engine.

This module calculates dynamic position sizing and risk allocation
based on regime quality, execution degradation, and expectancy.

NO strategy modification - only risk layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RiskBudgetConfig:
    """Configuration for risk budget."""

    # Base risk per trade (in R)
    base_risk: float = 1.0

    # Regimes that are always blocked
    blocked_regimes: tuple = ("contraction",)

    # Weights per regime
    regime_weights: tuple = (
        ("trend", 1.0),
        ("expansion", 0.9),
        ("range", 0.5),
        ("contraction", 0.0),
    )

    # Expectancy bounds
    min_expectancy_factor: float = 0.2
    max_expectancy_factor: float = 1.2

    # Execution factor bounds
    min_execution_factor: float = 0.3
    max_execution_factor: float = 1.0


# =============================================================================
# RISK DECISION OUTPUT
# =============================================================================

@dataclass
class RiskDecision:
    """Decision output for risk allocation."""

    regime: str
    base_risk: float
    adjusted_risk: float
    regime_weight: float
    expectancy_factor: float
    execution_factor: float
    final_allocation: float

    # Additional context
    blocked: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "base_risk": self.base_risk,
            "adjusted_risk": self.adjusted_risk,
            "regime_weight": self.regime_weight,
            "expectancy_factor": self.expectancy_factor,
            "execution_factor": self.execution_factor,
            "final_allocation": self.final_allocation,
            "blocked": self.blocked,
            "rejection_reason": self.rejection_reason,
        }


# =============================================================================
# RISK BUDGET ENGINE
# =============================================================================

class RegimeRiskBudget:
    """Calculate regime-based risk allocation."""

    # Default weights
    DEFAULT_WEIGHTS = {
        "trend": 1.0,
        "expansion": 0.9,
        "range": 0.5,
        "contraction": 0.0,
    }

    def __init__(self, config: RiskBudgetConfig | None = None):
        self.config = config or RiskBudgetConfig()
        self._weights = dict(self.config.regime_weights)

    def calculate(
        self,
        regime: str,
        base_risk: float | None = None,
        baseline_E: float = 1.0,
        realistic_E: float = 1.0,
        execution_cost_pips: float = 2.0,
    ) -> RiskDecision:
        """Calculate risk allocation.

        Args:
            regime: Current regime
            base_risk: Base risk in R (default from config)
            baseline_E: Baseline expectancy in R
            realistic_E: Realistic expectancy in R
            execution_cost_pips: Avg execution cost in pips

        Returns:
            RiskDecision with allocation
        """
        base_risk = base_risk or self.config.base_risk
        regime = regime.lower() if regime else "neutral"

        # Step 1: Get regime weight
        regime_weight = self._weights.get(regime, 1.0)

        # Step 2: Check if blocked
        if regime in self.config.blocked_regimes:
            return RiskDecision(
                regime=regime,
                base_risk=base_risk,
                adjusted_risk=0.0,
                regime_weight=regime_weight,
                expectancy_factor=1.0,
                execution_factor=1.0,
                final_allocation=0.0,
                blocked=True,
                rejection_reason="CONTRACTION regime blocked",
            )

        # Step 3: Calculate expectancy factor
        expectancy_factor = self._calculate_expectancy_factor(
            baseline_E, realistic_E
        )

        # Step 4: Calculate execution factor
        execution_factor = self._calculate_execution_factor(
            execution_cost_pips
        )

        # Step 5: Calculate final allocation
        adjusted_risk = (
            base_risk *
            regime_weight *
            expectancy_factor *
            execution_factor
        )

        # Ensure non-negative
        adjusted_risk = max(0.0, adjusted_risk)

        return RiskDecision(
            regime=regime,
            base_risk=base_risk,
            adjusted_risk=adjusted_risk,
            regime_weight=regime_weight,
            expectancy_factor=expectancy_factor,
            execution_factor=execution_factor,
            final_allocation=adjusted_risk,
        )

    def _calculate_expectancy_factor(
        self,
        baseline_E: float,
        realistic_E: float,
    ) -> float:
        """Calculate expectancy factor.

        Factor = realistic / baseline
        Clamped to bounds
        """
        if baseline_E == 0:
            return 1.0

        factor = realistic_E / baseline_E

        # Clamp to bounds
        return max(
            self.config.min_expectancy_factor,
            min(self.config.max_expectancy_factor, factor),
        )

    def _calculate_execution_factor(
        self,
        cost_pips: float,
    ) -> float:
        """Calculate execution factor from cost.

        factor = 1 - normalized_cost
        where normalized_cost maps 5+ pips to 1.0 and 0 pips to 0.0
        """
        # Normalize cost (0 pips = 0, 5+ pips = 1)
        normalized = min(1.0, cost_pips / 5.0)

        factor = 1.0 - normalized

        # Clamp to bounds
        return max(
            self.config.min_execution_factor,
            min(self.config.max_execution_factor, factor),
        )

    def get_regime_weight(self, regime: str) -> float:
        """Get weight for regime."""
        return self._weights.get(regime.lower(), 1.0)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_risk_budget(
    base_risk: float = 1.0,
    block_contraction: bool = True,
) -> RegimeRiskBudget:
    """Create risk budget engine."""
    blocked = ("contraction",) if block_contraction else ()

    config = RiskBudgetConfig(
        base_risk=base_risk,
        blocked_regimes=blocked,
    )

    return RegimeRiskBudget(config)