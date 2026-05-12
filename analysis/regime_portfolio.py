"""
Regime Portfolio Allocation View.

This module provides regime-based capital allocation recommendations
based on expectancy and execution costs.

NO strategy modification - analytical only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class RegimeAllocation:
    """Allocation data for one regime."""

    regime: str
    allocation_pct: float = 0.0
    risk_contribution_pct: float = 0.0
    pnl_contribution_pct: float = 0.0
    execution_cost_pct: float = 0.0

    # Expected values
    expected_pnl: float = 0.0
    expected_drawdown: float = 0.0
    execution_drag: float = 0.0


class RegimePortfolioView:
    """Calculate regime-based allocation."""

    def __init__(
        self,
        min_allocation: float = 0.05,
        max_allocation: float = 0.50,
    ):
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.allocations: Dict[str, RegimeAllocation] = {}

    def compute_allocation(
        self,
        regime_metrics: Dict[str, dict],
        regime_weights: Dict[str, float] | None = None,
    ) -> Dict[str, RegimeAllocation]:
        """Compute allocation per regime.

        Args:
            regime_metrics: Dict of {regime: {expectancy, cost, std, trade_count}}
            regime_weights: Optional override weights

        Returns:
            Dict of {regime: RegimeAllocation}
        """
        regime_weights = regime_weights or {}

        if not regime_metrics:
            return {}

        # Step 1: Calculate raw allocation score
        scores = {}
        total_raw = 0.0

        for regime, metrics in regime_metrics.items():
            expectancy = metrics.get("expectancy", 0)
            cost = metrics.get("cost", 2.0)
            std = metrics.get("std", 1.0)

            # Score = expectancy / cost (higher is better)
            # Adjusted by stability (lower std is better)
            stability = 1.0 / max(std, 0.5)
            score = (expectancy * stability) / max(cost, 0.1)

            scores[regime] = max(0, score)
            total_raw += scores[regime]

        # Step 2: Normalize to allocations
        if total_raw > 0:
            for regime, score in scores.items():
                raw_pct = score / total_raw
                # Apply bounds
                pct = max(
                    self.min_allocation,
                    min(self.max_allocation, raw_pct)
                )
                regime_weights.get(regime, 1.0)

                alloc = RegimeAllocation(
                    regime=regime,
                    allocation_pct=pct,
                )
                self.allocations[regime] = alloc

        # Step 3: Apply custom weights if provided
        if regime_weights:
            for regime, weight in regime_weights.items():
                if regime in self.allocations:
                    self.allocations[regime].allocation_pct *= weight

        # Step 4: Renormalize to 100%
        total = sum(
            a.allocation_pct for a in self.allocations.values()
        )
        if total > 0 and total != 1.0:
            for alloc in self.allocations.values():
                alloc.allocation_pct /= total

        return self.allocations

    def compute_expected_portfolio(
        self,
        regime_metrics: Dict[str, dict],
        allocation: Dict[str, RegimeAllocation] | None = None,
    ) -> Dict[str, float]:
        """Compute expected portfolio metrics.

        Args:
            regime_metrics: Dict of {regime: {expectancy, drawdown, cost}}
            allocation: Optional allocation overrides

        Returns:
            Dict with expected values
        """
        allocation = allocation or self.allocations

        expected_pnl = 0.0
        expected_dd = 0.0
        execution_drag = 0.0

        for regime, metrics in regime_metrics.items():
            alloc = allocation.get(regime)
            if not alloc:
                continue

            pct = alloc.allocation_pct
            exp = metrics.get("expectancy", 0)
            dd = metrics.get("drawdown", 0)
            cost = metrics.get("cost", 0)

            expected_pnl += exp * pct
            expected_dd += dd * pct
            execution_drag += cost * pct

        return {
            "expected_pnl": round(expected_pnl, 2),
            "expected_drawdown": round(expected_dd, 2),
            "execution_drag": round(execution_drag, 2),
        }

    def get_allocation_table(self) -> pd.DataFrame:
        """Get allocation table."""
        if not self.allocations:
            return pd.DataFrame()

        rows = []
        for regime, alloc in sorted(self.allocations.items()):
            rows.append({
                "regime": regime.upper(),
                "allocation": f"{alloc.allocation_pct:.0%}",
                "expected_pnl": f"{alloc.expected_pnl:.2f}R",
                "expected_dd": f"{alloc.expected_drawdown:.1f}%",
                "exec_drag": f"{alloc.execution_drag:.1f} pips",
            })

        return pd.DataFrame(rows)


def compute_portfolio_view(
    regime_metrics: Dict[str, dict],
    regime_weights: Dict[str, float] | None = None,
) -> RegimePortfolioView:
    """Compute portfolio view from regime metrics.

    Args:
        regime_metrics: Dict like:
            {"trend": {"expectancy": 5.0, "cost": 2.0, "std": 1.0, "drawdown": 5.0}}
        regime_weights: Optional weights

    Returns:
        RegimePortfolioView
    """
    view = RegimePortfolioView()
    view.compute_allocation(regime_metrics, regime_weights)
    return view