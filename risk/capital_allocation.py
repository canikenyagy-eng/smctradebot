"""
Capital Allocation Optimizer.

This module dynamically distributes capital across regimes based on
execution-adjusted expectancy, stability, and drawdown contribution.

NO strategy modification - pure portfolio layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AllocationConfig:
    """Configuration for capital allocation."""

    total_capital: float = 1.0
    rebalance_every_n_trades: int = 50
    rebalance_window_days: int = 30
    min_regime_allocation: float = 0.05
    max_regime_allocation: float = 0.50
    allow_contraction: bool = False
    use_decay: bool = True
    decay_factor: float = 0.8


# =============================================================================
# OUTPUTS
# =============================================================================

@dataclass
class RegimeCapitalAllocation:
    """Capital allocation for one regime."""

    regime: str
    allocation_pct: float
    expected_return: float = 0.0
    expected_drawdown: float = 0.0
    execution_cost: float = 0.0
    stability: float = 0.0
    trade_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "allocation": f"{self.allocation_pct:.0%}",
            "expected_return": f"{self.expected_return:.2f}R",
            "expected_drawdown": f"{self.expected_drawdown:.1f}%",
            "execution_cost": f"{self.execution_cost:.1f} pips",
            "stability": f"{self.stability:.0%}",
            "trades": self.trade_count,
        }


@dataclass
class PortfolioAllocation:
    """Full portfolio allocation."""

    allocations: Dict[str, RegimeCapitalAllocation] = field(default_factory=dict)
    total_capital: float = 1.0

    expected_return: float = 0.0
    expected_drawdown: float = 0.0
    execution_drag: float = 0.0
    herfindahl_index: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocations": {
                r: a.to_dict() for r, a in self.allocations.items()
            },
            "expected_return": self.expected_return,
            "expected_drawdown": self.expected_drawdown,
            "execution_drag": self.execution_drag,
            "herfindahl_index": self.herfindahl_index,
        }


# =============================================================================
# METRICS INPUT
# =============================================================================

@dataclass
class RegimeMetrics:
    """Input metrics for one regime."""

    regime: str
    trade_count: int = 0
    baseline_E: float = 0.0
    realistic_E: float = 0.0
    std_R: float = 1.0
    avg_cost_pips: float = 2.0
    max_drawdown: float = 5.0
    avg_drawdown: float = 2.0


# =============================================================================
# CAPITAL ALLOCATION OPTIMIZER
# =============================================================================

class CapitalAllocator:
    """Optimize capital allocation across regimes."""

    def __init__(self, config: AllocationConfig | None = None):
        self.config = config or AllocationConfig()
        self._current: Dict[str, RegimeCapitalAllocation] = {}
        self._history: List[PortfolioAllocation] = []

    def compute_allocation(
        self,
        metrics: Dict[str, RegimeMetrics],
    ) -> PortfolioAllocation:
        """Compute optimal allocation."""
        eps = 0.001

        # Step 1: Calculate raw scores
        scores = {}
        total_score = 0.0

        for regime, m in metrics.items():
            stability = 1.0 / max(m.std_R, eps)
            score = (m.realistic_E * stability) / max(m.avg_cost_pips, eps)
            scores[regime] = max(0.0, score)
            total_score += scores[regime]

        # Step 2: Normalize
        if total_score > 0:
            raw_weights = {r: s / total_score for r, s in scores.items()}
        else:
            n = len(metrics) if metrics else 1
            raw_weights = {r: 1.0 / n for r in metrics}

        # Step 3: Apply bounds
        allocations = {}
        for regime, weight in raw_weights.items():
            bounded = max(
                self.config.min_regime_allocation,
                min(self.config.max_regime_allocation, weight),
            )
            if regime == "contraction" and not self.config.allow_contraction:
                bounded = 0.0
            allocations[regime] = bounded

        # Step 4: Renormalize
        total_weight = sum(allocations.values())
        if total_weight > 0 and total_weight != 1.0:
            for regime in allocations:
                allocations[regime] /= total_weight

        # Step 5: Build output
        result = PortfolioAllocation(total_capital=self.config.total_capital)
        expected_return = 0.0
        expected_dd = 0.0
        exec_drag = 0.0
        weights_squared = []

        for regime, pct in allocations.items():
            m = metrics.get(regime)
            if not m:
                continue

            alloc = RegimeCapitalAllocation(
                regime=regime,
                allocation_pct=pct,
                expected_return=m.realistic_E * pct,
                expected_drawdown=m.avg_drawdown * pct,
                execution_cost=m.avg_cost_pips,
                stability=1.0 / max(m.std_R, 0.1),
                trade_count=m.trade_count,
            )

            result.allocations[regime] = alloc
            expected_return += alloc.expected_return
            expected_dd += alloc.expected_drawdown
            exec_drag += alloc.execution_cost * pct
            weights_squared.append(pct ** 2)

        result.expected_return = expected_return
        result.expected_drawdown = expected_dd
        result.execution_drag = exec_drag
        result.herfindahl_index = sum(weights_squared) if weights_squared else 0.0

        self._current = result.allocations
        return result


# =============================================================================
# DECAY WEIGHTED ALLOCATOR
# =============================================================================

class DecayWeightedAllocator(CapitalAllocator):
    """Allocator with decay for recent performance."""

    def __init__(self, config: AllocationConfig | None = None):
        super().__init__(config)
        self._trade_history: List[Dict[str, float]] = []

    def add_trade_result(self, regime: str, r_value: float) -> None:
        """Add trade result for decay calculation."""
        self._trade_history.append({regime: r_value})
        max_history = max(self.config.rebalance_every_n_trades * 2, 200)
        if len(self._trade_history) > max_history:
            self._trade_history = self._trade_history[-max_history:]

    def compute_with_decay(
        self,
        metrics: Dict[str, RegimeMetrics],
    ) -> PortfolioAllocation:
        """Compute with decay weighting."""
        if not self.config.use_decay:
            return self.compute_allocation(metrics)

        if len(self._trade_history) < 20:
            return self.compute_allocation(metrics)

        recent_window = min(50, len(self._trade_history) // 2)

        recent_r = {r: [] for r in metrics}
        hist_r = {r: [] for r in metrics}

        for i, result in enumerate(self._trade_history):
            for regime, r in result.items():
                if regime not in metrics:
                    continue
                if i >= len(self._trade_history) - recent_window:
                    recent_r[regime].append(r)
                else:
                    hist_r[regime].append(r)

        decay_factors = {}
        for regime in metrics:
            avg_recent = np.mean(recent_r[regime]) if recent_r[regime] else 0.0
            avg_hist = np.mean(hist_r[regime]) if hist_r[regime] else 0.0
            ratio = avg_recent / avg_hist if avg_hist != 0 else (1.0 if avg_recent > 0 else 0.5)
            decay_factors[regime] = self.config.decay_factor + (1 - self.config.decay_factor) * ratio

        adjusted = {}
        for regime, m in metrics.items():
            adjusted[regime] = RegimeMetrics(
                regime=regime,
                trade_count=m.trade_count,
                baseline_E=m.baseline_E,
                realistic_E=m.realistic_E * decay_factors.get(regime, 1.0),
                std_R=m.std_R,
                avg_cost_pips=m.avg_cost_pips,
                max_drawdown=m.max_drawdown,
                avg_drawdown=m.avg_drawdown,
            )

        return self.compute_allocation(adjusted)


# =============================================================================
# FACTORY
# =============================================================================

def create_allocator(
    total_capital: float = 1.0,
    allow_contraction: bool = False,
    use_decay: bool = True,
) -> DecayWeightedAllocator:
    """Create capital allocator."""
    config = AllocationConfig(
        total_capital=total_capital,
        allow_contraction=allow_contraction,
        use_decay=use_decay,
    )
    return DecayWeightedAllocator(config)