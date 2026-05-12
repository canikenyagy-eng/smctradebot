"""
Final Edge Certification System.

This module evaluates whether the trading system produces a survivable,
execution-adjusted edge suitable for prop-firm or institutional deployment.

NO strategy modification - pure evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class CertificationConfig:
    """Configuration for certification."""

    # Monte Carlo settings
    mc_simulations: int = 2000
    mc_trades_per_sim: int = 500

    # Survival thresholds
    max_acceptable_dd_pct: float = 15.0  # Maximum 15% drawdown
    min_expectancy: float = 0.5  # Minimum 0.5R expectancy

    # Score weights
    expectancy_weight: float = 0.40
    stability_weight: float = 0.25
    regime_weight: float = 0.20
    drawdown_weight: float = 0.15


# =============================================================================
# RESULTS
# =============================================================================

@dataclass
class EdgeMetrics:
    """Edge metrics summary."""

    # Expectancy
    baseline_E: float = 0.0
    realistic_E: float = 0.0
    execution_adjusted_E: float = 0.0
    regime_weighted_E: float = 0.0

    # Stability
    std_R: float = 0.0
    cv: float = 0.0  # Coefficient of variation

    # Trade stats
    total_trades: int = 0
    win_rate: float = 0.0


@dataclass
class RegimeSurvival:
    """Regime survival analysis."""

    regime: str
    trade_count: int = 0

    baseline_E: float = 0.0
    realistic_E: float = 0.0
    execution_cost: float = 0.0

    survives_execution: bool = False
    is_blocked: bool = False


@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results."""

    # Distribution metrics
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0

    # Risk metrics
    prob_loss: float = 0.0
    prob_loss_10pct: float = 0.0
    VaR_95: float = 0.0  # 95% Value at Risk

    # Drawdown
    max_dd_mean: float = 0.0
    max_dd_95: float = 0.0
    max_dd_99: float = 0.0

    # Stability
    equity_curve_stability: float = 0.0


@dataclass
class CertificationScore:
    """Final certification score."""

    # Component scores (0-100)
    expectancy_score: float = 0.0
    stability_score: float = 0.0
    regime_survival_score: float = 0.0
    drawdown_resilience_score: float = 0.0

    # Final score
    total_score: float = 0.0

    # Status
    status: str = "UNTRADABLE"

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# CERTIFICATION ENGINE
# =============================================================================

class EdgeCertification:
    """Certification engine."""

    def __init__(self, config: CertificationConfig | None = None):
        self.config = config or CertificationConfig()

    def certify(
        self,
        trades: List[dict],
        regime_trades: Dict[str, List[dict]],
    ) -> CertificationScore:
        """Run full certification.

        Args:
            trades: List of {regime, base_r, exec_r, cost_pips}
            regime_trades: Dict of {regime: [trades]}

        Returns:
            CertificationScore
        """
        # Step 1: Compute edge metrics
        metrics = self._compute_edge_metrics(trades)

        # Step 2: Regime survival analysis
        survival = self._analyze_regime_survival(regime_trades)

        # Step 3: Monte Carlo
        mc = self._run_monte_carlo(trades)

        # Step 4: Compute final score
        score = self._compute_certification_score(
            metrics, survival, mc
        )

        return score

    def _compute_edge_metrics(
        self,
        trades: List[dict],
    ) -> EdgeMetrics:
        """Compute edge metrics."""
        if not trades:
            return EdgeMetrics()

        base_r = [t.get("base_r", 0) for t in trades]
        exec_r = [t.get("exec_r", 0) for t in trades]

        wins = [r for r in exec_r if r > 0]
        win_rate = len(wins) / len(trades) if trades else 0

        return EdgeMetrics(
            baseline_E=np.mean(base_r),
            realistic_E=np.mean(exec_r),
            execution_adjusted_E=np.mean(exec_r),
            std_R=np.std(exec_r),
            total_trades=len(trades),
            win_rate=win_rate,
        )

    def _analyze_regime_survival(
        self,
        regime_trades: Dict[str, List[dict]],
    ) -> Dict[str, RegimeSurvival]:
        """Analyze regime survival."""
        results = {}

        for regime, trades_list in regime_trades.items():
            if not trades_list:
                continue

            base_r = [t.get("base_r", 0) for t in trades_list]
            exec_r = [t.get("exec_r", 0) for t in trades_list]
            costs = [t.get("cost_pips", 2) for t in trades_list]

            baseline_E = np.mean(base_r)
            realistic_E = np.mean(exec_r)
            avg_cost = np.mean(costs)

            survives = realistic_E > 0

            results[regime] = RegimeSurvival(
                regime=regime,
                trade_count=len(trades_list),
                baseline_E=baseline_E,
                realistic_E=realistic_E,
                execution_cost=avg_cost,
                survives_execution=survives,
            )

        return results

    def _run_monte_carlo(
        self,
        trades: List[dict],
    ) -> MonteCarloResults:
        """Run Monte Carlo simulation."""
        config = self.config

        if not trades:
            return MonteCarloResults()

        # Extract trade returns
        exec_r = [t.get("exec_r", 0) for t in trades]
        mu = np.mean(exec_r)
        sigma = np.std(exec_r)

        # Simulate
        returns = []
        max_dds = []

        for _ in range(config.mc_simulations):
            # Sample trades with replacement
            sim_trades = np.random.choice(
                exec_r,
                size=config.mc_trades_per_sim,
                replace=True,
            )

            # Cumulative return
            cum_return = np.sum(sim_trades)
            returns.append(cum_return)

            # Max drawdown (equity curve)
            equity = np.cumsum(sim_trades)
            running_max = np.maximum.accumulate(equity)
            dd = running_max - equity
            max_dd = np.max(dd)
            max_dds.append(max_dd)

        returns = np.array(returns)
        max_dds = np.array(max_dds)

        prob_loss = np.mean(returns < 0)
        prob_loss_10 = np.mean(returns < -10)

        return MonteCarloResults(
            mean_return=np.mean(returns),
            median_return=np.median(returns),
            std_return=np.std(returns),
            prob_loss=prob_loss,
            prob_loss_10pct=prob_loss_10,
            VaR_95=np.percentile(returns, 5),
            max_dd_mean=np.mean(max_dds),
            max_dd_95=np.percentile(max_dds, 95),
            max_dd_99=np.percentile(max_dds, 99),
            equity_curve_stability=np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else 1.0,
        )

    def _compute_certification_score(
        self,
        metrics: EdgeMetrics,
        survival: Dict[str, RegimeSurvival],
        mc: MonteCarloResults,
    ) -> CertificationScore:
        """Compute final certification score."""
        cfg = self.config

        # 1. Expectancy score (0-100)
        if metrics.execution_adjusted_E >= cfg.min_expectancy:
            exp_score = min(100, metrics.execution_adjusted_E * 50)
        else:
            exp_score = max(0, (metrics.execution_adjusted_E / cfg.min_expectancy) * 50)

        # 2. Stability score (0-100) - lower CV = better
        cv = abs(metrics.std_R / metrics.realistic_E) if metrics.realistic_E != 0 else 1.0
        stability_score = max(0, 100 * (1 - cv))

        # 3. Regime survival score (0-100)
        survival_count = sum(1 for s in survival.values() if s.survives_execution)
        total_regimes = len(survival)
        regime_score = (survival_count / total_regimes * 100) if total_regimes > 0 else 0

        # 4. Drawdown resilience (0-100)
        dd_thresh = cfg.max_acceptable_dd_pct
        if mc.max_dd_mean < dd_thresh:
            dd_score = 100 * (1 - mc.max_dd_mean / dd_thresh)
        else:
            dd_score = 0

        # Total score
        total = (
            exp_score * cfg.expectancy_weight +
            stability_score * cfg.stability_weight +
            regime_score * cfg.regime_weight +
            dd_score * cfg.drawdown_weight
        )

        # Status determination
        if total < 30:
            status = "UNTRADABLE"
        elif total < 50:
            status = "BORDERLINE"
        elif total < 75:
            status = "PROP-FIRM READY"
        else:
            status = "INSTITUTIONAL READY"

        # Recommendations
        recs = []
        if exp_score < 40:
            recs.append("Improve execution-adjusted expectancy")
        if regime_score < 50:
            recs.append("Filter poorly-performing regimes")
        if dd_score < 50:
            recs.append("Reduce position sizing for better drawdown control")
        if stability_score < 50:
            recs.append("Increase sample size for stability")

        return CertificationScore(
            expectancy_score=exp_score,
            stability_score=stability_score,
            regime_survival_score=regime_score,
            drawdown_resilience_score=dd_score,
            total_score=total,
            status=status,
            recommendations=recs,
        )


# =============================================================================
# REGIME FAILURE ANALYSIS
# =============================================================================

def analyze_regime_failures(
    survival: Dict[str, RegimeSurvival],
) -> Dict[str, Any]:
    """Analyze regime failures."""
    failing = []
    surviving = []
    unstable = []

    for regime, s in survival.items():
        if s.survives_execution:
            surviving.append(regime)
        else:
            failing.append(regime)

        # Check stability (high execution cost)
        if s.execution_cost > 3.0:
            unstable.append(regime)

    return {
        "failing_regimes": failing,
        "surviving_regimes": surviving,
        "unstable_regimes": unstable,
        "total_analyzed": len(survival),
    }


# =============================================================================
# FACTORY
# =============================================================================

def create_certifier(
    mc_simulations: int = 2000,
) -> EdgeCertification:
    """Create certification engine."""
    config = CertificationConfig(mc_simulations=mc_simulations)
    return EdgeCertification(config)