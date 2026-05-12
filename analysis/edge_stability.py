"""
Enhanced Edge Stability Analysis.

This module provides enhanced analysis for edge stability including
Monte Carlo improvements, bootstrap resampling, and edge concentration analysis.

NO strategy modification - pure analysis layer.
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
class StabilityConfig:
    """Configuration for stability analysis."""

    # Monte Carlo settings
    mc_simulations: int = 10000
    bootstrap_samples: int = 1000
    trades_per_sim: int = 500

    # Stability scoring
    concentration_threshold: float = 0.6  # 60% threshold
    temperature: float = 1.5

    # Risk thresholds
    max_acceptable_dd: float = 0.15
    min_edge_ratio: float = 0.5


# =============================================================================
# OUTPUTS
# =============================================================================

@dataclass
class EnhancedMonteCarloResult:
    """Enhanced Monte Carlo results."""

    # Basic stats
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0

    # Risk metrics
    prob_loss: float = 0.0
    prob_loss_10pct: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Drawdown
    max_dd_mean: float = 0.0
    max_dd_95: float = 0.0
    max_dd_99: float = 0.0
    worst_1pct_dd: float = 0.0

    # Stability
    regime_stability_score: float = 0.0


@dataclass
class EdgeStabilityReport:
    """Edge stability report."""

    # Concentration
    herfindahl_index: float = 0.0
    concentration_index: float = 0.0
    diversification_score: float = 0.0

    # Edge normalization
    normalized_edge: float = 0.0
    raw_expectancy: float = 0.0

    # Risk flags
    is_concentrated: bool = False
    is_regime_restricted: bool = False

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# MONTE CARLO ENHANCEMENT
# =============================================================================

class EnhancedStabilityAnalyzer:
    """Enhanced stability analyzer."""

    def __init__(self, config: StabilityConfig | None = None):
        self.config = config or StabilityConfig()

    def run_enhanced_monte_carlo(
        self,
        trades: List[dict],
        regime_trades: Dict[str, List[dict]],
    ) -> EnhancedMonteCarloResult:
        """Run enhanced Monte Carlo simulation.

        Args:
            trades: List of {regime, exec_r}
            regime_trades: Trades grouped by regime

        Returns:
            EnhancedMonteCarloResult
        """
        cfg = self.config

        if not trades:
            return EnhancedMonteCarloResult()

        # Extract trade returns
        exec_r = [t.get("exec_r", 0) for t in trades]
        baseline_r = [t.get("base_r", t.get("exec_r", 0)) for t in trades]

        # Run standard MC
        mc_result = self._run_stratified_mc(exec_r, regime_trades)

        # Run bootstrap
        bootstrap_results = self._run_bootstrap(exec_r, baseline_r)

        # Combine results
        return EnhancedMonteCarloResult(
            mean_return=mc_result["mean"],
            median_return=mc_result["median"],
            std_return=mc_result["std"],
            prob_loss=mc_result["prob_loss"],
            prob_loss_10pct=mc_result["prob_loss_10"],
            var_95=mc_result["var_95"],
            cvar_95=bootstrap_results.get("cvar_95", mc_result["var_95"]),
            max_dd_mean=mc_result["max_dd_mean"],
            max_dd_95=mc_result["max_dd_95"],
            max_dd_99=mc_result["max_dd_99"],
            worst_1pct_dd=bootstrap_results.get("worst_1pct", mc_result["max_dd_95"]),
            regime_stability_score=bootstrap_results.get("regime_stability", 0.8),
        )

    def _run_stratified_mc(
        self,
        exec_r: List[float],
        regime_trades: Dict[str, List[dict]],
    ) -> Dict[str, float]:
        """Run stratified Monte Carlo."""
        cfg = self.config
        n = cfg.trades_per_sim

        results = []

        for _ in range(cfg.mc_simulations):
            # Stratified sampling
            sim_trades = []

            # Get regime weights
            regime_weights = {
                r: len(t) / len(exec_r) if exec_r else 0.25
                for r, t in regime_trades.items()
            }

            # Sample proportionally from each regime
            for regime, weight in regime_weights.items():
                n_regime = max(1, int(n * weight))
                regime_returns = [
                    t.get("exec_r", 0) for t in regime_trades.get(regime, [])
                ]
                if regime_returns:
                    sampled = np.random.choice(
                        regime_returns,
                        size=n_regime,
                        replace=True,
                    )
                    sim_trades.extend(sampled)

            # If we don't have enough, fill with random
            if len(sim_trades) < n:
                sim_trades.extend(
                    np.random.choice(exec_r, size=n-len(sim_trades), replace=True)
                )

            results.append(np.sum(sim_trades))

        results = np.array(results)

        return {
            "mean": np.mean(results),
            "median": np.median(results),
            "std": np.std(results),
            "prob_loss": np.mean(results < 0),
            "prob_loss_10": np.mean(results < -10),
            "var_95": np.percentile(results, 5),
            "max_dd_mean": self._estimate_max_dd(results),
            "max_dd_95": self._estimate_max_dd_percentile(results, 95),
            "max_dd_99": self._estimate_max_dd_percentile(results, 99),
        }

    def _run_bootstrap(
        self,
        exec_r: List[float],
        baseline_r: List[float],
    ) -> Dict[str, float]:
        """Run bootstrap resampling."""
        cfg = self.config

        if len(exec_r) < 10:
            return {"cvar_95": 0, "worst_1pct": 0, "regime_stability": 0.8}

        cvar_samples = []
        worst_samples = []

        for _ in range(cfg.bootstrap_samples):
            # Bootstrap sample (with replacement)
            idx = np.random.choice(len(exec_r), size=len(exec_r), replace=True)
            boot_sample = [exec_r[i] for i in idx]

            # Calculate CVaR (expected return given < VaR)
            var_95 = np.percentile(boot_sample, 5)
            cvar = np.mean([r for r in boot_sample if r <= var_95])
            cvar_samples.append(cvar)

            # Estimate worst 1%
            worst_samples.append(min(boot_sample))

        return {
            "cvar_95": np.mean(cvar_samples),
            "worst_1pct": np.percentile(worst_samples, 1),
            "regime_stability": 1.0 - abs(np.mean(cvar_samples) / np.mean(exec_r))
            if np.mean(exec_r) != 0 else 0.5,
        }

    def _estimate_max_dd(self, returns: np.ndarray) -> float:
        """Estimate max drawdown from cumulative returns."""
        equity = np.cumsum(returns)
        running_max = np.maximum.accumulate(equity)
        dd = running_max - equity
        return np.mean(dd) if len(dd) > 0 else 0

    def _estimate_max_dd_percentile(
        self,
        returns: np.ndarray,
        percentile: float,
    ) -> float:
        """Estimate max drawdown at percentile."""
        equity = np.cumsum(returns)
        running_max = np.maximum.accumulate(equity)
        dd = running_max - equity
        return np.percentile(dd, percentile)

    def analyze_stability(
        self,
        regime_trades: Dict[str, List[dict]],
    ) -> EdgeStabilityReport:
        """Analyze edge stability and concentration.

        Args:
            regime_trades: Trades grouped by regime

        Returns:
            EdgeStabilityReport
        """
        cfg = self.config

        if not regime_trades:
            return EdgeStabilityReport()

        # Calculate regime weights
        total = sum(len(t) for t in regime_trades.values())
        if total == 0:
            return EdgeStabilityReport()
        
        weights = {r: len(t) / total for r, t in regime_trades.items()}

        # Herfindahl index (concentration)
        hhi = sum(w ** 2 for w in weights.values())

        # Concentration index (focus on TREND + EXPANSION)
        trend_exp = weights.get("trend", 0) + weights.get("expansion", 0)
        concentration = trend_exp

        # Is concentrated?
        is_concentrated = concentration > cfg.concentration_threshold

        # Diversification score (inverse of concentration)
        divers_score = max(0, 1 - concentration)

        # Normalized edge (per regime)
        raw_edge = 0
        for regime, trades_list in regime_trades.items():
            if trades_list:
                e = np.mean([t.get("exec_r", 0) for t in trades_list])
                raw_edge += e * weights.get(regime, 0)

        # Apply temperature scaling
        temp_edge = raw_edge
        for regime, trades_list in regime_trades.items():
            if trades_list:
                e = np.mean([t.get("exec_r", 0) for t in trades_list])
                score = np.exp(e / cfg.temperature)
                temp_edge += score * weights.get(regime, 0)

        normalized = temp_edge / (len(regime_trades) + 0.001)

        # Risk flags
        is_regime_restricted = len([r for r in regime_trades if len(regime_trades[r]) > 0]) < 3

        # Recommendations
        recs = []
        if is_concentrated:
            recs.append("Edge is concentrated in TREND/EXPANSION - consider diversification")
        if hhi > 0.4:
            recs.append("High concentration - increase regime diversity")
        if raw_edge < cfg.min_edge_ratio:
            recs.append("Edge below threshold - review execution costs")

        return EdgeStabilityReport(
            herfindahl_index=hhi,
            concentration_index=concentration,
            diversification_score=divers_score,
            normalized_edge=normalized,
            raw_expectancy=raw_edge,
            is_concentrated=is_concentrated,
            is_regime_restricted=is_regime_restricted,
            recommendations=recs,
        )


# =============================================================================
# FACTORY
# =============================================================================

def create_stability_analyzer(
    mc_simulations: int = 10000,
) -> EnhancedStabilityAnalyzer:
    """Create enhanced stability analyzer."""
    config = StabilityConfig(mc_simulations=mc_simulations)
    return EnhancedStabilityAnalyzer(config)