"""
Regime Expectancy Recalibration Engine.

This module calculates execution-adjusted expectancy per regime
and computes the system-wide expected value.

NO strategy modification - pure analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import numpy as np


@dataclass
class RegimeExpectancy:
    """Expectancy data for a single regime."""

    regime: str
    trade_count: int = 0

    # Baseline metrics
    baseline_wins: int = 0
    baseline_losses: int = 0
    baseline_total_r: float = 0.0
    baseline_avg_r: float = 0.0
    baseline_expectancy: float = 0.0
    baseline_win_rate: float = 0.0

    # Realistic metrics
    realistic_wins: int = 0
    realistic_losses: int = 0
    realistic_total_r: float = 0.0
    realistic_avg_r: float = 0.0
    realistic_expectancy: float = 0.0
    realistic_win_rate: float = 0.0

    # Execution-adjusted metrics
    execution_cost_pips: float = 0.0
    execution_adjusted_expectancy: float = 0.0

    # Confidence
    confidence: float = 0.0
    sample_size_adequate: bool = False

    def compute_from_trades(self, trades: List[dict]) -> None:
        """Compute from trade records."""
        if not trades:
            return

        self.trade_count = len(trades)
        self.sample_size_adequate = self.trade_count >= 20

        # Extract values
        baseline_r = [t.get("baseline_r", 0) for t in trades]
        realistic_r = [t.get("realistic_r", 0) for t in trades]

        # Baseline
        wins = [r for r in baseline_r if r > 0]
        losses = [r for r in baseline_r if r < 0]

        self.baseline_wins = len(wins)
        self.baseline_losses = len(losses)
        self.baseline_total_r = sum(baseline_r)
        self.baseline_avg_r = np.mean(baseline_r) if baseline_r else 0.0
        self.baseline_win_rate = self.baseline_wins / self.trade_count if self.trade_count > 0 else 0.0

        # Expectancy baseline
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        self.baseline_expectancy = (
            self.baseline_win_rate * avg_win -
            (1 - self.baseline_win_rate) * avg_loss
        )

        # Realistic
        r_wins = [r for r in realistic_r if r > 0]
        r_losses = [r for r in realistic_r if r < 0]

        self.realistic_wins = len(r_wins)
        self.realistic_losses = len(r_losses)
        self.realistic_total_r = sum(realistic_r)
        self.realistic_avg_r = np.mean(realistic_r) if realistic_r else 0.0
        self.realistic_win_rate = self.realistic_wins / self.trade_count if self.trade_count > 0 else 0.0

        # Expectancy realistic
        avg_win = np.mean(r_wins) if r_wins else 0.0
        avg_loss = abs(np.mean(r_losses)) if r_losses else 0.0
        self.realistic_expectancy = (
            self.realistic_win_rate * avg_win -
            (1 - self.realistic_win_rate) * avg_loss
        )

        # Execution cost
        costs = [t.get("cost_pips", 0) for t in trades]
        self.execution_cost_pips = np.mean(costs) if costs else 0.0

        # Execution-adjusted expectancy
        self.execution_adjusted_expectancy = (
            self.realistic_expectancy - self.execution_cost_pips / 10.0  # Rough R conversion
        )

        # Confidence based on sample size and stability
        self._compute_confidence()

    def _compute_confidence(self) -> None:
        """Compute confidence score."""
        if self.trade_count < 10:
            self.confidence = 0.2
        elif self.trade_count < 20:
            self.confidence = 0.5
        else:
            # Higher if expectancy is stable
            if self.baseline_expectancy != 0:
                stability = 1.0 - abs(
                    self.realistic_expectancy - self.baseline_expectancy
                ) / abs(self.baseline_expectancy)
                self.confidence = min(1.0, max(0.5, stability))
            else:
                self.confidence = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "trade_count": self.trade_count,
            "baseline_expectancy": round(self.baseline_expectancy, 2),
            "realistic_expectancy": round(self.realistic_expectancy, 2),
            "execution_adjusted_expectancy": round(self.execution_adjusted_expectancy, 2),
            "confidence": round(self.confidence, 2),
        }


class RegimeExpectancyEngine:
    """Compute expectancy across all regimes."""

    def __init__(self):
        self.regimes: Dict[str, RegimeExpectancy] = {}

    def add_regime_data(self, regime: str, trades: List[dict]) -> None:
        """Add trades for a regime."""
        exp = RegimeExpectancy(regime=regime)
        exp.compute_from_trades(trades)
        self.regimes[regime] = exp

    def get_system_expectancy(
        self,
        weights: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """Get weighted system expectancy.

        Args:
            weights: Optional per-regime weights

        Returns:
            Dict with baseline, realistic, execution_adjusted
        """
        if not self.regimes:
            return {
                "baseline": 0.0,
                "realistic": 0.0,
                "execution_adjusted": 0.0,
                "count": 0,
            }

        weights = weights or {}

        baseline = 0.0
        realistic = 0.0
        exec_adj = 0.0
        total_weight = 0.0

        for regime, exp in self.regimes.items():
            w = weights.get(regime, 1.0)
            baseline += exp.baseline_expectancy * w
            realistic += exp.realistic_expectancy * w
            exec_adj += (
                exp.execution_adjusted_expectancy * w
            )
            total_weight += w

        if total_weight > 0:
            baseline /= total_weight
            realistic /= total_weight
            exec_adj /= total_weight

        return {
            "baseline": round(baseline, 2),
            "realistic": round(realistic, 2),
            "execution_adjusted": round(exec_adj, 2),
            "count": sum(e.trade_count for e in self.regimes.values()),
        }

    def get_report(self) -> pd.DataFrame:
        """Get expectancy report."""
        rows = []
        for regime, exp in sorted(self.regimes.items()):
            rows.append({
                "regime": regime.upper(),
                "trades": exp.trade_count,
                "baseline_E": round(exp.baseline_expectancy, 2),
                "realistic_E": round(exp.realistic_expectancy, 2),
                "exec_adj_E": round(exp.execution_adjusted_expectancy, 2),
                "confidence": f"{exp.confidence:.0%}",
            })
        return pd.DataFrame(rows)


def compute_expectancy_from_trades(
    trades_df: pd.DataFrame,
) -> RegimeExpectancyEngine:
    """Compute expectancy from trade DataFrame.

    Args:
        trades_df: DataFrame with columns:
            - regime
            - baseline_r
            - realistic_r
            - cost_pips

    Returns:
        RegimeExpectancyEngine
    """
    engine = RegimeExpectancyEngine()

    if trades_df.empty:
        return engine

    # Group by regime
    for regime, group in trades_df.groupby("regime"):
        trades = group.to_dict("records")
        engine.add_regime_data(regime, trades)

    return engine