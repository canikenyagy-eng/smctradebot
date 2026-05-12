"""
Regime-Aware Execution Degradation Analysis.

This module analyzes how execution costs (spread, slippage, latency, partial fills)
affect system profitability across different market regimes.

Metrics computed:
- Execution cost breakdown per regime
- PnL degradation per regime
- Expectancy comparison (baseline vs realistic)
- Edge survival matrix
- Execution sensitivity score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


# =============================================================================
# REGIME DEFINITIONS
# =============================================================================

REGIME_LABELS = [
    "trend",      # Directional with momentum
    "range",      # Low directional, oscillating
    "contraction",  # Low volatility
    "expansion",  # High volatility + directional
    "neutral",    # Unknown/undetermined
]


# =============================================================================
# TRADE REPRESENTATION
# =============================================================================

@dataclass
class TradeRecord:
    """Single trade with execution details."""
    trade_id: str
    pair: str
    direction: str  # BUY or SELL
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    
    # Baseline execution (no costs)
    baseline_entry: float = 0.0
    baseline_exit: float = 0.0
    
    # Realistic execution (with costs)
    realistic_entry: float = 0.0
    realistic_exit: float = 0.0
    
    # Execution costs
    entry_slippage_pips: float = 0.0
    exit_slippage_pips: float = 0.0
    spread_cost_pips: float = 0.0
    latency_cost_pips: float = 0.0
    fill_ratio: float = 1.0
    
    # Regime
    entry_regime: str = "neutral"
    exit_regime: str = "neutral"
    
    @property
    def baseline_pnl(self) -> float:
        """Baseline PnL in price terms."""
        mult = 1.0 if self.direction == "BUY" else -1.0
        return (self.baseline_exit - self.baseline_entry) * self.size * mult
    
    @property
    def baseline_pnl_r(self) -> float:
        """Baseline PnL in R terms (R = risk per trade)."""
        risk = abs(self.baseline_entry - self.entry_price) if self.entry_price > 0 else 0.001
        if risk <= 0:
            return 0.0
        return self.baseline_pnl / risk
    
    @property
    def realistic_pnl(self) -> float:
        """Realistic PnL with all costs."""
        mult = 1.0 if self.direction == "BUY" else -1.0
        return (self.realistic_exit - self.realistic_entry) * self.size * mult
    
    @property
    def realistic_pnl_r(self) -> float:
        """Realistic PnL in R terms."""
        risk = abs(self.realistic_entry - self.entry_price) if self.entry_price > 0 else 0.001
        if risk <= 0:
            return 0.0
        return self.realistic_pnl / risk
    
    @property
    def total_cost_pips(self) -> float:
        """Total execution cost in pips."""
        return (
            self.entry_slippage_pips +
            self.exit_slippage_pips +
            self.spread_cost_pips +
            self.latency_cost_pips
        )
    
    @property
    def pnl_degradation(self) -> float:
        """PnL degradation (baseline - realistic)."""
        return self.baseline_pnl - self.realistic_pnl
    
    @property
    def pnl_degradation_pct(self) -> float:
        """PnL degradation as percentage."""
        if self.baseline_pnl == 0:
            return 0.0
        return (self.pnl_degradation / abs(self.baseline_pnl)) * 100
    
    @property
    def is_winner(self) -> bool:
        """Baseline win status."""
        return self.baseline_pnl > 0
    
    @property
    def realistic_winner(self) -> bool:
        """Realistic win status."""
        return self.realistic_pnl > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "direction": self.direction,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "entry_regime": self.entry_regime,
            "exit_regime": self.exit_regime,
            "baseline_pnl": self.baseline_pnl,
            "baseline_pnl_r": self.baseline_pnl_r,
            "realistic_pnl": self.realistic_pnl,
            "realistic_pnl_r": self.realistic_pnl_r,
            "total_cost_pips": self.total_cost_pips,
            "pnl_degradation": self.pnl_degradation,
            "pnl_degradation_pct": self.pnl_degradation_pct,
            "is_winner": self.is_winner,
            "realistic_winner": self.realistic_winner,
        }


# =============================================================================
# REGIME METRICS
# =============================================================================

@dataclass
class RegimeMetrics:
    """Metrics for a single regime."""
    
    regime: str = "neutral"
    trade_count: int = 0
    
    # Baseline metrics
    baseline_wins: int = 0
    baseline_losses: int = 0
    baseline_total_pnl: float = 0.0
    baseline_win_rate: float = 0.0
    baseline_avg_r: float = 0.0
    baseline_expectancy: float = 0.0
    
    # Realistic metrics
    realistic_wins: int = 0
    realistic_losses: int = 0
    realistic_total_pnl: float = 0.0
    realistic_win_rate: float = 0.0
    realistic_avg_r: float = 0.0
    realistic_expectancy: float = 0.0
    
    # Execution costs
    total_spread_cost_pips: float = 0.0
    total_slippage_cost_pips: float = 0.0
    total_latency_cost_pips: float = 0.0
    avg_cost_per_trade: float = 0.0
    
    # Partial fills
    partial_fills: int = 0
    partial_fill_rate: float = 0.0
    
    # Degradation
    pnl_degradation: float = 0.0
    pnl_degradation_pct: float = 0.0
    
    # Sensitivity
    sensitivity_score: float = 0.0
    
    def compute_from_trades(self, trades: List[TradeRecord]) -> None:
        """Compute metrics from trade list."""
        if not trades:
            return
        
        self.trade_count = len(trades)
        
        # Baseline metrics
        wins = [t for t in trades if t.is_winner]
        losses = [t for t in trades if not t.is_winner]
        
        self.baseline_wins = len(wins)
        self.baseline_losses = len(losses)
        self.baseline_total_pnl = sum(t.baseline_pnl for t in trades)
        self.baseline_win_rate = self.baseline_wins / self.trade_count if self.trade_count > 0 else 0.0
        
        rs = [t.baseline_pnl_r for t in trades]
        self.baseline_avg_r = np.mean(rs) if len(rs) > 0 else 0.0
        
        # Expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        win_r = [t.baseline_pnl_r for t in wins] if wins else []
        loss_r = [t.baseline_pnl_r for t in losses] if losses else []
        
        avg_win = np.mean(win_r) if win_r else 0.0
        avg_loss = abs(np.mean(loss_r)) if loss_r else 0.0
        
        self.baseline_expectancy = (
            self.baseline_win_rate * avg_win -
            (1 - self.baseline_win_rate) * avg_loss
        )
        
        # Realistic metrics
        r_wins = [t for t in trades if t.realistic_winner]
        r_losses = [t for t in trades if not t.realistic_winner]
        
        self.realistic_wins = len(r_wins)
        self.realistic_losses = len(r_losses)
        self.realistic_total_pnl = sum(t.realistic_pnl for t in trades)
        self.realistic_win_rate = self.realistic_wins / self.trade_count if self.trade_count > 0 else 0.0
        
        rs = [t.realistic_pnl_r for t in trades]
        self.realistic_avg_r = np.mean(rs) if len(rs) > 0 else 0.0
        
        # Expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        win_r = [t.realistic_pnl_r for t in r_wins] if r_wins else []
        loss_r = [t.realistic_pnl_r for t in r_losses] if r_losses else []
        
        avg_win = np.mean(win_r) if win_r else 0.0
        avg_loss = abs(np.mean(loss_r)) if loss_r else 0.0
        
        self.realistic_expectancy = (
            self.realistic_win_rate * avg_win -
            (1 - self.realistic_win_rate) * avg_loss
        )
        
        # Execution costs
        self.total_spread_cost_pips = sum(t.spread_cost_pips for t in trades)
        self.total_slippage_cost_pips = sum(t.entry_slippage_pips + t.exit_slippage_pips for t in trades)
        self.total_latency_cost_pips = sum(t.latency_cost_pips for t in trades)
        
        costs = [t.total_cost_pips for t in trades]
        self.avg_cost_per_trade = np.mean(costs) if costs else 0.0
        
        # Partial fills
        partial = [t for t in trades if t.fill_ratio < 1.0]
        self.partial_fills = len(partial)
        self.partial_fill_rate = self.partial_fills / self.trade_count if self.trade_count > 0 else 0.0
        
        # Degradation
        self.pnl_degradation = self.baseline_total_pnl - self.realistic_total_pnl
        if self.baseline_total_pnl != 0:
            self.pnl_degradation_pct = (
                self.pnl_degradation / abs(self.baseline_total_pnl)
            ) * 100
        
        # Sensitivity score (0-100)
        self.sensitivity_score = self._calculate_sensitivity()
    
    def _calculate_sensitivity(self) -> float:
        """Calculate execution sensitivity score (0-100).
        
        Based on:
        - Cost impact relative to PnL
        - Expectancy degradation
        - Fill quality
        """
        if self.trade_count == 0:
            return 0.0
        
        # Cost impact ratio (0-40 points)
        avg_cost = self.avg_cost_per_trade
        cost_impact = min(40, avg_cost * 20)  # 2 pips = 40 points
        
        # Expectancy degradation (0-40 points)
        expect_deg = max(0, self.baseline_expectancy - self.realistic_expectancy)
        expect_impact = min(40, expect_deg * 40)
        
        # Partial fill impact (0-20 points)
        fill_impact = self.partial_fill_rate * 20
        
        return cost_impact + expect_impact + fill_impact
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "trade_count": self.trade_count,
            # Baseline
            "baseline_wins": self.baseline_wins,
            "baseline_losses": self.baseline_losses,
            "baseline_total_pnl": self.baseline_total_pnl,
            "baseline_win_rate": self.baseline_win_rate,
            "baseline_avg_r": self.baseline_avg_r,
            "baseline_expectancy": self.baseline_expectancy,
            # Realistic
            "realistic_wins": self.realistic_wins,
            "realistic_losses": self.realistic_losses,
            "realistic_total_pnl": self.realistic_total_pnl,
            "realistic_win_rate": self.realistic_win_rate,
            "realistic_avg_r": self.realistic_avg_r,
            "realistic_expectancy": self.realistic_expectancy,
            # Costs
            "total_spread_cost_pips": self.total_spread_cost_pips,
            "total_slippage_cost_pips": self.total_slippage_cost_pips,
            "total_latency_cost_pips": self.total_latency_cost_pips,
            "avg_cost_per_trade": self.avg_cost_per_trade,
            "partial_fills": self.partial_fills,
            "partial_fill_rate": self.partial_fill_rate,
            # Degradation
            "pnl_degradation": self.pnl_degradation,
            "pnl_degradation_pct": self.pnl_degradation_pct,
            # Sensitivity
            "sensitivity_score": self.sensitivity_score,
        }


# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

class RegimeExecutionAnalyzer:
    """Analyze execution degradation by regime."""
    
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self._metrics_by_regime: Dict[str, RegimeMetrics] = {}
    
    def add_trade(self, trade: TradeRecord) -> None:
        """Add a trade record."""
        self.trades.append(trade)
    
    def get_metrics_by_entry_regime(self) -> Dict[str, RegimeMetrics]:
        """Get metrics segmented by entry regime."""
        if self._metrics_by_regime:
            return self._metrics_by_regime
        
        # Group trades by entry regime
        by_regime: Dict[str, List[TradeRecord]] = {}
        for trade in self.trades:
            regime = trade.entry_regime
            if regime not in by_regime:
                by_regime[regime] = []
            by_regime[regime].append(trade)
        
        # Compute metrics for each regime
        for regime, trades in by_regime.items():
            metrics = RegimeMetrics(regime=regime)
            metrics.compute_from_trades(trades)
            self._metrics_by_regime[regime] = metrics
        
        return self._metrics_by_regime
    
    def get_edge_survival_matrix(self) -> pd.DataFrame:
        """Get edge survival matrix."""
        metrics = self.get_metrics_by_entry_regime()
        
        rows = []
        for regime, m in sorted(metrics.items()):
            baseline_survives = m.baseline_expectancy > 0
            realistic_survives = m.realistic_expectancy > 0
            
            rows.append({
                "regime": regime,
                "baseline_edge": f"{m.baseline_expectancy:.2f}R",
                "realistic_edge": f"{m.realistic_expectancy:.2f}R",
                "survives": "Y" if realistic_survives else "N",
                "sensitivity": self._interpret_sensitivity(m.sensitivity_score),
            })
        
        return pd.DataFrame(rows)
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table by regime."""
        metrics = self.get_metrics_by_entry_regime()
        
        rows = []
        for regime, m in sorted(metrics.items()):
            rows.append({
                "regime": regime.upper(),
                "trades": m.trade_count,
                "baseline_pnl": round(m.baseline_total_pnl, 2),
                "realistic_pnl": round(m.realistic_total_pnl, 2),
                "degradation": f"{m.pnl_degradation_pct:.1f}%",
                "win_rate_base": f"{m.baseline_win_rate:.0%}",
                "win_rate_real": f"{m.realistic_win_rate:.0%}",
            })
        
        return pd.DataFrame(rows)
    
    def get_cost_breakdown(self) -> pd.DataFrame:
        """Get execution cost breakdown."""
        metrics = self.get_metrics_by_entry_regime()
        
        rows = []
        for regime, m in sorted(metrics.items()):
            if m.trade_count == 0:
                continue
            rows.append({
                "regime": regime.upper(),
                "avg_spread_pips": round(m.total_spread_cost_pips / m.trade_count, 2),
                "avg_slippage_pips": round(m.total_slippage_cost_pips / m.trade_count, 2),
                "avg_latency_pips": round(m.total_latency_cost_pips / m.trade_count, 2),
                "total_cost": round(m.avg_cost_per_trade, 2),
                "partial_fill_rate": f"{m.partial_fill_rate:.0%}",
            })
        
        return pd.DataFrame(rows)
    
    def get_expectancy_comparison(self) -> pd.DataFrame:
        """Get expectancy comparison."""
        metrics = self.get_metrics_by_entry_regime()
        
        rows = []
        for regime, m in sorted(metrics.items()):
            deg_pct = (
                (m.baseline_expectancy - m.realistic_expectancy) /
                max(abs(m.baseline_expectancy), 0.01) * 100
                if m.baseline_expectancy != 0 else 0
            )
            
            rows.append({
                "regime": regime.upper(),
                "baseline_E[R]": round(m.baseline_expectancy, 2),
                "realistic_E[R]": round(m.realistic_expectancy, 2),
                "degradation": f"{deg_pct:.1f}%",
                "status": "SURVIVES" if m.realistic_expectancy > 0 else "FAILS",
            })
        
        return pd.DataFrame(rows)
    
    def get_systemic_insights(self) -> Dict[str, Any]:
        """Extract systemic insights."""
        metrics = self.get_metrics_by_entry_regime()
        
        if not metrics:
            return {"error": "No data"}
        
        # Total PnL
        total_baseline = sum(m.baseline_total_pnl for m in metrics.values())
        total_realistic = sum(m.realistic_total_pnl for m in metrics.values())
        
        # Which regime contributes most
        regime_contrib = {
            r: m.baseline_total_pnl for r, m in metrics.items()
        }
        best_regime = max(regime_contrib, key=regime_contrib.get)
        worst_regime = min(regime_contrib, key=regime_contrib.get)
        
        # Failing regimes
        failing = [
            r for r, m in metrics.items()
            if m.realistic_expectancy <= 0
        ]
        
        # Robust regimes
        robust = [
            r for r, m in metrics.items()
            if m.sensitivity_score < 30
        ]
        
        # Fragile regimes
        fragile = [
            r for r, m in metrics.items()
            if m.sensitivity_score > 70
        ]
        
        return {
            "total_baseline_pnl": total_baseline,
            "total_realistic_pnl": total_realistic,
            "overall_degradation_pct": (
                (total_baseline - total_realistic) / max(abs(total_baseline), 0.01) * 100
                if total_baseline != 0 else 0
            ),
            "best_regime": best_regime,
            "worst_regime": worst_regime,
            "failing_regimes": failing,
            "robust_regimes": robust,
            "fragile_regimes": fragile,
            "is_regime_dependent": len(failing) > 0,
            "edge_survives": total_realistic > 0,
        }
    
    def _interpret_sensitivity(self, score: float) -> str:
        """Interpret sensitivity score."""
        if score < 30:
            return "ROBUST"
        elif score < 70:
            return "MEDIUM"
        else:
            return "FRAGILE"
    
    def generate_report(self) -> str:
        """Generate full report."""
        lines = []
        lines.append("=" * 70)
        lines.append("REGIME-AWARE EXECUTION DEGRADATION ANALYSIS")
        lines.append("=" * 70)
        
        # Summary table
        summary = self.get_summary_table()
        if not summary.empty:
            lines.append("\n## 1. REGIME SUMMARY")
            lines.append("-" * 40)
            lines.append(summary.to_string(index=False))
        
        # Cost breakdown
        costs = self.get_cost_breakdown()
        if not costs.empty:
            lines.append("\n## 2. EXECUTION COST BREAKDOWN")
            lines.append("-" * 40)
            lines.append(costs.to_string(index=False))
        
        # Expectancy comparison
        expect = self.get_expectancy_comparison()
        if not expect.empty:
            lines.append("\n## 3. REGIME EXPECTANCY COMPARISON")
            lines.append("-" * 40)
            lines.append(expect.to_string(index=False))
        
        # Edge survival matrix
        edge = self.get_edge_survival_matrix()
        if not edge.empty:
            lines.append("\n## 4. EDGE SURVIVAL MATRIX")
            lines.append("-" * 40)
            lines.append(edge.to_string(index=False))
        
        # Systemic insights
        insights = self.get_systemic_insights()
        lines.append("\n## 5. SYSTEMIC INSIGHTS")
        lines.append("-" * 40)
        lines.append(f"Best regime: {insights.get('best_regime', 'N/A')}")
        lines.append(f"Worst regime: {insights.get('worst_regime', 'N/A')}")
        lines.append(f"Failing regimes: {insights.get('failing_regimes', [])}")
        lines.append(f"Robust regimes: {insights.get('robust_regimes', [])}")
        lines.append(f"Fragile regimes: {insights.get('fragile_regimes', [])}")
        lines.append(f"System is regime-dependent: {insights.get('is_regime_dependent', False)}")
        lines.append(f"Edge survives: {insights.get('edge_survives', False)}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_regime_analyzer() -> RegimeExecutionAnalyzer:
    """Create regime execution analyzer."""
    return RegimeExecutionAnalyzer()


def analyze_from_backtest_results(
    trades_df: pd.DataFrame,
    execution_df: pd.DataFrame | None = None,
) -> RegimeExecutionAnalyzer:
    """Analyze regime execution from backtest results.
    
    Args:
        trades_df: DataFrame with trade records
        execution_df: DataFrame with execution costs per trade
    
    Returns:
        RegimeExecutionAnalyzer with computed metrics
    """
    analyzer = create_regime_analyzer()
    
    # This would be called with actual backtest data
    # For now, returns the empty analyzer
    return analyzer