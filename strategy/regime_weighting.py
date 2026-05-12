"""
Regime-Weighted Scoring Layer.

This module provides regime-weighted score adjustment as an overlay
to the base signal scoring.

NO SMC logic modification - weighted overlay only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from strategy.regime_filter import RegimeFilter


@dataclass
class WeightedScoreResult:
    """Result of weighted scoring."""

    base_score: int
    adjusted_score: float
    regime_weight: float
    regime: str
    final_decision_score: float

    # Whether passes threshold
    passes_threshold: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_score": self.base_score,
            "adjusted_score": self.adjusted_score,
            "regime_weight": self.regime_weight,
            "regime": self.regime,
            "final_decision_score": self.final_decision_score,
            "passes_threshold": self.passes_threshold,
        }


class RegimeWeightedScorer:
    """Apply regime-based scoring overlay."""

    # Default regime weights
    DEFAULT_WEIGHTS = {
        "trend": 1.0,
        "expansion": 0.9,
        "range": 0.5,
        "contraction": 0.0,
    }

    def __init__(self, threshold: int = 10):
        self.threshold = threshold
        self.filter = RegimeFilter()

    def score(
        self,
        base_score: int,
        regime: str,
        custom_weights: Dict[str, float] | None = None,
    ) -> WeightedScoreResult:
        """Apply regime weighting to score.

        Args:
            base_score: Raw signal score
            regime: Current regime
            custom_weights: Optional custom weights per regime

        Returns:
            WeightedScoreResult with all scores
        """
        regime = regime.lower() if regime else "neutral"
        weights = custom_weights or self.DEFAULT_WEIGHTS

        # Get regime weight
        weight = weights.get(regime, 1.0)

        # Calculate adjusted score (before threshold)
        adjusted_score = base_score * weight

        # Calculate final decision score
        # This is what the decision layer sees
        final_score = adjusted_score

        # Check if passes threshold
        passes = final_score >= self.threshold

        return WeightedScoreResult(
            base_score=base_score,
            adjusted_score=adjusted_score,
            regime_weight=weight,
            regime=regime,
            final_decision_score=final_score,
            passes_threshold=passes,
        )

    def batch_score(
        self,
        trades: list[dict],
        custom_weights: Dict[str, float] | None = None,
    ) -> list[WeightedScoreResult]:
        """Score multiple trades.

        Args:
            trades: List of {score, regime} dicts
            custom_weights: Optional custom weights

        Returns:
            List of WeightedScoreResults
        """
        results = []
        for trade in trades:
            base = trade.get("score", 10)
            regime = trade.get("regime", "neutral")
            results.append(self.score(base, regime, custom_weights))
        return results


def create_scorer(
    threshold: int = 10,
    custom_weights: Dict[str, float] | None = None,
) -> RegimeWeightedScorer:
    """Create weighted scorer."""
    if custom_weights:
        return RegimeWeightedScorer(threshold=threshold)
    return RegimeWeightedScorer(threshold=threshold)