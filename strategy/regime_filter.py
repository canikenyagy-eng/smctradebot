"""
Regime Filter Layer.

This module provides regime-based filtering to block trades in
unprofitable regimes under realistic execution conditions.

NO SMC logic modification - pure filtering overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import Settings


@dataclass(frozen=True)
class RegimeFilterConfig:
    """Configuration for regime filtering."""

    # Enable regime filtering
    enable_regime_filter: bool = False

    # Allow contraction trading
    enable_contraction_trading: bool = False

    # Score override for contraction (need score > this to allow)
    contraction_score_override: int = 85

    # Score threshold for range regime
    range_score_threshold: int = 10

    # Score threshold for all regimes
    min_score: int = 5

    # Default weights per regime (0 = blocked, 1 = full)
    regime_weights: tuple = (
        ("trend", 1.0),
        ("expansion", 1.0),
        ("range", 0.5),
        ("contraction", 0.0),
    )

    @classmethod
    def from_settings(cls, settings: Settings) -> "RegimeFilterConfig":
        """Create from Settings."""
        return cls(
            enable_regime_filter=bool(settings.enable_regime_filter),
            enable_contraction_trading=bool(settings.enable_contraction_trading),
            contraction_score_override=max(
                0, min(100, settings.contraction_min_trigger_strength)
            ),
            range_score_threshold=max(0, settings.range_min_trigger_strength),
            min_score=max(0, settings.min_score),
        )


@dataclass
class FilterResult:
    """Result of regime filtering."""

    allowed: bool
    rejected: bool = False
    reason: str = ""
    weight: float = 1.0
    regime: str = "neutral"
    score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "rejected": self.rejected,
            "reason": self.reason,
            "weight": self.weight,
            "regime": self.regime,
            "score": self.score,
        }


class RegimeFilter:
    """Regime-based trade filter."""

    # Regime to weight mapping
    REGIME_WEIGHTS = {
        "trend": 1.0,
        "expansion": 1.0,
        "range": 0.5,
        "contraction": 0.0,
    }

    # Regimes that are always allowed
    ALLOWED_REGIMES = {"trend", "expansion"}

    # Regimes that need score check
    SCORE_CHECK_REGIMES = {"range"}

    # Regimes that are blocked by default
    BLOCKED_REGIMES = {"contraction"}

    def __init__(self, config: RegimeFilterConfig | None = None):
        self.config = config or RegimeFilterConfig()

    def check(
        self,
        regime: str,
        score: int,
    ) -> FilterResult:
        """Check if trade is allowed.

        Args:
            regime: Current regime label
            score: Trade score

        Returns:
            FilterResult with decision
        """
        regime = regime.lower() if regime else "neutral"

        # If filtering disabled, allow all
        if not self.config.enable_regime_filter:
            return FilterResult(
                allowed=True,
                regime=regime,
                score=score,
            )

        # Check minimum score
        if score < self.config.min_score:
            return FilterResult(
                allowed=False,
                rejected=True,
                reason=f"score {score} < min {self.config.min_score}",
                weight=0.0,
                regime=regime,
                score=score,
            )

        # Blocked regime check
        if regime in self.BLOCKED_REGIMES:
            if not self.config.enable_contraction_trading:
                return FilterResult(
                    allowed=False,
                    rejected=True,
                    reason="CONTRACTION regime blocked",
                    weight=0.0,
                    regime=regime,
                    score=score,
                )
            # Check score override
            if score < self.config.contraction_score_override:
                return FilterResult(
                    allowed=False,
                    rejected=True,
                    reason=f"score {score} < override {self.config.contraction_score_override}",
                    weight=0.0,
                    regime=regime,
                    score=score,
                )
            # Override allows with reduced weight
            return FilterResult(
                allowed=True,
                reason="CONTRACTION with score override",
                weight=self.REGIME_WEIGHTS.get(regime, 0.0),
                regime=regime,
                score=score,
            )

        # Score check regime
        if regime in self.SCORE_CHECK_REGIMES:
            if score < self.config.range_score_threshold:
                return FilterResult(
                    allowed=False,
                    rejected=True,
                    reason=f"RANGE score {score} < threshold {self.config.range_score_threshold}",
                    weight=0.0,
                    regime=regime,
                    score=score,
                )
            # Apply reduced weight
            return FilterResult(
                allowed=True,
                reason="RANGE with score threshold",
                weight=self.REGIME_WEIGHTS.get(regime, 0.5),
                regime=regime,
                score=score,
            )

        # Always allowed regimes
        if regime in self.ALLOWED_REGIMES:
            return FilterResult(
                allowed=True,
                regime=regime,
                weight=self.REGIME_WEIGHTS.get(regime, 1.0),
                score=score,
            )

        # Unknown regime - default allow with warning
        return FilterResult(
            allowed=True,
            reason=f"unknown regime: {regime}",
            weight=1.0,
            regime=regime,
            score=score,
        )

    def get_weight(self, regime: str) -> float:
        """Get weight for regime."""
        return self.REGIME_WEIGHTS.get(regime.lower(), 1.0)


def create_filter(
    enable: bool = False,
    enable_contraction: bool = False,
) -> RegimeFilter:
    """Create regime filter."""
    config = RegimeFilterConfig(
        enable_regime_filter=enable,
        enable_contraction_trading=enable_contraction,
    )
    return RegimeFilter(config)