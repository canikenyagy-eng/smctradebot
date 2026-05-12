"""
RANGE Regime Stability Filter.

This overlay layer filters RANGE trades by requiring specific
conditions to be met for acceptance.

NO SMC modification - pure filtering overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RangeFilterConfig:
    """Configuration for RANGE filtering."""

    # Enable filter
    enable_filter: bool = True

    # Required score threshold
    min_confidence_score: float = 0.5

    # Volatility threshold (ATR-based)
    atr_compression_threshold: float = 0.8

    # Require liquidity condition
    require_liquidity: bool = True

    # Require volatility compression
    require_volatility: bool = True

    # Require transition condition
    require_transition: bool = True


# =============================================================================
# FILTERS
# =============================================================================

@dataclass
class RangeFilterResult:
    """Result of RANGE filtering."""

    allowed: bool
    confidence_score: float
    rejection_reason: str = ""

    # Details
    liquidity_confirmed: bool = False
    volatility_confirmed: bool = False
    transition_confirmed: bool = False
    smt_confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "confidence_score": self.confidence_score,
            "rejection_reason": self.rejection_reason,
            "liquidity_confirmed": self.liquidity_confirmed,
            "volatility_confirmed": self.volatility_confirmed,
            "transition_confirmed": self.transition_confirmed,
            "smt_confirmed": self.smt_confirmed,
        }


class RangeStabilityFilter:
    """Filter RANGE trades that don't meet stability criteria."""

    def __init__(self, config: RangeFilterConfig | None = None):
        self.config = config or RangeFilterConfig()

    def check(
        self,
        price_data: pd.DataFrame | None = None,
        liquidity_sweeps: List | None = None,
        smt_signal: Any = None,
    ) -> RangeFilterResult:
        """Check if RANGE trade meets stability criteria.

        Args:
            price_data: OHLCV data with ATR
            liquidity_sweeps: List of detected liquidity sweeps
            smt_signal: SMT divergence signal

        Returns:
            RangeFilterResult
        """
        if not self.config.enable_filter:
            return RangeFilterResult(
                allowed=True,
                confidence_score=1.0,
            )

        # Track conditions
        conditions = []

        # 1. Liquidity condition (equal highs/lows or sweeps)
        liq_confirmed = self._check_liquidity(
            price_data, liquidity_sweeps
        )
        conditions.append(liq_confirmed)

        # 2. Volatility compression condition
        vol_confirmed = self._check_volatility(price_data)
        conditions.append(vol_confirmed)

        # 3. Transition condition
        trans_confirmed = self._check_transition(price_data)
        conditions.append(trans_confirmed)

        # 4. SMT condition (optional)
        smt_confirmed = smt_signal is not None if smt_signal else False
        conditions.append(smt_confirmed)

        # Calculate confidence score
        score = self._compute_confidence(
            liq_confirmed, vol_confirmed, trans_confirmed, smt_confirmed
        )

        # Determine if allowed
        allowed = (
            score >= self.config.min_confidence_score and
            conditions.count(True) >= 2  # At least 2 conditions
        )

        # Build rejection reason
        if not allowed:
            reasons = []
            if not liq_confirmed:
                reasons.append("no liquidity condition")
            if not vol_confirmed:
                reasons.append("no volatility compression")
            if not trans_confirmed:
                reasons.append("no transition signal")
            rejection = "; ".join(reasons)
        else:
            rejection = ""

        return RangeFilterResult(
            allowed=allowed,
            confidence_score=score,
            rejection_reason=rejection,
            liquidity_confirmed=liq_confirmed,
            volatility_confirmed=vol_confirmed,
            transition_confirmed=trans_confirmed,
            smt_confirmed=smt_confirmed,
        )

    def _check_liquidity(
        self,
        data: pd.DataFrame | None,
        sweeps: List | None,
    ) -> bool:
        """Check for liquidity condition."""
        if data is None or len(data) < 10:
            return True  # Allow if no data

        # Check for equal highs or equal lows
        highs = data["high"].tail(10)
        lows = data["low"].tail(10)

        # Equal highs
        equal_highs = highs.nunique() <= 3
        equal_lows = lows.nunique() <= 3

        # Check for recent sweeps
        has_sweeps = False
        if sweeps:
            if len(sweeps) >= 2:
                has_sweeps = True

        return equal_highs or equal_lows or has_sweeps

    def _check_volatility(self, data: pd.DataFrame | None) -> bool:
        """Check for volatility compression."""
        if data is None:
            return True

        # Check for ATR column
        if "atr" not in data.columns:
            return True  # Allow if no ATR

        # Compare to rolling mean
        atr = data["atr"].tail(10)
        if len(atr) < 10:
            return True

        atr_mean = atr.mean()
        atr_current = atr.iloc[-1]

        # Current ATR should be below threshold relative to mean
        threshold = self.config.atr_compression_threshold
        return atr_current < (atr_mean * threshold)

    def _check_transition(self, data: pd.DataFrame | None) -> bool:
        """Check for transition signal."""
        if data is None or len(data) < 20:
            return True

        # Check for compression → expansion transition
        # By comparing recent vs longer-term volatility
        recent_std = data["close"].tail(5).std()
        longer_std = data["close"].tail(20).std()

        if longer_std == 0:
            return True

        ratio = recent_std / longer_std

        # Transition if ratio is increasing
        return ratio > 1.1

    def _compute_confidence(
        self,
        liq: bool,
        vol: bool,
        trans: bool,
        smt: bool,
    ) -> float:
        """Compute confidence score."""
        if not self.config.require_liquidity:
            liq = True
        if not self.config.require_volatility:
            vol = True
        if not self.config.require_transition:
            trans = True

        # Count confirmed
        confirmed = sum([liq, vol, trans, smt])

        # Score based on confirmed conditions
        return confirmed / 4


# =============================================================================
# FACTORY
# =============================================================================

def create_range_filter(
    enable: bool = True,
) -> RangeStabilityFilter:
    """Create RANGE filter."""
    config = RangeFilterConfig(enable_filter=enable)
    return RangeStabilityFilter(config)