"""
Regime Engine v2 - Pure Market State Classification

This module classifies market into distinct regimes WITHOUT strategy logic.
No SMC concepts, no indicators - PURE price/volume analysis only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class RegimeType(Enum):
    """Market regime types."""
    TREND_STRONG = "trend_strong"
    TREND_WEAK = "trend_weak"
    RANGE_TIGHT = "range_tight"
    RANGE_WIDE = "range_wide"
    EXPANSION = "expansion"
    TRANSITION = "transition"


@dataclass(frozen=True)
class RegimeOutput:
    """Output from regime classifier."""
    regime: str
    confidence: float
    tradability_score: int
    volatility_estimate: float
    liquidity_quality: float
    risk_multiplier: float
    
    @property
    def is_tradable(self) -> bool:
        return self.tradability_score > 0
    
    @property
    def is_transition(self) -> bool:
        return self.regime == RegimeType.TRANSITION.value


# Default tradability scores per regime
REGIME_TRADABILITY: dict[str, int] = {
    RegimeType.TREND_STRONG.value: 90,
    RegimeType.TREND_WEAK.value: 70,
    RegimeType.RANGE_TIGHT.value: 80,
    RegimeType.RANGE_WIDE.value: 30,
    RegimeType.EXPANSION.value: 50,
    RegimeType.TRANSITION.value: 0,
}

# Regime multipliers for risk
REGIME_MULTIPLIERS: dict[str, float] = {
    RegimeType.TREND_STRONG.value: 1.0,
    RegimeType.TREND_WEAK.value: 0.7,
    RegimeType.RANGE_TIGHT.value: 0.8,
    RegimeType.RANGE_WIDE.value: 0.3,
    RegimeType.EXPANSION.value: 0.5,
    RegimeType.TRANSITION.value: 0.0,
}


def _compute_volatility(frame: pd.DataFrame, lookback: int = 20) -> float:
    """Compute normalized volatility (0-1 scale)."""
    if frame.empty or len(frame) < lookback:
        return 0.5
    
    recent = frame.tail(lookback)
    returns = recent["close"].pct_change().dropna()
    
    if returns.empty:
        return 0.5
    
    # Use annualized volatility normalized
    vol = returns.std()
    if vol <= 0:
        return 0.5
    
    # Normalize: typical forex volatility is 0.5-2% daily
    normalized = min(1.0, vol / 0.02)
    return round(max(0.0, min(1.0, normalized)), 4)


def _compute_range(frame: pd.DataFrame, lookback: int = 20) -> float:
    """Compute range (high-low) relative to close."""
    if frame.empty or len(frame) < 2:
        return 0.0
    
    recent = frame.tail(lookback)
    ranges = (recent["high"] - recent["low"]).abs()
    closes = recent["close"]
    
    if closes.empty or ranges.empty:
        return 0.0
    
    avg_range = ranges.mean()
    avg_close = closes.mean()
    
    if avg_close <= 0:
        return 0.0
    
    return round(float(avg_range / avg_close), 4)


def _compute_trend_strength(frame: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> float:
    """Compute trend strength (0-1)."""
    if frame.empty or len(frame) < long_window:
        return 0.0
    
    closes = frame["close"].astype(float)
    short_ma = closes.rolling(short_window).mean()
    long_ma = closes.rolling(long_window).mean()
    
    if short_ma.empty or long_ma.empty:
        return 0.0
    
    current_short = float(short_ma.iloc[-1])
    current_long = float(long_ma.iloc[-1])
    
    if current_long <= 0:
        return 0.0
    
    # Direction + strength
    direction = 1.0 if current_short > current_long else -1.0
    strength = abs(current_short - current_long) / current_long
    
    return direction * min(1.0, strength * 10)


def _detect_range(frame: pd.DataFrame, lookback: int = 20) -> tuple[bool, float]:
    """Detect if market is in range."""
    if frame.empty or len(frame) < lookback:
        return False, 0.0
    
    recent = frame.tail(lookback)
    high = recent["high"].max()
    low = recent["low"].min()
    close = recent["close"].iloc[-1]
    
    range_size = high - low
    if range_size <= 0:
        return False, 0.0
    
    # How close is close to range edges
    distance_to_high = (high - close) / range_size
    distance_to_low = (close - low) / range_size
    
    # If close is near middle of range
    in_range = distance_to_high > 0.2 and distance_to_low > 0.2
    
    # Range tightness (smaller = tighter)
    tightness = 1.0 - (range_size / close)
    
    return in_range, round(max(0.0, min(1.0, tightness)), 4)


def classify_regime(frame: pd.DataFrame) -> RegimeOutput:
    """
    Classify market regime using PURE price analysis.
    
    No SMC logic, no indicators - only price action.
    """
    if frame.empty or len(frame) < 50:
        return RegimeOutput(
            regime=RegimeType.TRANSITION.value,
            confidence=0.0,
            tradability_score=0,
            volatility_estimate=0.5,
            liquidity_quality=0.5,
            risk_multiplier=0.0,  # TRADABILITY = 0 for transition
        )
    
    # Compute metrics
    volatility = _compute_volatility(frame)
    trend = _compute_trend_strength(frame)
    in_range, range_tightness = _detect_range(frame)
    range_size = _compute_range(frame)
    
    # Classification logic
    regime = RegimeType.TRANSITION.value
    confidence = 0.0
    
    # Strong trend: high direction, high confidence
    if abs(trend) > 0.3 and volatility > 0.3:
        regime = RegimeType.TREND_STRONG.value
        confidence = min(1.0, abs(trend) + volatility)
    
    # Weak trend: some direction but lower confidence
    elif abs(trend) > 0.1 and abs(trend) <= 0.3:
        regime = RegimeType.TREND_WEAK.value
        confidence = min(1.0, abs(trend) * 2)
    
    # Range detection
    elif in_range:
        if range_tightness > 0.7:
            regime = RegimeType.RANGE_TIGHT.value
            confidence = range_tightness
        else:
            regime = RegimeType.RANGE_WIDE.value
            confidence = 1.0 - range_tightness
    
    # Expansion: high volatility, unclear direction
    elif volatility > 0.8:
        regime = RegimeType.EXPANSION.value
        confidence = min(1.0, volatility)
    
    # High volatility mixed signals
    elif volatility > 0.5 and abs(trend) < 0.1:
        regime = RegimeType.EXPANSION.value
        confidence = volatility
    
    # Default to transition for ambiguous states
    else:
        regime = RegimeType.TRANSITION.value
        confidence = 0.5
    
    # Compute tradability
    tradability = REGIME_TRADABILITY.get(regime, 0)
    
    # Adjust confidence based on data quality
    confidence = round(confidence, 4)
    
    # Liquidity quality (based on volatility - lower = higher quality)
    liquidity = round(max(0.0, 1.0 - volatility), 4)
    
    return RegimeOutput(
        regime=regime,
        confidence=confidence,
        tradability_score=tradability,
        volatility_estimate=volatility,
        liquidity_quality=liquidity,
        risk_multiplier=REGIME_MULTIPLIERS.get(regime, 1.0),
    )


def get_regime_multiplier(regime: str) -> float:
    """Get risk multiplier for regime."""
    return REGIME_MULTIPLIERS.get(regime, 1.0)


def is_trade_allowed(regime_output: RegimeOutput, min_tradability: int = 30) -> bool:
    """Check if trade allowed based on regime."""
    if regime_output.is_transition:
        return False
    if regime_output.tradability_score < min_tradability:
        return False
    return True