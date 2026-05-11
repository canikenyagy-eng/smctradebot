from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeState:
    label: str
    direction: str
    volatility_ratio: float
    trend_strength: float
    confidence: float

    @property
    def is_directional(self) -> bool:
        return self.label in {"trend", "expansion"} and self.direction in {"bullish", "bearish"}


def analyze_regime(
    frame: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 80,
    min_bars: int = 120,
) -> RegimeState:
    if frame.empty or len(frame) < min_bars:
        return RegimeState(
            label="neutral",
            direction="neutral",
            volatility_ratio=1.0,
            trend_strength=0.0,
            confidence=0.0,
        )

    closes = frame["close"].astype(float)
    returns = closes.pct_change().dropna()
    if len(returns) < max(short_window, long_window):
        return RegimeState(
            label="neutral",
            direction="neutral",
            volatility_ratio=1.0,
            trend_strength=0.0,
            confidence=0.0,
        )

    short = returns.tail(short_window)
    long = returns.tail(long_window)
    eps = 1e-9

    vol_short = float(short.std(ddof=0) or 0.0)
    vol_long = float(long.std(ddof=0) or 0.0)
    vol_ratio = vol_short / max(vol_long, eps)
    mean_short = float(short.mean() or 0.0)
    trend_strength = abs(mean_short) / max(vol_short, eps)

    if mean_short > 0:
        direction = "bullish"
    elif mean_short < 0:
        direction = "bearish"
    else:
        direction = "neutral"

    if vol_ratio >= 1.15 and trend_strength >= 0.20:
        label = "expansion"
    elif trend_strength >= 0.28:
        label = "trend"
    elif vol_ratio <= 0.85 and trend_strength <= 0.15:
        label = "contraction"
    else:
        label = "range"

    confidence = min(
        1.0,
        max(
            0.0,
            0.25
            + min(trend_strength / 2.0, 0.5)
            + min(abs(vol_ratio - 1.0), 1.0) * 0.25,
        ),
    )

    return RegimeState(
        label=label,
        direction=direction,
        volatility_ratio=round(vol_ratio, 4),
        trend_strength=round(trend_strength, 4),
        confidence=round(confidence, 4),
    )
