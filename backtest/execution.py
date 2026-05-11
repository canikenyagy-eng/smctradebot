from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import pandas as pd


@dataclass(frozen=True)
class RealisticExecutionSettings:
    enabled: bool = False
    spread_default_pips: float = 0.0
    spread_by_pair: dict[str, float] = field(default_factory=dict)
    slippage_mode: str = "none"
    max_slippage_pips: float = 0.0
    execution_delay_bars: int = 0
    partial_fill_probability: float = 1.0
    partial_fill_min_ratio: float = 0.5
    limit_touch_tolerance_pips: float = 0.0
    apply_spread_to_limit: bool = False
    random_seed: int | None = None

    def normalized_pair(self, pair: str) -> str:
        return pair.upper().replace("/", "")

    def pip_size(self, pair: str) -> float:
        normalized = self.normalized_pair(pair)
        quote = normalized[3:6] if len(normalized) >= 6 else ""
        return 0.01 if quote == "JPY" else 0.0001

    def spread_pips(self, pair: str) -> float:
        normalized = self.normalized_pair(pair)
        return max(0.0, float(self.spread_by_pair.get(normalized, self.spread_default_pips)))

    def spread_price(self, pair: str) -> float:
        return self.spread_pips(pair) * self.pip_size(pair)

    def tolerance_price(self, pair: str) -> float:
        return max(0.0, self.limit_touch_tolerance_pips) * self.pip_size(pair)

    def sanitized(self) -> "RealisticExecutionSettings":
        mode = self.slippage_mode.strip().lower()
        if mode not in {"none", "random", "volatility"}:
            mode = "none"
        return RealisticExecutionSettings(
            enabled=bool(self.enabled),
            spread_default_pips=max(0.0, float(self.spread_default_pips)),
            spread_by_pair={k.upper().replace("/", ""): max(0.0, float(v)) for k, v in self.spread_by_pair.items()},
            slippage_mode=mode,
            max_slippage_pips=max(0.0, float(self.max_slippage_pips)),
            execution_delay_bars=max(0, int(self.execution_delay_bars)),
            partial_fill_probability=max(0.0, min(1.0, float(self.partial_fill_probability))),
            partial_fill_min_ratio=max(0.01, min(0.99, float(self.partial_fill_min_ratio))),
            limit_touch_tolerance_pips=max(0.0, float(self.limit_touch_tolerance_pips)),
            apply_spread_to_limit=bool(self.apply_spread_to_limit),
            random_seed=self.random_seed,
        )


def build_rng(seed: int | None) -> Random:
    return Random(seed) if seed is not None else Random()


def volatility_slippage_factor(frame: pd.DataFrame, index: int, lookback: int = 20) -> float:
    if frame.empty or index <= 0:
        return 0.0
    start = max(0, index - lookback + 1)
    scoped = frame.iloc[start : index + 1]
    if scoped.empty:
        return 0.0

    ranges = (scoped["high"].astype(float) - scoped["low"].astype(float)).abs()
    avg_range = float(ranges.mean() or 0.0)
    current_range = float(ranges.iloc[-1] or 0.0)
    if avg_range <= 1e-12:
        return 0.0

    ratio = current_range / avg_range
    return max(0.0, min(1.0, ratio / 2.0))
