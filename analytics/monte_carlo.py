from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import mean, median
from typing import Iterable


@dataclass(frozen=True)
class MonteCarloSettings:
    iterations: int = 2000
    seed: int = 42
    ruin_drawdown_r: float = 10.0

    def sanitized(self) -> "MonteCarloSettings":
        return MonteCarloSettings(
            iterations=max(100, int(self.iterations)),
            seed=int(self.seed),
            ruin_drawdown_r=max(0.1, float(self.ruin_drawdown_r)),
        )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    q = max(0.0, min(1.0, percentile / 100.0))
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * weight)


def _path_stats(path: list[float]) -> tuple[float, float, int]:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    longest_loss_streak = 0
    current_loss_streak = 0

    for value in path:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        if value < 0:
            current_loss_streak += 1
            longest_loss_streak = max(longest_loss_streak, current_loss_streak)
        else:
            current_loss_streak = 0

    return equity, max_drawdown, longest_loss_streak


def run_monte_carlo(
    r_values: Iterable[float],
    settings: MonteCarloSettings | None = None,
) -> dict[str, object]:
    cfg = (settings or MonteCarloSettings()).sanitized()
    sample = [float(value) for value in r_values]
    if not sample:
        return {
            "sample_size": 0,
            "iterations": cfg.iterations,
            "ruin_drawdown_r": cfg.ruin_drawdown_r,
            "error": "no trades",
        }

    rng = random.Random(cfg.seed)
    terminal_r: list[float] = []
    max_drawdowns: list[float] = []
    loss_streaks: list[float] = []

    for _ in range(cfg.iterations):
        path = [rng.choice(sample) for _ in range(len(sample))]
        total_r, max_dd, longest_loss_streak = _path_stats(path)
        terminal_r.append(total_r)
        max_drawdowns.append(max_dd)
        loss_streaks.append(float(longest_loss_streak))

    ruin_count = sum(1 for drawdown in max_drawdowns if drawdown >= cfg.ruin_drawdown_r)
    positive_count = sum(1 for value in terminal_r if value > 0)
    return {
        "sample_size": len(sample),
        "iterations": cfg.iterations,
        "seed": cfg.seed,
        "ruin_drawdown_r": cfg.ruin_drawdown_r,
        "terminal_r": {
            "mean": round(mean(terminal_r), 6),
            "median": round(median(terminal_r), 6),
            "p05": round(_percentile(terminal_r, 5), 6),
            "p25": round(_percentile(terminal_r, 25), 6),
            "p75": round(_percentile(terminal_r, 75), 6),
            "p95": round(_percentile(terminal_r, 95), 6),
        },
        "max_drawdown_r": {
            "mean": round(mean(max_drawdowns), 6),
            "median": round(median(max_drawdowns), 6),
            "p95": round(_percentile(max_drawdowns, 95), 6),
            "p99": round(_percentile(max_drawdowns, 99), 6),
        },
        "longest_loss_streak": {
            "median": round(median(loss_streaks), 6),
            "p95": round(_percentile(loss_streaks, 95), 6),
            "p99": round(_percentile(loss_streaks, 99), 6),
        },
        "positive_terminal_probability": round(positive_count / cfg.iterations, 6),
        "risk_of_ruin_probability": round(ruin_count / cfg.iterations, 6),
    }

