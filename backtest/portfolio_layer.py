from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from core.signal_engine import TradeSignal


@dataclass(frozen=True)
class PortfolioLayerSettings:
    enabled: bool = False
    mode: str = "analysis_only"
    min_multiplier: float = 0.70
    max_multiplier: float = 1.25
    learning_window: int = 30
    min_trades_per_sleeve: int = 5
    max_sleeve_concentration: float = 0.55

    def sanitized(self) -> "PortfolioLayerSettings":
        mode = self.mode.strip().lower()
        if mode not in {"analysis_only", "apply"}:
            mode = "analysis_only"
        min_mult = max(0.05, float(self.min_multiplier))
        max_mult = max(min_mult, float(self.max_multiplier))
        return PortfolioLayerSettings(
            enabled=bool(self.enabled),
            mode=mode,
            min_multiplier=min_mult,
            max_multiplier=max_mult,
            learning_window=max(5, int(self.learning_window)),
            min_trades_per_sleeve=max(1, int(self.min_trades_per_sleeve)),
            max_sleeve_concentration=max(0.10, min(0.95, float(self.max_sleeve_concentration))),
        )


@dataclass(frozen=True)
class PortfolioDecision:
    sleeve: str
    multiplier: float
    applied: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "sleeve": self.sleeve,
            "multiplier": round(float(self.multiplier), 6),
            "applied": self.applied,
            "reason": self.reason,
        }


class PortfolioLayerState:
    def __init__(self, settings: PortfolioLayerSettings | None = None) -> None:
        self.settings = (settings or PortfolioLayerSettings()).sanitized()
        self._history: defaultdict[str, list[float]] = defaultdict(list)
        self._trade_counts: defaultdict[str, int] = defaultdict(int)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def classify_sleeve(self, signal: TradeSignal) -> str:
        regime = signal.regime_label.strip().lower()
        event = signal.trigger_event.strip().upper()

        if regime in {"trend", "expansion"} and event in {"BOS", "BOS_CONTINUATION"}:
            return "continuation"
        if regime == "expansion":
            return "breakout_expansion"
        if regime in {"range", "contraction"}:
            return "mean_reversion"
        return "liquidity_reversal"

    def _global_expectancy(self) -> float:
        joined: list[float] = []
        for values in self._history.values():
            joined.extend(values[-self.settings.learning_window :])
        return mean(joined) if joined else 0.0

    def _sleeve_concentration(self, sleeve: str) -> float:
        total = sum(self._trade_counts.values())
        if total <= 0:
            return 0.0
        return float(self._trade_counts.get(sleeve, 0)) / float(total)

    def decide(self, signal: TradeSignal) -> PortfolioDecision:
        sleeve = self.classify_sleeve(signal)
        if not self.settings.enabled:
            return PortfolioDecision(
                sleeve=sleeve,
                multiplier=1.0,
                applied=False,
                reason="disabled",
            )

        sleeve_history = self._history.get(sleeve, [])
        if len(sleeve_history) < self.settings.min_trades_per_sleeve:
            return PortfolioDecision(
                sleeve=sleeve,
                multiplier=1.0,
                applied=self.settings.mode == "apply",
                reason="insufficient_history",
            )

        windowed = sleeve_history[-self.settings.learning_window :]
        sleeve_exp = mean(windowed) if windowed else 0.0
        global_exp = self._global_expectancy()
        edge_delta = self._clamp(sleeve_exp - global_exp, -0.50, 0.50)

        concentration = self._sleeve_concentration(sleeve)
        concentration_penalty = 0.0
        if concentration > self.settings.max_sleeve_concentration:
            concentration_penalty = min(0.35, concentration - self.settings.max_sleeve_concentration)

        multiplier = 1.0 + (edge_delta * 0.80) - concentration_penalty
        multiplier = self._clamp(multiplier, self.settings.min_multiplier, self.settings.max_multiplier)
        return PortfolioDecision(
            sleeve=sleeve,
            multiplier=multiplier,
            applied=self.settings.mode == "apply",
            reason="applied" if self.settings.mode == "apply" else "analysis_only",
        )

    def register(self, sleeve: str, r_multiple: float) -> None:
        normalized = (sleeve or "unknown").strip().lower()
        self._trade_counts[normalized] += 1
        self._history[normalized].append(float(r_multiple))
