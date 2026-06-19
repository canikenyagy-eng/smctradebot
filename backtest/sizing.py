from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from backtest.risk import EquityProtectionState
from core.signal_engine import TradeSignal


@dataclass(frozen=True)
class AdaptiveSizingSettings:
    enabled: bool = False
    min_multiplier: float = 0.40
    max_multiplier: float = 1.50
    confidence_floor_score: int = 65
    confidence_ceiling_score: int = 90
    confidence_weight: float = 0.45
    regime_weight: float = 0.30
    volatility_weight: float = 0.15
    drawdown_weight: float = 0.10
    regime_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "trend": 1.15,
            "expansion": 1.00,
            "range": 0.90,
            "contraction": 0.75,
            "neutral": 0.85,
        }
    )

    def sanitized(self) -> "AdaptiveSizingSettings":
        floor = max(0, min(100, int(self.confidence_floor_score)))
        ceiling = max(floor + 1, min(100, int(self.confidence_ceiling_score)))
        min_mult = max(0.05, float(self.min_multiplier))
        max_mult = max(min_mult, float(self.max_multiplier))
        weights = [
            max(0.0, float(self.confidence_weight)),
            max(0.0, float(self.regime_weight)),
            max(0.0, float(self.volatility_weight)),
            max(0.0, float(self.drawdown_weight)),
        ]
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0, 0.0, 0.0, 0.0]
            total_weight = 1.0
        normalized = [value / total_weight for value in weights]
        regime_map = {
            str(key).strip().lower(): max(min_mult, min(max_mult, float(value)))
            for key, value in (self.regime_multipliers or {}).items()
        }
        if "neutral" not in regime_map:
            regime_map["neutral"] = 1.0
        return AdaptiveSizingSettings(
            enabled=bool(self.enabled),
            min_multiplier=min_mult,
            max_multiplier=max_mult,
            confidence_floor_score=floor,
            confidence_ceiling_score=ceiling,
            confidence_weight=normalized[0],
            regime_weight=normalized[1],
            volatility_weight=normalized[2],
            drawdown_weight=normalized[3],
            regime_multipliers=regime_map,
        )


@dataclass(frozen=True)
class SizingDecision:
    multiplier: float
    confidence_component: float
    regime_component: float
    volatility_component: float
    drawdown_component: float

    def as_dict(self) -> dict[str, float]:
        return {
            "multiplier": round(float(self.multiplier), 6),
            "confidence_component": round(float(self.confidence_component), 6),
            "regime_component": round(float(self.regime_component), 6),
            "volatility_component": round(float(self.volatility_component), 6),
            "drawdown_component": round(float(self.drawdown_component), 6),
        }


class AdaptiveSizingEngine:
    def __init__(self, settings: AdaptiveSizingSettings | None = None) -> None:
        self.settings = (settings or AdaptiveSizingSettings()).sanitized()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _confidence_component(self, score: int) -> float:
        cfg = self.settings
        if cfg.confidence_ceiling_score <= cfg.confidence_floor_score:
            return 1.0
        normalized = (float(score) - float(cfg.confidence_floor_score)) / (
            float(cfg.confidence_ceiling_score - cfg.confidence_floor_score)
        )
        normalized = self._clamp(normalized, 0.0, 1.0)
        return cfg.min_multiplier + (cfg.max_multiplier - cfg.min_multiplier) * normalized

    def _regime_component(self, regime_label: str) -> float:
        label = (regime_label or "neutral").strip().lower()
        if label not in self.settings.regime_multipliers:
            label = "neutral"
        value = float(self.settings.regime_multipliers.get(label, 1.0))
        return self._clamp(value, self.settings.min_multiplier, self.settings.max_multiplier)

    def _volatility_component(self, volatility_ratio: float) -> float:
        ratio = max(0.1, min(3.0, float(volatility_ratio)))
        distance = abs(ratio - 1.0)
        penalty = min(0.6, distance * 0.45)
        value = 1.0 - penalty
        return self._clamp(value, self.settings.min_multiplier, self.settings.max_multiplier)

    def _drawdown_component(self, equity_state: EquityProtectionState | None) -> float:
        if equity_state is None or not equity_state.settings.enabled:
            return 1.0
        max_dd = max(1e-9, float(equity_state.settings.max_drawdown_limit))
        drawdown = float(equity_state.drawdown_r())
        ratio = self._clamp(drawdown / max_dd, 0.0, 1.0)
        value = 1.0 - (ratio * (1.0 - float(equity_state.settings.min_risk_multiplier)))
        return self._clamp(value, self.settings.min_multiplier, self.settings.max_multiplier)

    def decide(
        self,
        *,
        signal: TradeSignal,
        volatility_ratio: float,
        equity_state: EquityProtectionState | None,
    ) -> SizingDecision:
        if not self.settings.enabled:
            return SizingDecision(
                multiplier=1.0,
                confidence_component=1.0,
                regime_component=1.0,
                volatility_component=1.0,
                drawdown_component=1.0,
            )

        conf = self._confidence_component(signal.score)
        regime = self._regime_component(signal.regime_label)
        vol = self._volatility_component(volatility_ratio)
        dd = self._drawdown_component(equity_state)

        values = [conf, regime, vol, dd]
        weights = [
            self.settings.confidence_weight,
            self.settings.regime_weight,
            self.settings.volatility_weight,
            self.settings.drawdown_weight,
        ]
        weighted = [value * weight for value, weight in zip(values, weights)]
        multiplier = sum(weighted) if weights else mean(values)
        multiplier = self._clamp(multiplier, self.settings.min_multiplier, self.settings.max_multiplier)
        return SizingDecision(
            multiplier=multiplier,
            confidence_component=conf,
            regime_component=regime,
            volatility_component=vol,
            drawdown_component=dd,
        )
