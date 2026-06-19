from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp
from statistics import mean, pstdev
from typing import Mapping


FEATURE_KEYS = (
    "htf",
    "regime",
    "trigger",
    "liquidity",
    "pd",
    "session",
    "news",
    "shadow_fvg",
    "shadow_ob",
    "shadow_mitigation",
    "shadow_smt",
)

LEGACY_FEATURE_MAX = {
    "htf": 20.0,
    "regime": 15.0,
    "trigger": 15.0,
    "liquidity": 20.0,
    "pd": 15.0,
    "session": 12.0,
    "news": 5.0,
    "shadow_fvg": 7.0,
    "shadow_ob": 7.0,
    "shadow_mitigation": 6.0,
    "shadow_smt": 8.0,
}
LEGACY_SCORE_CAPACITY = sum(LEGACY_FEATURE_MAX.values())

DEFAULT_ADAPTIVE_WEIGHTS: dict[str, dict[str, float]] = {
    "trend": {
        "htf": 18.0,
        "regime": 16.0,
        "trigger": 14.0,
        "liquidity": 13.0,
        "pd": 11.0,
        "session": 8.0,
        "news": 6.0,
        "shadow_fvg": 6.0,
        "shadow_ob": 4.0,
        "shadow_mitigation": 2.0,
        "shadow_smt": 2.0,
    },
    "range": {
        "htf": 10.0,
        "regime": 8.0,
        "trigger": 14.0,
        "liquidity": 16.0,
        "pd": 14.0,
        "session": 9.0,
        "news": 6.0,
        "shadow_fvg": 8.0,
        "shadow_ob": 7.0,
        "shadow_mitigation": 6.0,
        "shadow_smt": 2.0,
    },
    "expansion": {
        "htf": 16.0,
        "regime": 15.0,
        "trigger": 14.0,
        "liquidity": 17.0,
        "pd": 10.0,
        "session": 8.0,
        "news": 6.0,
        "shadow_fvg": 5.0,
        "shadow_ob": 4.0,
        "shadow_mitigation": 2.0,
        "shadow_smt": 3.0,
    },
    "contraction": {
        "htf": 12.0,
        "regime": 10.0,
        "trigger": 10.0,
        "liquidity": 12.0,
        "pd": 11.0,
        "session": 11.0,
        "news": 7.0,
        "shadow_fvg": 9.0,
        "shadow_ob": 8.0,
        "shadow_mitigation": 7.0,
        "shadow_smt": 3.0,
    },
    "neutral": {
        "htf": 14.0,
        "regime": 11.0,
        "trigger": 12.0,
        "liquidity": 13.0,
        "pd": 12.0,
        "session": 9.0,
        "news": 6.0,
        "shadow_fvg": 8.0,
        "shadow_ob": 6.0,
        "shadow_mitigation": 5.0,
        "shadow_smt": 4.0,
    },
}

EFFECTIVENESS_V1_WEIGHTS: dict[str, dict[str, float]] = {
    "trend": {
        "htf": 26.0,
        "regime": 3.0,
        "trigger": 12.0,
        "liquidity": 22.0,
        "pd": 2.0,
        "session": 16.0,
        "news": 5.0,
        "shadow_fvg": 24.0,
        "shadow_ob": 10.0,
        "shadow_mitigation": 0.0,
        "shadow_smt": 0.0,
    },
    "range": {
        "htf": 20.0,
        "regime": 2.0,
        "trigger": 10.0,
        "liquidity": 28.0,
        "pd": 2.0,
        "session": 18.0,
        "news": 5.0,
        "shadow_fvg": 24.0,
        "shadow_ob": 12.0,
        "shadow_mitigation": 0.0,
        "shadow_smt": 0.0,
    },
    "expansion": {
        "htf": 24.0,
        "regime": 2.0,
        "trigger": 12.0,
        "liquidity": 28.0,
        "pd": 1.0,
        "session": 14.0,
        "news": 5.0,
        "shadow_fvg": 24.0,
        "shadow_ob": 8.0,
        "shadow_mitigation": 0.0,
        "shadow_smt": 0.0,
    },
    "contraction": {
        "htf": 20.0,
        "regime": 2.0,
        "trigger": 8.0,
        "liquidity": 24.0,
        "pd": 1.0,
        "session": 20.0,
        "news": 5.0,
        "shadow_fvg": 26.0,
        "shadow_ob": 14.0,
        "shadow_mitigation": 0.0,
        "shadow_smt": 0.0,
    },
    "neutral": {
        "htf": 22.0,
        "regime": 2.0,
        "trigger": 10.0,
        "liquidity": 24.0,
        "pd": 1.0,
        "session": 18.0,
        "news": 5.0,
        "shadow_fvg": 24.0,
        "shadow_ob": 12.0,
        "shadow_mitigation": 0.0,
        "shadow_smt": 0.0,
    },
}

WEIGHT_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "default": DEFAULT_ADAPTIVE_WEIGHTS,
    "effectiveness_v1": EFFECTIVENESS_V1_WEIGHTS,
}


@dataclass(frozen=True)
class AdaptiveWeightSettings:
    enabled: bool = False
    regime_weights: dict[str, dict[str, float]] | None = None
    preset: str = "default"


@dataclass(frozen=True)
class ScoreNormalizationSettings:
    enabled: bool = False
    method: str = "minmax"
    window: int = 200
    scale_factor: float = 1.0
    backtest_only: bool = True
    allow_live: bool = False


@dataclass(frozen=True)
class DynamicThresholdSettings:
    enabled: bool = False
    percentile: float = 80.0
    rolling_window: int = 200
    apply_threshold: bool = False
    backtest_only: bool = True
    allow_live: bool = False


def _base_regime_weights(preset: str | None) -> dict[str, dict[str, float]]:
    key = (preset or "default").strip().lower()
    return WEIGHT_PRESETS.get(key, DEFAULT_ADAPTIVE_WEIGHTS)


def _sanitize_regime_map(
    weights: Mapping[str, Mapping[str, float]] | None,
    *,
    preset: str | None = None,
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    base_preset = _base_regime_weights(preset)
    source = weights or {}
    for regime in ("trend", "range", "expansion", "contraction", "neutral"):
        base = dict(base_preset[regime])
        override = source.get(regime, {})
        for key in FEATURE_KEYS:
            value = override.get(key, base[key])
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = base[key]
            base[key] = max(0.0, parsed)
        total = sum(base.values())
        if total <= 0:
            base = dict(base_preset[regime])
            total = sum(base.values())
        scale = LEGACY_SCORE_CAPACITY / total
        merged[regime] = {key: round(base[key] * scale, 6) for key in FEATURE_KEYS}
    return merged


def resolve_regime_weights(
    regime_label: str,
    settings: AdaptiveWeightSettings | None,
) -> tuple[str, dict[str, float]]:
    if settings is None or not settings.enabled:
        return "legacy_fixed", {}

    weights = _sanitize_regime_map(settings.regime_weights, preset=settings.preset)
    label = regime_label.lower().strip()
    if label not in weights:
        label = "neutral"
    return label, weights[label]


def apply_regime_weights(
    components: Mapping[str, int],
    regime_label: str,
    settings: AdaptiveWeightSettings | None,
) -> tuple[dict[str, int], int, dict[str, object]]:
    profile_name, weights = resolve_regime_weights(regime_label, settings)
    if not weights:
        legacy_total = max(0, min(100, int(round(sum(float(components.get(key, 0)) for key in FEATURE_KEYS)))))
        payload = {key: int(components.get(key, 0)) for key in FEATURE_KEYS}
        return payload, legacy_total, {
            "enabled": False,
            "profile": profile_name,
            "weights": {},
            "raw_total": legacy_total,
            "weighted_total": legacy_total,
        }

    weighted: dict[str, int] = {}
    for key in FEATURE_KEYS:
        raw_value = max(0.0, float(components.get(key, 0)))
        feature_max = max(1e-9, float(LEGACY_FEATURE_MAX.get(key, 1.0)))
        normalized = min(1.0, raw_value / feature_max)
        weighted[key] = int(round(normalized * weights[key]))

    total = max(0, min(100, int(round(sum(weighted.values())))))
    raw_total = max(0, min(100, int(round(sum(float(components.get(key, 0)) for key in FEATURE_KEYS)))))
    return weighted, total, {
        "enabled": True,
        "profile": profile_name,
        "preset": settings.preset if settings is not None else "default",
        "weights": {key: round(float(weights[key]), 6) for key in FEATURE_KEYS},
        "raw_total": raw_total,
        "weighted_total": total,
    }


class ScoreNormalizer:
    def __init__(self, settings: ScoreNormalizationSettings) -> None:
        self.settings = settings
        self._history: deque[float] = deque(maxlen=max(10, settings.window))

    def reset(self) -> None:
        self._history.clear()

    def is_active(self, runtime_mode: str) -> bool:
        if not self.settings.enabled:
            return False
        if not self.settings.backtest_only:
            return True
        if runtime_mode.lower() == "backtest":
            return True
        return self.settings.allow_live

    def transform(self, score: int, runtime_mode: str) -> tuple[int, dict[str, float | str | bool]]:
        raw_score = max(0.0, min(100.0, float(score)))
        if not self.is_active(runtime_mode):
            return int(round(raw_score)), {
                "enabled": False,
                "method": self.settings.method,
                "raw_score": raw_score,
                "normalized_score": raw_score,
            }

        values = list(self._history)
        method = self.settings.method.lower().strip()
        normalized = raw_score
        if method == "minmax":
            normalized = self._minmax(raw_score, values)
        elif method == "zscore":
            normalized = self._zscore(raw_score, values)
        elif method == "scale":
            normalized = raw_score * max(0.0, self.settings.scale_factor)

        normalized = max(0.0, min(100.0, normalized))
        self._history.append(raw_score)
        return int(round(normalized)), {
            "enabled": True,
            "method": method,
            "raw_score": round(raw_score, 6),
            "normalized_score": round(normalized, 6),
            "window_size": len(values),
        }

    @staticmethod
    def _minmax(value: float, values: list[float]) -> float:
        if len(values) < 2:
            return value
        low = min(values)
        high = max(values)
        if high <= low:
            return value
        return ((value - low) / (high - low)) * 100.0

    @staticmethod
    def _zscore(value: float, values: list[float]) -> float:
        if len(values) < 2:
            return value
        mu = mean(values)
        sigma = pstdev(values)
        if sigma <= 1e-9:
            return value
        z = (value - mu) / sigma
        logistic = 1.0 / (1.0 + exp(-z))
        return logistic * 100.0


class DynamicThresholdTracker:
    def __init__(self, settings: DynamicThresholdSettings) -> None:
        self.settings = settings
        self._history: deque[float] = deque(maxlen=max(10, settings.rolling_window))

    def reset(self) -> None:
        self._history.clear()

    def is_active(self, runtime_mode: str) -> bool:
        if not self.settings.enabled:
            return False
        if not self.settings.backtest_only:
            return True
        if runtime_mode.lower() == "backtest":
            return True
        return self.settings.allow_live

    def recommended_threshold(self, runtime_mode: str) -> int | None:
        if not self.is_active(runtime_mode):
            return None
        values = list(self._history)
        if len(values) < 5:
            return None
        percentile = max(0.0, min(100.0, float(self.settings.percentile)))
        quantile = percentile / 100.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return int(round(sorted_values[0]))

        position = quantile * (len(sorted_values) - 1)
        low_idx = int(position)
        high_idx = min(low_idx + 1, len(sorted_values) - 1)
        weight = position - low_idx
        interpolated = sorted_values[low_idx] + (sorted_values[high_idx] - sorted_values[low_idx]) * weight
        return max(0, min(100, int(round(interpolated))))

    def observe(self, score: int, runtime_mode: str) -> None:
        if not self.is_active(runtime_mode):
            return
        clamped = max(0.0, min(100.0, float(score)))
        self._history.append(clamped)
