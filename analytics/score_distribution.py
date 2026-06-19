from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from statistics import mean, median
from typing import Any


def _score_value(item: Any) -> float | None:
    if isinstance(item, (int, float)):
        return float(item)
    value = getattr(item, "score", None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def analyze_scores(
    signals: Iterable[Any],
    *,
    threshold: int = 70,
    total_evaluations: int | None = None,
    accepted_count: int | None = None,
    bucket_size: int = 5,
) -> dict[str, Any]:
    values = [score for item in signals if (score := _score_value(item)) is not None]
    if bucket_size <= 0:
        bucket_size = 5

    bucket_counter: Counter[str] = Counter()
    for score in values:
        clamped = max(0, min(100, int(score)))
        lower = (clamped // bucket_size) * bucket_size
        upper = min(100, lower + bucket_size - 1)
        label = f"{lower:02d}-{upper:02d}"
        bucket_counter[label] += 1

    above_threshold = sum(1 for score in values if score >= threshold)
    count = len(values)
    score_coverage_rate = (count / total_evaluations) if total_evaluations else (1.0 if count > 0 else 0.0)
    if accepted_count is not None and total_evaluations:
        acceptance_rate = accepted_count / total_evaluations
    else:
        acceptance_rate = score_coverage_rate

    return {
        "count": count,
        "mean": round(mean(values), 4) if values else 0.0,
        "median": round(median(values), 4) if values else 0.0,
        "threshold": threshold,
        "above_threshold_count": above_threshold,
        "above_threshold_pct": round((above_threshold / count) if count else 0.0, 6),
        "accepted_count": int(accepted_count) if accepted_count is not None else None,
        "acceptance_rate": round(acceptance_rate, 6),
        "score_coverage_rate": round(score_coverage_rate, 6),
        "histogram": {key: bucket_counter[key] for key in sorted(bucket_counter)},
    }


def _categorize_reason(reason: str) -> str:
    value = reason.strip().lower()
    if "scor" in value:
        return "score_too_low"
    if "regime" in value:
        return "regime"
    if "news" in value:
        return "news"
    if "smt" in value:
        return "smt"
    if "session" in value:
        return "session"
    return "other"


def analyze_rejections(rejected_signals: Any) -> dict[str, Any]:
    categorized: Counter[str] = Counter()
    raw: Counter[str] = Counter()

    if isinstance(rejected_signals, dict):
        for key, value in rejected_signals.items():
            count = int(value)
            raw[str(key)] += count
            categorized[_categorize_reason(str(key))] += count
    else:
        for item in rejected_signals:
            if isinstance(item, str):
                reason = item
            else:
                reason = str(
                    getattr(item, "rejection_stage", None)
                    or getattr(item, "reason", None)
                    or getattr(item, "rejection_reason", None)
                    or "unknown"
                )
            raw[reason] += 1
            categorized[_categorize_reason(reason)] += 1

    total = sum(raw.values())
    pct = {
        key: round((value / total), 6) if total else 0.0
        for key, value in categorized.items()
    }
    for key in ("score_too_low", "regime", "news", "smt", "session", "other"):
        categorized.setdefault(key, 0)
        pct.setdefault(key, 0.0)

    return {
        "total_rejections": total,
        "by_category": {key: categorized[key] for key in ("score_too_low", "regime", "news", "smt", "session", "other")},
        "by_category_pct": {key: pct[key] for key in ("score_too_low", "regime", "news", "smt", "session", "other")},
        "raw_reasons": dict(sorted(raw.items())),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = max(0.0, min(1.0, percentile / 100.0))
    position = q * (len(sorted_values) - 1)
    low_idx = int(position)
    high_idx = min(low_idx + 1, len(sorted_values) - 1)
    weight = position - low_idx
    return float(sorted_values[low_idx] + (sorted_values[high_idx] - sorted_values[low_idx]) * weight)


def analyze_dynamic_threshold(
    signals: Iterable[Any],
    *,
    percentile: float = 80.0,
    rolling_window: int = 200,
    min_samples: int = 5,
) -> dict[str, Any]:
    values = [score for item in signals if (score := _score_value(item)) is not None]
    window = max(10, int(rolling_window))
    traces: list[dict[str, float | int | bool]] = []

    for index, score in enumerate(values):
        start = max(0, index - window)
        history = values[start:index]
        if len(history) < max(1, min_samples):
            continue
        threshold = _percentile(history, percentile)
        traces.append(
            {
                "index": index,
                "score": round(float(score), 6),
                "recommended_threshold": round(float(threshold), 6),
                "accepted_by_dynamic_threshold": bool(score >= threshold),
            }
        )

    thresholds = [float(item["recommended_threshold"]) for item in traces]
    accepted_count = sum(1 for item in traces if bool(item["accepted_by_dynamic_threshold"]))
    return {
        "samples": len(values),
        "rolling_window": window,
        "percentile": float(percentile),
        "trace_count": len(traces),
        "mean_recommended_threshold": round(mean(thresholds), 6) if thresholds else 0.0,
        "median_recommended_threshold": round(median(thresholds), 6) if thresholds else 0.0,
        "min_recommended_threshold": round(min(thresholds), 6) if thresholds else 0.0,
        "max_recommended_threshold": round(max(thresholds), 6) if thresholds else 0.0,
        "accepted_by_dynamic_threshold": accepted_count,
        "acceptance_rate_if_applied": round((accepted_count / len(traces)) if traces else 0.0, 6),
        "trace": traces[-200:],
    }
