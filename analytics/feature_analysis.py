from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any

import pandas as pd


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


def _fallback_breakdown(trade: Any) -> dict[str, float]:
    return {
        "htf": float(getattr(trade, "score_htf", 0.0)),
        "regime": float(getattr(trade, "score_regime", 0.0)),
        "trigger": float(getattr(trade, "score_trigger", 0.0)),
        "liquidity": float(getattr(trade, "score_liquidity", 0.0)),
        "pd": float(getattr(trade, "score_zone", 0.0)),
        "session": float(getattr(trade, "score_session", 0.0)),
        "news": float(getattr(trade, "score_news", 0.0)),
        "shadow_fvg": float(getattr(trade, "score_fvg", 0.0)),
        "shadow_ob": float(getattr(trade, "score_order_block", 0.0)),
        "shadow_mitigation": float(getattr(trade, "score_mitigation", 0.0)),
        "shadow_smt": float(getattr(trade, "score_smt", 0.0)),
    }


def _extract_breakdown(trade: Any) -> dict[str, float]:
    payload = getattr(trade, "feature_breakdown", None)
    if isinstance(payload, dict):
        data: dict[str, float] = {}
        for key in FEATURE_KEYS:
            data[key] = float(payload.get(key, 0.0))
        return data
    return _fallback_breakdown(trade)


def compute_avg_contribution(trades: Iterable[Any]) -> dict[str, float]:
    items = list(trades)
    if not items:
        return {key: 0.0 for key in FEATURE_KEYS}

    bucket: dict[str, list[float]] = {key: [] for key in FEATURE_KEYS}
    for trade in items:
        breakdown = _extract_breakdown(trade)
        for key in FEATURE_KEYS:
            bucket[key].append(float(breakdown.get(key, 0.0)))

    return {key: round(mean(values), 4) if values else 0.0 for key, values in bucket.items()}


def compute_win_vs_loss_contribution(trades: Iterable[Any]) -> dict[str, dict[str, float]]:
    winners = []
    losers = []
    for trade in trades:
        pnl = float(getattr(trade, "r_multiple", 0.0))
        if pnl > 0:
            winners.append(trade)
        elif pnl < 0:
            losers.append(trade)

    win_avg = compute_avg_contribution(winners)
    loss_avg = compute_avg_contribution(losers)

    return {
        key: {
            "winner_avg": round(win_avg.get(key, 0.0), 4),
            "loser_avg": round(loss_avg.get(key, 0.0), 4),
            "delta": round(win_avg.get(key, 0.0) - loss_avg.get(key, 0.0), 4),
        }
        for key in FEATURE_KEYS
    }


def compute_correlation_with_pnl(trades: Iterable[Any]) -> dict[str, dict[str, float | int]]:
    items = list(trades)
    if not items:
        return {key: {"correlation": 0.0, "samples": 0} for key in FEATURE_KEYS}

    rows: list[dict[str, float]] = []
    for trade in items:
        row = _extract_breakdown(trade)
        row["pnl"] = float(getattr(trade, "r_multiple", 0.0))
        rows.append(row)

    frame = pd.DataFrame(rows)
    result: dict[str, dict[str, float]] = {}
    for key in FEATURE_KEYS:
        if key not in frame.columns or frame[key].nunique(dropna=True) <= 1 or frame["pnl"].nunique(dropna=True) <= 1:
            corr = 0.0
        else:
            corr_value = frame[key].corr(frame["pnl"])
            corr = 0.0 if pd.isna(corr_value) else float(corr_value)

        result[key] = {
            "correlation": round(corr, 6),
            "samples": len(frame),
        }

    return result
