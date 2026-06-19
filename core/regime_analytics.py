from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
import json


def _safe_label(value: object) -> str:
    text = str(value or "").strip().upper()
    return text if text else "UNKNOWN"


def _profit_factor(values: list[float]) -> float:
    gross_profit = sum(item for item in values if item > 0)
    gross_loss = abs(sum(item for item in values if item < 0))
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _expectancy(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_r": 0.0,
            "sharpe_r": 0.0,
        }
    wins = [item for item in values if item > 0]
    losses = [abs(item) for item in values if item < 0]
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    win_rate = len(wins) / len(values)
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)
    volatility = pstdev(values) if len(values) > 1 else 0.0
    return {
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy_r": (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss),
        "sharpe_r": (mean(values) / volatility) if volatility > 0 else 0.0,
    }


def analyze_regime_performance(
    *,
    trades: Iterable[object],
    regime_evaluations: Mapping[str, int] | None = None,
    regime_acceptances: Mapping[str, int] | None = None,
) -> dict[str, object]:
    trade_buckets: defaultdict[str, list[float]] = defaultdict(list)
    for trade in trades:
        label = _safe_label(getattr(trade, "regime_label", None))
        trade_buckets[label].append(float(getattr(trade, "r_multiple", 0.0)))

    eval_counts = Counter({_safe_label(key): int(value) for key, value in (regime_evaluations or {}).items()})
    acceptance_counts = Counter({_safe_label(key): int(value) for key, value in (regime_acceptances or {}).items()})

    all_regimes = sorted(set(trade_buckets) | set(eval_counts) | set(acceptance_counts))
    rows: dict[str, dict[str, float | int | None]] = {}
    for regime in all_regimes:
        pnl = trade_buckets.get(regime, [])
        trades_count = len(pnl)
        wins = sum(1 for value in pnl if value > 0)
        win_rate = (wins / trades_count) if trades_count else 0.0
        avg_r = mean(pnl) if pnl else 0.0
        pf = _profit_factor(pnl)
        expectancy = _expectancy(pnl)
        evaluations = int(eval_counts.get(regime, 0))
        accepted = int(acceptance_counts.get(regime, trades_count))
        acceptance_rate = (accepted / evaluations) if evaluations > 0 else None

        rows[regime] = {
            "signal_count": trades_count,
            "evaluations": evaluations,
            "accepted_signals": accepted,
            "acceptance_rate": round(float(acceptance_rate), 6) if acceptance_rate is not None else None,
            "win_rate": round(float(win_rate), 6),
            "profit_factor": None if pf == float("inf") else round(float(pf), 6),
            "avg_r": round(float(avg_r), 6),
            "avg_win_r": round(float(expectancy["avg_win_r"]), 6),
            "avg_loss_r": round(float(expectancy["avg_loss_r"]), 6),
            "payoff_ratio": None if expectancy["payoff_ratio"] == float("inf") else round(float(expectancy["payoff_ratio"]), 6),
            "expectancy_r": round(float(expectancy["expectancy_r"]), 6),
            "sharpe_r": round(float(expectancy["sharpe_r"]), 6),
        }

    total_evaluations = int(sum(eval_counts.values()))
    total_accepted = int(sum(acceptance_counts.values())) if acceptance_counts else int(sum(len(v) for v in trade_buckets.values()))
    total_trades = int(sum(len(v) for v in trade_buckets.values()))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "regime_count": len(rows),
            "total_trades": total_trades,
            "total_evaluations": total_evaluations,
            "total_accepted_signals": total_accepted,
            "global_acceptance_rate": round(total_accepted / total_evaluations, 6) if total_evaluations > 0 else None,
        },
        "regimes": rows,
    }


def analyze_regime_performance_from_run(run_result: object) -> dict[str, object]:
    reports = getattr(run_result, "pair_reports", [])
    evaluations: Counter[str] = Counter()
    acceptances: Counter[str] = Counter()
    for report in reports:
        evaluations.update(getattr(report, "regime_evaluations", {}) or {})
        acceptances.update(getattr(report, "regime_acceptances", {}) or {})

    return analyze_regime_performance(
        trades=getattr(run_result, "trades", []),
        regime_evaluations=dict(evaluations),
        regime_acceptances=dict(acceptances),
    )


def export_regime_report(payload: Mapping[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    return target
