from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from backtest.engine import BacktestTrade
from backtest.trade_cache import result_from_payload


@dataclass(frozen=True)
class TradeRun:
    scenario: str
    stress: str
    report_path: str
    cache_path: str
    trades: list[BacktestTrade]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float) -> float:
    return round(value * 100.0, 4)


def trade_key(trade: BacktestTrade) -> str:
    return "|".join(
        [
            trade.pair.upper(),
            trade.signal_time.isoformat(),
            trade.side.upper(),
            str(trade.entry_index),
        ]
    )


def trigger_bucket(value: int) -> str:
    strength = int(value or 0)
    if strength < 8:
        return "<8"
    if strength < 10:
        return "8-9"
    if strength < 12:
        return "10-11"
    if strength < 14:
        return "12-13"
    if strength < 16:
        return "14-15"
    return "16+"


def group_value(trade: BacktestTrade, field: str) -> str:
    if field == "trigger_bucket":
        return trigger_bucket(trade.trigger_strength)
    value = getattr(trade, field, None)
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    return text.upper() if text else "UNKNOWN"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_run(report: dict[str, object], scenario: str, stress: str) -> dict[str, object]:
    for run in report.get("runs", []):
        if not isinstance(run, dict):
            continue
        if str(run.get("scenario")) == scenario and str(run.get("stress")) == stress:
            return run
    raise ValueError(f"Run not found: scenario={scenario} stress={stress}")


def cache_path_from_run(run: dict[str, object]) -> Path:
    payload = run.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("Run payload is missing")
    cache = payload.get("trade_cache", {})
    if not isinstance(cache, dict):
        raise ValueError("Run trade_cache payload is missing")
    path = cache.get("path")
    if not path:
        raise ValueError("Run trade_cache.path is missing")
    return Path(str(path))


def load_trade_run(report_path: Path, scenario: str, stress: str) -> TradeRun:
    report = read_json(report_path)
    run = find_run(report, scenario, stress)
    cache_path = cache_path_from_run(run)
    payload = read_json(cache_path)
    result = result_from_payload(payload)
    return TradeRun(
        scenario=scenario,
        stress=stress,
        report_path=str(report_path),
        cache_path=str(cache_path),
        trades=result.trades,
    )


def trade_maps(trades: Iterable[BacktestTrade]) -> tuple[dict[str, BacktestTrade], dict[str, int]]:
    items: dict[str, BacktestTrade] = {}
    counts: Counter[str] = Counter()
    for trade in trades:
        key = trade_key(trade)
        counts[key] += 1
        if counts[key] > 1:
            key = f"{key}|dup{counts[key]}"
        items[key] = trade
    duplicates = {key: count for key, count in counts.items() if count > 1}
    return items, duplicates


def stats_for_r(values: Iterable[float], risk_per_trade: float) -> dict[str, object]:
    items = [float(value) for value in values]
    wins = [value for value in items if value > 0]
    losses = [value for value in items if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(items),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(items), 6) if items else 0.0,
        "avg_r": round(mean(items), 6) if items else 0.0,
        "total_r": round(sum(items), 6),
        "net_pnl_usd": round(sum(items) * risk_per_trade, 2),
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
    }


def summarize_trades(trades: Iterable[BacktestTrade], risk_per_trade: float) -> dict[str, object]:
    items = list(trades)
    return {
        **stats_for_r((trade.r_multiple for trade in items), risk_per_trade),
        "spread_cost_r": round(sum(trade.spread_cost_r for trade in items), 6),
        "slippage_cost_r": round(sum(trade.slippage_cost_r for trade in items), 6),
        "delay_cost_r": round(sum(trade.execution_delay_cost_r for trade in items), 6),
        "avg_spread_pips": round(mean([trade.spread_pips for trade in items]), 6) if items else 0.0,
        "avg_slippage_pips": round(mean([trade.slippage_pips for trade in items]), 6) if items else 0.0,
    }


def group_summary(trades: Iterable[BacktestTrade], field: str, risk_per_trade: float) -> list[dict[str, object]]:
    grouped: dict[str, list[BacktestTrade]] = defaultdict(list)
    for trade in trades:
        grouped[group_value(trade, field)].append(trade)
    rows: list[dict[str, object]] = []
    for key, items in grouped.items():
        rows.append(
            {
                "group": key,
                **summarize_trades(items, risk_per_trade),
            }
        )
    rows.sort(key=lambda row: (safe_float(row.get("total_r")), safe_float(row.get("trades"))), reverse=True)
    return rows


def contribution_groups(
    trades: Iterable[BacktestTrade],
    *,
    field: str,
    risk_per_trade: float,
    sign: float,
) -> list[dict[str, object]]:
    rows = group_summary(trades, field, risk_per_trade)
    for row in rows:
        total_r = safe_float(row.get("total_r"))
        row["delta_r"] = round(total_r * sign, 6)
        row["delta_usd"] = round(total_r * sign * risk_per_trade, 2)
    rows.sort(key=lambda row: abs(safe_float(row.get("delta_r"))), reverse=True)
    return rows


def common_delta_groups(
    common: Iterable[tuple[BacktestTrade, BacktestTrade]],
    *,
    field: str,
    risk_per_trade: float,
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[BacktestTrade, BacktestTrade]]] = defaultdict(list)
    for base_trade, candidate_trade in common:
        grouped[group_value(candidate_trade, field)].append((base_trade, candidate_trade))
    rows: list[dict[str, object]] = []
    for key, pairs in grouped.items():
        deltas = [candidate.r_multiple - base.r_multiple for base, candidate in pairs]
        rows.append(
            {
                "group": key,
                "trades": len(pairs),
                "delta_r": round(sum(deltas), 6),
                "delta_usd": round(sum(deltas) * risk_per_trade, 2),
                "avg_delta_r": round(mean(deltas), 6) if deltas else 0.0,
                "base_total_r": round(sum(base.r_multiple for base, _ in pairs), 6),
                "candidate_total_r": round(sum(candidate.r_multiple for _, candidate in pairs), 6),
            }
        )
    rows.sort(key=lambda row: abs(safe_float(row.get("delta_r"))), reverse=True)
    return rows


def trade_digest(trade: BacktestTrade) -> dict[str, object]:
    return {
        "pair": trade.pair,
        "side": trade.side,
        "signal_time": trade.signal_time.isoformat(),
        "entry_index": trade.entry_index,
        "r_multiple": round(trade.r_multiple, 6),
        "raw_r_multiple": round(trade.raw_r_multiple, 6),
        "exit_reason": trade.exit_reason,
        "entry_mode": trade.entry_mode,
        "entry_source": trade.entry_source,
        "regime_label": trade.regime_label,
        "portfolio_sleeve": trade.portfolio_sleeve,
        "trigger_strength": trade.trigger_strength,
        "trigger_bucket": trigger_bucket(trade.trigger_strength),
        "trigger_event": trade.trigger_event,
        "structure_event": trade.structure_event,
        "score": trade.score,
        "bars_held": trade.bars_held,
        "spread_cost_r": round(trade.spread_cost_r, 6),
        "slippage_cost_r": round(trade.slippage_cost_r, 6),
        "delay_cost_r": round(trade.execution_delay_cost_r, 6),
    }


def top_common_deltas(common: Iterable[tuple[BacktestTrade, BacktestTrade]], limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for base_trade, candidate_trade in common:
        delta = candidate_trade.r_multiple - base_trade.r_multiple
        rows.append(
            {
                "delta_r": round(delta, 6),
                "base": trade_digest(base_trade),
                "candidate": trade_digest(candidate_trade),
            }
        )
    rows.sort(key=lambda row: abs(safe_float(row.get("delta_r"))), reverse=True)
    return rows[:limit]


def compare_runs(
    base: TradeRun,
    candidate: TradeRun,
    *,
    risk_per_trade: float,
    top_n: int,
) -> dict[str, object]:
    base_map, base_duplicates = trade_maps(base.trades)
    candidate_map, candidate_duplicates = trade_maps(candidate.trades)
    base_keys = set(base_map)
    candidate_keys = set(candidate_map)
    common_keys = sorted(base_keys & candidate_keys)
    base_only_keys = sorted(base_keys - candidate_keys)
    candidate_only_keys = sorted(candidate_keys - base_keys)
    common_pairs = [(base_map[key], candidate_map[key]) for key in common_keys]
    base_only = [base_map[key] for key in base_only_keys]
    candidate_only = [candidate_map[key] for key in candidate_only_keys]

    base_total = sum(trade.r_multiple for trade in base.trades)
    candidate_total = sum(trade.r_multiple for trade in candidate.trades)
    common_delta = sum(candidate.r_multiple - base_trade.r_multiple for base_trade, candidate in common_pairs)
    removed_delta = -sum(trade.r_multiple for trade in base_only)
    added_delta = sum(trade.r_multiple for trade in candidate_only)
    total_delta = candidate_total - base_total

    group_fields = [
        "pair",
        "regime_label",
        "entry_source",
        "entry_mode",
        "exit_reason",
        "portfolio_sleeve",
        "trigger_bucket",
    ]
    return {
        "base": {
            "scenario": base.scenario,
            "stress": base.stress,
            "cache_path": base.cache_path,
            "summary": summarize_trades(base.trades, risk_per_trade),
        },
        "candidate": {
            "scenario": candidate.scenario,
            "stress": candidate.stress,
            "cache_path": candidate.cache_path,
            "summary": summarize_trades(candidate.trades, risk_per_trade),
        },
        "overlap": {
            "common_trades": len(common_pairs),
            "base_only_trades": len(base_only),
            "candidate_only_trades": len(candidate_only),
            "base_duplicate_keys": base_duplicates,
            "candidate_duplicate_keys": candidate_duplicates,
        },
        "delta": {
            "total_delta_r": round(total_delta, 6),
            "total_delta_usd": round(total_delta * risk_per_trade, 2),
            "common_delta_r": round(common_delta, 6),
            "common_delta_usd": round(common_delta * risk_per_trade, 2),
            "removed_delta_r": round(removed_delta, 6),
            "removed_delta_usd": round(removed_delta * risk_per_trade, 2),
            "added_delta_r": round(added_delta, 6),
            "added_delta_usd": round(added_delta * risk_per_trade, 2),
        },
        "base_only": {
            "summary": summarize_trades(base_only, risk_per_trade),
            "by_group": {field: contribution_groups(base_only, field=field, risk_per_trade=risk_per_trade, sign=-1.0) for field in group_fields},
            "top": [trade_digest(trade) for trade in sorted(base_only, key=lambda item: abs(item.r_multiple), reverse=True)[:top_n]],
        },
        "candidate_only": {
            "summary": summarize_trades(candidate_only, risk_per_trade),
            "by_group": {field: contribution_groups(candidate_only, field=field, risk_per_trade=risk_per_trade, sign=1.0) for field in group_fields},
            "top": [trade_digest(trade) for trade in sorted(candidate_only, key=lambda item: abs(item.r_multiple), reverse=True)[:top_n]],
        },
        "common": {
            "delta_by_group": {field: common_delta_groups(common_pairs, field=field, risk_per_trade=risk_per_trade) for field in group_fields},
            "top_deltas": top_common_deltas(common_pairs, top_n),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Phase 3 trade lists between two hypothesis runs.")
    parser.add_argument("--base-report", default="reports/phase3_hypothesis_calibration_combined.json")
    parser.add_argument("--candidate-report", default="reports/phase3_hypothesis_soft_fallback_calibration.json")
    parser.add_argument("--base-scenario", default="timeout_fast")
    parser.add_argument("--candidate-scenario", default="timeout_fast_fallback_8")
    parser.add_argument("--stress-presets", default="moderate,harsh")
    parser.add_argument("--risk-per-trade", type=float, default=50.0)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--output", default="reports/phase3_trade_diff_timeout_fast_vs_soft_fallback.json")
    return parser


def print_summary(stress: str, payload: dict[str, object]) -> None:
    delta = payload.get("delta", {}) if isinstance(payload.get("delta"), dict) else {}
    overlap = payload.get("overlap", {}) if isinstance(payload.get("overlap"), dict) else {}
    base = payload.get("base", {}) if isinstance(payload.get("base"), dict) else {}
    candidate = payload.get("candidate", {}) if isinstance(payload.get("candidate"), dict) else {}
    base_summary = base.get("summary", {}) if isinstance(base.get("summary"), dict) else {}
    candidate_summary = candidate.get("summary", {}) if isinstance(candidate.get("summary"), dict) else {}
    print(
        "{stress:8s} base={base_trades:>2} cand={cand_trades:>2} common={common:>2} "
        "base_only={base_only:>2} cand_only={cand_only:>2} delta={delta_r:+.3f}R (${delta_usd:+.2f})".format(
            stress=stress,
            base_trades=int(safe_float(base_summary.get("trades"))),
            cand_trades=int(safe_float(candidate_summary.get("trades"))),
            common=int(safe_float(overlap.get("common_trades"))),
            base_only=int(safe_float(overlap.get("base_only_trades"))),
            cand_only=int(safe_float(overlap.get("candidate_only_trades"))),
            delta_r=safe_float(delta.get("total_delta_r")),
            delta_usd=safe_float(delta.get("total_delta_usd")),
        ),
        flush=True,
    )
    print(
        "  common={common:+.3f}R removed={removed:+.3f}R added={added:+.3f}R".format(
            common=safe_float(delta.get("common_delta_r")),
            removed=safe_float(delta.get("removed_delta_r")),
            added=safe_float(delta.get("added_delta_r")),
        ),
        flush=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    base_report = Path(args.base_report)
    candidate_report = Path(args.candidate_report)
    stresses = [item.strip() for item in str(args.stress_presets).split(",") if item.strip()]
    output: dict[str, object] = {
        "runner": "research.phase3_trade_diff_analysis",
        "settings": {
            "base_report": str(base_report),
            "candidate_report": str(candidate_report),
            "base_scenario": args.base_scenario,
            "candidate_scenario": args.candidate_scenario,
            "stress_presets": stresses,
            "risk_per_trade": float(args.risk_per_trade),
        },
        "comparisons": {},
    }
    print(
        "Phase 3 trade diff | base={base} candidate={candidate} stress={stress}".format(
            base=args.base_scenario,
            candidate=args.candidate_scenario,
            stress=",".join(stresses),
        ),
        flush=True,
    )
    for stress in stresses:
        base = load_trade_run(base_report, args.base_scenario, stress)
        candidate = load_trade_run(candidate_report, args.candidate_scenario, stress)
        comparison = compare_runs(base, candidate, risk_per_trade=float(args.risk_per_trade), top_n=max(1, int(args.top_n)))
        output["comparisons"][stress] = comparison  # type: ignore[index]
        print_summary(stress, comparison)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(f"Saved trade diff report: {path}", flush=True)


if __name__ == "__main__":
    main()
