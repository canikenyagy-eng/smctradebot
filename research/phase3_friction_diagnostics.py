from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GROUP_KEYS = (
    "pair",
    "regime_label",
    "entry_mode",
    "entry_source",
    "exit_reason",
    "portfolio_sleeve",
    "side",
)


@dataclass(frozen=True)
class TradeRow:
    raw: dict[str, str]

    @property
    def r(self) -> float:
        return to_float(self.raw.get("r_multiple"))

    @property
    def pnl_usd(self) -> float:
        return to_float(self.raw.get("pnl_usd"))

    @property
    def friction_cost_r(self) -> float:
        return (
            to_float(self.raw.get("spread_cost_r"))
            + to_float(self.raw.get("slippage_cost_r"))
            + to_float(self.raw.get("execution_delay_cost_r"))
        )


def to_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    if math.isinf(parsed):
        return 999.0 if parsed > 0 else -999.0
    return parsed


def json_safe(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 999.0 if value > 0 else -999.0
        return round(value, 6)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_trades(path: Path) -> list[TradeRow]:
    if not path.exists():
        raise SystemExit(f"Missing trades file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [TradeRow(row) for row in csv.DictReader(handle)]


def profit_factor(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    gross_profit = sum(value for value in vals if value > 0)
    gross_loss = abs(sum(value for value in vals if value < 0))
    if gross_loss > 0:
        return gross_profit / gross_loss
    return float("inf") if gross_profit > 0 else 0.0


def summarize(trades: list[TradeRow]) -> dict[str, object]:
    values = [trade.r for trade in trades]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    friction = [trade.friction_cost_r for trade in trades]
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "total_r": sum(values),
        "avg_r": sum(values) / len(values) if values else 0.0,
        "loss_r": sum(losses),
        "avg_loss_r": sum(losses) / len(losses) if losses else 0.0,
        "profit_factor": profit_factor(values),
        "net_pnl_usd": sum(trade.pnl_usd for trade in trades),
        "avg_spread_pips": mean_field(trades, "spread_pips"),
        "avg_slippage_pips": mean_field(trades, "slippage_pips"),
        "avg_execution_delay_bars": mean_field(trades, "execution_delay_bars"),
        "total_spread_cost_r": sum(to_float(trade.raw.get("spread_cost_r")) for trade in trades),
        "total_slippage_cost_r": sum(to_float(trade.raw.get("slippage_cost_r")) for trade in trades),
        "total_delay_cost_r": sum(to_float(trade.raw.get("execution_delay_cost_r")) for trade in trades),
        "total_friction_cost_r": sum(friction),
        "avg_friction_cost_r": sum(friction) / len(friction) if friction else 0.0,
        "timeout_trades": sum(1 for trade in trades if trade.raw.get("exit_reason") == "timeout"),
        "stop_loss_trades": sum(1 for trade in trades if trade.raw.get("exit_reason") == "stop_loss"),
        "market_entries": sum(1 for trade in trades if trade.raw.get("entry_mode") == "MARKET"),
        "limit_entries": sum(1 for trade in trades if trade.raw.get("entry_mode") == "MITIGATION_LIMIT"),
    }


def mean_field(trades: list[TradeRow], key: str) -> float:
    values = [to_float(trade.raw.get(key)) for trade in trades]
    return sum(values) / len(values) if values else 0.0


def group_summary(trades: list[TradeRow], key: str, *, limit: int = 20) -> list[dict[str, object]]:
    grouped: dict[str, list[TradeRow]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.raw.get(key) or "UNKNOWN")].append(trade)
    rows = [
        {
            "key": group_key,
            **summarize(items),
        }
        for group_key, items in grouped.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            to_float(row.get("total_r")),
            to_float(row.get("avg_r")),
            -to_float(row.get("trades")),
        ),
    )[:limit]


def losing_group_summary(trades: list[TradeRow], key: str, *, limit: int = 20) -> list[dict[str, object]]:
    return group_summary([trade for trade in trades if trade.r < 0], key, limit=limit)


def worst_trades(trades: list[TradeRow], *, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trade in sorted(trades, key=lambda item: item.r)[:limit]:
        raw = trade.raw
        rows.append(
            {
                "pair": raw.get("pair"),
                "side": raw.get("side"),
                "signal_time": raw.get("signal_time"),
                "entry_time": raw.get("entry_time"),
                "r_multiple": trade.r,
                "pnl_usd": trade.pnl_usd,
                "friction_cost_r": trade.friction_cost_r,
                "spread_pips": to_float(raw.get("spread_pips")),
                "slippage_pips": to_float(raw.get("slippage_pips")),
                "execution_delay_bars": to_float(raw.get("execution_delay_bars")),
                "execution_delay_cost_r": to_float(raw.get("execution_delay_cost_r")),
                "regime_label": raw.get("regime_label"),
                "portfolio_sleeve": raw.get("portfolio_sleeve"),
                "entry_mode": raw.get("entry_mode"),
                "entry_source": raw.get("entry_source"),
                "exit_reason": raw.get("exit_reason"),
                "score": to_float(raw.get("score")),
                "trigger_event": raw.get("trigger_event"),
                "trigger_strength": to_float(raw.get("trigger_strength")),
                "structure_event": raw.get("structure_event"),
                "zone": raw.get("zone"),
            }
        )
    return rows


def trade_signature(trade: TradeRow) -> tuple[str, str, str]:
    return (
        str(trade.raw.get("pair") or ""),
        str(trade.raw.get("side") or ""),
        str(trade.raw.get("signal_time") or ""),
    )


def compare_to_baseline(baseline: list[TradeRow], scenario: list[TradeRow]) -> dict[str, object]:
    base_map = {trade_signature(trade): trade for trade in baseline}
    scenario_map = {trade_signature(trade): trade for trade in scenario}
    matched_keys = sorted(set(base_map) & set(scenario_map))
    deltas = [scenario_map[key].r - base_map[key].r for key in matched_keys]
    winner_to_loss = [
        key
        for key in matched_keys
        if base_map[key].r > 0 and scenario_map[key].r < 0
    ]
    loss_to_winner = [
        key
        for key in matched_keys
        if base_map[key].r < 0 and scenario_map[key].r > 0
    ]
    return {
        "baseline_trades": len(baseline),
        "scenario_trades": len(scenario),
        "matched_trades": len(matched_keys),
        "baseline_only": len(set(base_map) - set(scenario_map)),
        "scenario_only": len(set(scenario_map) - set(base_map)),
        "total_delta_r": sum(deltas),
        "avg_delta_r": sum(deltas) / len(deltas) if deltas else 0.0,
        "winner_to_loss_count": len(winner_to_loss),
        "loss_to_winner_count": len(loss_to_winner),
        "worst_delta_trades": worst_delta_trades(base_map, scenario_map, matched_keys),
    }


def worst_delta_trades(
    baseline: dict[tuple[str, str, str], TradeRow],
    scenario: dict[tuple[str, str, str], TradeRow],
    keys: list[tuple[str, str, str]],
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    rows = []
    for key in keys:
        base = baseline[key]
        stress = scenario[key]
        rows.append(
            {
                "pair": key[0],
                "side": key[1],
                "signal_time": key[2],
                "baseline_r": base.r,
                "scenario_r": stress.r,
                "delta_r": stress.r - base.r,
                "scenario_friction_cost_r": stress.friction_cost_r,
                "regime_label": stress.raw.get("regime_label"),
                "entry_mode": stress.raw.get("entry_mode"),
                "entry_source": stress.raw.get("entry_source"),
                "exit_reason": stress.raw.get("exit_reason"),
            }
        )
    return sorted(rows, key=lambda row: to_float(row.get("delta_r")))[:limit]


def edge_flags(scenario: str, trades: list[TradeRow], grouped: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    total = summarize(trades)
    if to_float(total.get("total_friction_cost_r")) < -1.0:
        flags.append(
            {
                "type": "execution_friction",
                "severity": "high",
                "message": "Total reported spread/slippage/delay cost exceeds -1R.",
                "total_friction_cost_r": total.get("total_friction_cost_r"),
            }
        )
    for row in grouped.get("regime_label", []):
        if row.get("key") == "EXPANSION" and to_float(row.get("avg_r")) < 0:
            flags.append(
                {
                    "type": "regime_drag",
                    "severity": "high",
                    "message": f"{scenario}: EXPANSION trades are negative under stress.",
                    "metrics": row,
                }
            )
    for row in grouped.get("entry_source", []):
        if row.get("key") == "fallback" and to_float(row.get("avg_r")) < 0.05:
            flags.append(
                {
                    "type": "fallback_entry_drag",
                    "severity": "medium",
                    "message": f"{scenario}: MARKET fallback entries have weak expectancy.",
                    "metrics": row,
                }
            )
    for row in grouped.get("exit_reason", []):
        if row.get("key") == "timeout" and to_float(row.get("avg_r")) < 0.08:
            flags.append(
                {
                    "type": "timeout_drag",
                    "severity": "medium",
                    "message": f"{scenario}: timeout exits dominate low-R outcomes.",
                    "metrics": row,
                }
            )
    return flags


def scenario_payload(
    scenario: str,
    trades: list[TradeRow],
    *,
    baseline: list[TradeRow] | None,
    top_losses: int,
) -> dict[str, object]:
    grouped = {key: group_summary(trades, key) for key in GROUP_KEYS}
    return {
        "scenario": scenario,
        "overall": summarize(trades),
        "losers_only": summarize([trade for trade in trades if trade.r < 0]),
        "groups": grouped,
        "loser_groups": {key: losing_group_summary(trades, key) for key in GROUP_KEYS},
        "worst_trades": worst_trades(trades, limit=top_losses),
        "baseline_comparison": compare_to_baseline(baseline, trades) if baseline else None,
        "edge_flags": edge_flags(scenario, trades, grouped),
    }


def write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, default=str, allow_nan=False), encoding="utf-8")


def print_scenario(payload: dict[str, object]) -> None:
    overall = payload.get("overall", {}) if isinstance(payload.get("overall"), dict) else {}
    losers = payload.get("losers_only", {}) if isinstance(payload.get("losers_only"), dict) else {}
    print(
        "{scenario}: trades={trades} pf={pf:.2f} avg_r={avg_r:.3f} total_r={total_r:.3f} "
        "losses={losses} loss_r={loss_r:.3f} friction={friction:.3f}".format(
            scenario=payload.get("scenario"),
            trades=int(to_float(overall.get("trades"))),
            pf=to_float(overall.get("profit_factor")),
            avg_r=to_float(overall.get("avg_r")),
            total_r=to_float(overall.get("total_r")),
            losses=int(to_float(losers.get("trades"))),
            loss_r=to_float(losers.get("total_r")),
            friction=to_float(overall.get("total_friction_cost_r")),
        )
    )
    for group_key in ("pair", "regime_label", "entry_source", "exit_reason"):
        rows = payload.get("groups", {}).get(group_key, []) if isinstance(payload.get("groups"), dict) else []
        worst = rows[0] if rows else {}
        if worst:
            print(
                "  weakest_{key}: {name} trades={trades} avg_r={avg_r:.3f} total_r={total_r:.3f}".format(
                    key=group_key,
                    name=worst.get("key"),
                    trades=int(to_float(worst.get("trades"))),
                    avg_r=to_float(worst.get("avg_r")),
                    total_r=to_float(worst.get("total_r")),
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Phase 3 execution-friction edge compression.")
    parser.add_argument("--suite-dir", default="reports/phase3_strict_ltf_step3_full_suite")
    parser.add_argument("--baseline", default="ideal")
    parser.add_argument("--scenarios", default="moderate,harsh")
    parser.add_argument("--top-losses", type=int, default=10)
    parser.add_argument("--output", default="reports/phase3_friction_diagnostics.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suite_dir = Path(args.suite_dir)
    baseline = load_trades(suite_dir / args.baseline / "trades.csv") if args.baseline else None
    report: dict[str, object] = {
        "runner": "research.phase3_friction_diagnostics",
        "suite_dir": str(suite_dir),
        "baseline": args.baseline,
        "scenarios": [],
    }
    for scenario in parse_names(args.scenarios):
        trades = load_trades(suite_dir / scenario / "trades.csv")
        payload = scenario_payload(
            scenario,
            trades,
            baseline=baseline,
            top_losses=max(1, int(args.top_losses)),
        )
        report["scenarios"].append(payload)  # type: ignore[union-attr]
        print_scenario(payload)
    write_report(Path(args.output), report)
    print(f"\nSaved Phase 3 friction diagnostics: {args.output}")


if __name__ == "__main__":
    main()
