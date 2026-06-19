from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

from backtest.engine import BacktestEngine, BacktestRunResult, BacktestTrade, expectancy_stats
from backtest.news import NeutralNewsFeed
from backtest.snapshot_cache import SnapshotCache, SnapshotCacheSettings
from backtest_runner import (
    build_atr_risk_settings,
    build_equity_protection_settings,
    build_execution_settings,
    build_exit_settings,
    build_meta_label_settings,
    build_portfolio_layer_settings,
    build_signal_engine,
    build_sizing_settings,
)
from config import Settings
from data.market_data import MarketDataCacheConfig, MarketDataClient
from execution.news import NewsFilter


@dataclass(frozen=True)
class StructureQualityScenario:
    name: str
    enabled: bool
    max_bonus: int
    allowed_regimes: tuple[str, ...] = ()
    allowed_pairs: tuple[str, ...] = ()
    excluded_pairs: tuple[str, ...] = ()


METRIC_KEYS = (
    "trades",
    "wins",
    "losses",
    "win_rate",
    "avg_r",
    "expectancy_r",
    "avg_win_r",
    "avg_loss_r",
    "payoff_ratio",
    "profit_factor",
    "max_drawdown_r",
    "avg_score",
    "avg_shadow_bonus",
    "acceptance_rate",
    "fill_rate",
    "limit_entries",
    "market_entries",
    "partial_exits",
    "break_even_activations",
    "trailing_activations",
    "tp_hits",
    "sl_hits",
    "timeout_exits",
)


def _parse_pairs(raw: str) -> list[str]:
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _clean_float(value: float) -> float | None:
    if math.isnan(value):
        return None
    if math.isinf(value):
        return 999.0 if value > 0 else -999.0
    return round(float(value), 6)


def _finite_number(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    if math.isinf(parsed):
        return 999.0 if parsed > 0 else -999.0
    return parsed


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return _clean_float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _max_drawdown(values: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in METRIC_KEYS:
        if key in metrics:
            output[key] = metrics[key]
    return output


def _overall_acceptance_metrics(result: BacktestRunResult) -> dict[str, object]:
    evaluations = sum(int(report.evaluations) for report in result.pair_reports)
    trades = len(result.trades)
    overall = result.overall_metrics()
    regime_acceptances = overall.get("regime_acceptances", {})
    accepted = sum(int(value) for value in regime_acceptances.values()) if isinstance(regime_acceptances, dict) else trades
    return {
        "evaluations": evaluations,
        "accepted_signals": accepted,
        "acceptance_rate": round(accepted / evaluations, 6) if evaluations else 0.0,
        "trade_acceptance_rate": round(trades / evaluations, 6) if evaluations else 0.0,
    }


def _structure_bonus(trade: BacktestTrade) -> int:
    if isinstance(trade.feature_breakdown, dict):
        try:
            return int(trade.feature_breakdown.get("structure_quality", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _structure_bonus_stats(trades: list[BacktestTrade]) -> dict[str, object]:
    bonuses = [_structure_bonus(trade) for trade in trades]
    positive = [value for value in bonuses if value > 0]
    winners = [_structure_bonus(trade) for trade in trades if trade.r_multiple > 0]
    losers = [_structure_bonus(trade) for trade in trades if trade.r_multiple < 0]
    return {
        "avg_bonus": round(mean(bonuses), 6) if bonuses else 0.0,
        "max_bonus": max(bonuses) if bonuses else 0,
        "bonus_trade_count": len(positive),
        "bonus_trade_rate": round(len(positive) / len(bonuses), 6) if bonuses else 0.0,
        "winner_avg_bonus": round(mean(winners), 6) if winners else 0.0,
        "loser_avg_bonus": round(mean(losers), 6) if losers else 0.0,
    }


def _regime_metrics(result: BacktestRunResult) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[BacktestTrade]] = {}
    for trade in result.trades:
        grouped.setdefault((trade.regime_label or "UNKNOWN").upper(), []).append(trade)

    overall = result.overall_metrics()
    evaluations = overall.get("regime_evaluations", {}) if isinstance(overall.get("regime_evaluations"), dict) else {}
    acceptances = overall.get("regime_acceptances", {}) if isinstance(overall.get("regime_acceptances"), dict) else {}
    regimes = sorted(set(grouped) | {str(key).upper() for key in evaluations} | {str(key).upper() for key in acceptances})

    output: dict[str, dict[str, object]] = {}
    for regime in regimes:
        trades = grouped.get(regime, [])
        values = [float(trade.r_multiple) for trade in trades]
        stats = expectancy_stats(values)
        gross_profit = sum(value for value in values if value > 0)
        gross_loss = abs(sum(value for value in values if value < 0))
        eval_count = int(evaluations.get(regime, 0))
        accepted_count = int(acceptances.get(regime, 0))
        output[regime] = {
            "trades": len(values),
            "evaluations": eval_count,
            "acceptances": accepted_count,
            "acceptance_rate": round(accepted_count / eval_count, 6) if eval_count else 0.0,
            "win_rate": round(sum(1 for value in values if value > 0) / len(values), 6) if values else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
            "avg_r": mean(values) if values else 0.0,
            "expectancy_r": stats["expectancy_r"],
            "max_drawdown_r": _max_drawdown(values),
            "structure_bonus": _structure_bonus_stats(trades),
        }
    return output


def _pair_rows(result: BacktestRunResult) -> list[dict[str, object]]:
    rows = result.pair_rows()
    trades_by_pair: dict[str, list[BacktestTrade]] = {}
    for trade in result.trades:
        trades_by_pair.setdefault(trade.pair, []).append(trade)
    for row in rows:
        pair = str(row.get("pair", "")).upper().replace("/", "")
        row["structure_bonus"] = _structure_bonus_stats(trades_by_pair.get(pair, []))
    return rows


def _scenario_settings(
    base: Settings,
    *,
    enabled: bool,
    max_bonus: int,
    min_score_for_bonus: float,
    allowed_regimes: tuple[str, ...] = (),
    allowed_pairs: tuple[str, ...] = (),
    excluded_pairs: tuple[str, ...] = (),
) -> Settings:
    return replace(
        base,
        enable_structure_quality_scoring=enabled,
        structure_quality_max_bonus=max(0, int(max_bonus)),
        structure_quality_min_score_for_bonus=max(0.0, min(100.0, float(min_score_for_bonus))),
        structure_quality_backtest_only=True,
        allow_live_structure_quality_scoring=False,
        structure_quality_allowed_regimes=list(allowed_regimes),
        structure_quality_allowed_pairs=list(allowed_pairs),
        structure_quality_excluded_pairs=list(excluded_pairs),
    )


def _build_engine(
    *,
    settings: Settings,
    history_limit: int,
    evaluation_step: int,
    max_hold_bars: int,
    warmup_bars: int,
    cache_only: bool,
    ltf: str,
    htf: str,
    trigger: str,
    snapshot_cache: SnapshotCache | None,
    snapshot_cache_settings: SnapshotCacheSettings,
) -> BacktestEngine:
    market_data = MarketDataClient(
        history_limit=max(settings.history_limit, history_limit),
        data_source=settings.data_source,
        mt5_login=settings.mt5_login,
        mt5_password=settings.mt5_password,
        mt5_server=settings.mt5_server,
        mt5_path=settings.mt5_path,
        cache_config=MarketDataCacheConfig(
            enabled=settings.market_data_cache_enabled,
            cache_dir=settings.market_data_cache_dir,
            ttl_hours=settings.market_data_cache_ttl_hours,
            mode="cache_only" if cache_only else settings.market_data_cache_mode,
        ),
    )
    news_filter = NewsFilter(
        blackout_before_minutes=settings.news_blackout_before_minutes,
        blackout_after_minutes=settings.news_blackout_after_minutes,
        surprise_threshold=settings.news_surprise_threshold,
    )
    signal_engine = build_signal_engine(
        market_data=market_data,
        news_filter=news_filter,
        settings=settings,
        htf=htf,
        ltf=ltf,
        trigger=trigger,
        enable_shadow_scoring=True,
        enable_mitigation_entry=settings.enable_mitigation_entry,
    )
    return BacktestEngine(
        market_data=market_data,
        signal_engine=signal_engine,
        history_limit=history_limit,
        max_hold_bars=max_hold_bars,
        warmup_bars=warmup_bars,
        evaluation_step=evaluation_step,
        news_feed=NeutralNewsFeed(),
        execution_settings=build_execution_settings(settings),
        atr_risk_settings=build_atr_risk_settings(settings),
        equity_protection_settings=build_equity_protection_settings(settings),
        exit_settings=build_exit_settings(settings),
        sizing_settings=build_sizing_settings(settings),
        meta_label_settings=build_meta_label_settings(settings),
        portfolio_layer_settings=build_portfolio_layer_settings(settings),
        snapshot_cache_settings=snapshot_cache_settings,
        snapshot_cache=snapshot_cache,
    )


def _result_payload(
    *,
    name: str,
    max_bonus: int,
    settings: Settings,
    result: BacktestRunResult,
    history_limit: int,
    evaluation_step: int,
) -> dict[str, object]:
    overall = result.overall_metrics()
    metrics = _compact_metrics(overall)
    metrics.update(_overall_acceptance_metrics(result))
    return {
        "scenario": name,
        "history_limit": history_limit,
        "evaluation_step": evaluation_step,
        "structure_quality": {
            "enabled": bool(settings.enable_structure_quality_scoring),
            "max_bonus": int(max_bonus),
            "min_score_for_bonus": settings.structure_quality_min_score_for_bonus,
            "backtest_only": settings.structure_quality_backtest_only,
            "allowed_regimes": list(settings.structure_quality_allowed_regimes),
            "allowed_pairs": list(settings.structure_quality_allowed_pairs),
            "excluded_pairs": list(settings.structure_quality_excluded_pairs),
        },
        "metrics": metrics,
        "structure_bonus": _structure_bonus_stats(result.trades),
        "pairs": _pair_rows(result),
        "regimes": _regime_metrics(result),
        "rejections": overall.get("rejections", {}),
        "snapshot_cache": result.parameters.get("snapshot_cache_stats", {}),
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
    }


def _delta(candidate: dict[str, object], baseline: dict[str, object] | None) -> dict[str, object]:
    if baseline is None:
        return {}
    left = candidate.get("metrics", {}) if isinstance(candidate.get("metrics"), dict) else {}
    right = baseline.get("metrics", {}) if isinstance(baseline.get("metrics"), dict) else {}
    keys = ("trades", "win_rate", "profit_factor", "avg_r", "expectancy_r", "max_drawdown_r", "acceptance_rate")
    output: dict[str, object] = {}
    for key in keys:
        try:
            left_value = _finite_number(left.get(key, 0.0))
            right_value = _finite_number(right.get(key, 0.0))
            output[key] = round(left_value - right_value, 6)
        except (TypeError, ValueError):
            output[key] = 0.0
    return output


def _refresh_comparisons(report: dict[str, object]) -> None:
    runs = report.get("runs", [])
    if not isinstance(runs, list):
        return
    baselines: dict[tuple[int, int], dict[str, object]] = {}
    for row in runs:
        if not isinstance(row, dict):
            continue
        if row.get("scenario") == "baseline":
            baselines[(int(row.get("history_limit", 0)), int(row.get("evaluation_step", 0)))] = row
    for row in runs:
        if not isinstance(row, dict):
            continue
        baseline = baselines.get((int(row.get("history_limit", 0)), int(row.get("evaluation_step", 0))))
        row["delta_vs_baseline"] = _delta(row, baseline)

    ranking = []
    for row in runs:
        if not isinstance(row, dict) or row.get("scenario") == "baseline":
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        delta = row.get("delta_vs_baseline", {}) if isinstance(row.get("delta_vs_baseline"), dict) else {}
        score = (
            _finite_number(metrics.get("profit_factor", 0.0))
            + _finite_number(metrics.get("avg_r", 0.0)) * 4.0
            - _finite_number(metrics.get("max_drawdown_r", 0.0)) * 0.04
            + _finite_number(delta.get("profit_factor", 0.0)) * 0.5
            + _finite_number(delta.get("avg_r", 0.0)) * 2.0
        )
        ranking.append(
            {
                "scenario": row.get("scenario"),
                "history_limit": row.get("history_limit"),
                "evaluation_step": row.get("evaluation_step"),
                "candidate_score": round(score, 6),
                "metrics": metrics,
                "delta_vs_baseline": delta,
            }
        )
    report["ranking"] = sorted(ranking, key=lambda item: float(item["candidate_score"]), reverse=True)


def _unique_scenarios(scenarios: Iterable[StructureQualityScenario]) -> list[StructureQualityScenario]:
    output: dict[str, StructureQualityScenario] = {}
    for scenario in scenarios:
        output.setdefault(scenario.name, scenario)
    return list(output.values())


def _build_scenarios(*, mode: str, bonus_values: list[int]) -> list[StructureQualityScenario]:
    scenarios: list[StructureQualityScenario] = [
        StructureQualityScenario(name="baseline", enabled=False, max_bonus=0),
    ]
    if mode in {"bonus-grid", "both"}:
        scenarios.extend(
            StructureQualityScenario(name=f"structure_bonus_{bonus}", enabled=True, max_bonus=bonus)
            for bonus in bonus_values
        )
    if mode in {"conditional", "both"}:
        scenarios.extend(
            [
                StructureQualityScenario(
                    name="range_bonus_4",
                    enabled=True,
                    max_bonus=4,
                    allowed_regimes=("RANGE",),
                ),
                StructureQualityScenario(
                    name="usdjpy_bonus_4",
                    enabled=True,
                    max_bonus=4,
                    allowed_pairs=("USDJPY",),
                ),
                StructureQualityScenario(
                    name="usdjpy_range_bonus_4",
                    enabled=True,
                    max_bonus=4,
                    allowed_regimes=("RANGE",),
                    allowed_pairs=("USDJPY",),
                ),
                StructureQualityScenario(
                    name="exclude_gbpusd_bonus_4",
                    enabled=True,
                    max_bonus=4,
                    excluded_pairs=("GBPUSD",),
                ),
            ]
        )
        scenarios.extend(
            StructureQualityScenario(
                name=f"range_bonus_{bonus}",
                enabled=True,
                max_bonus=bonus,
                allowed_regimes=("RANGE",),
            )
            for bonus in (2, 3, 4, 5)
        )
    return _unique_scenarios(scenarios)


def _filter_scenarios(scenarios: list[StructureQualityScenario], raw_filter: str | None) -> list[StructureQualityScenario]:
    if not raw_filter or not raw_filter.strip():
        return scenarios
    wanted = {item.strip() for item in raw_filter.split(",") if item.strip()}
    if not wanted:
        return scenarios
    return [scenario for scenario in scenarios if scenario.name in wanted]


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate structure-quality soft scoring bonuses with persisted JSON output.")
    parser.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY")
    parser.add_argument("--history-limits", default="1200,3000")
    parser.add_argument("--evaluation-steps", default="2")
    parser.add_argument("--bonus-values", default="4,6,8")
    parser.add_argument("--scenario-mode", choices=("bonus-grid", "conditional", "both"), default="bonus-grid")
    parser.add_argument("--scenario-filter", default=None, help="Comma-separated scenario names to run, e.g. baseline,range_bonus_2")
    parser.add_argument("--min-score-for-bonus", type=float, default=60.0)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--warmup-bars", type=int, default=120)
    parser.add_argument("--ltf", default=None)
    parser.add_argument("--htf", default=None)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--no-snapshot-cache", action="store_true")
    parser.add_argument("--snapshot-cache-size", type=int, default=250000)
    parser.add_argument("--output", default="reports/structure_quality_calibration.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    pairs = _parse_pairs(args.pairs)
    history_limits = _parse_int_csv(args.history_limits)
    evaluation_steps = _parse_int_csv(args.evaluation_steps)
    bonus_values = _parse_int_csv(args.bonus_values)
    scenario_mode = str(args.scenario_mode)
    snapshot_settings = SnapshotCacheSettings(
        enabled=not args.no_snapshot_cache and base_settings.enable_backtest_snapshot_cache,
        max_entries=max(1000, int(args.snapshot_cache_size)),
    )
    shared_snapshot_cache = SnapshotCache(snapshot_settings)
    output = Path(args.output)
    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "settings": {
            "pairs": pairs,
            "history_limits": history_limits,
            "evaluation_steps": evaluation_steps,
            "bonus_values": bonus_values,
            "scenario_mode": scenario_mode,
            "scenario_filter": args.scenario_filter,
            "min_score_for_bonus": args.min_score_for_bonus,
            "max_hold_bars": args.max_hold_bars,
            "warmup_bars": args.warmup_bars,
            "cache_only": args.cache_only,
            "snapshot_cache": asdict(snapshot_settings),
            "timeframes": {
                "ltf": (args.ltf or base_settings.ltf_timeframe).upper(),
                "htf": (args.htf or base_settings.htf_timeframe).upper(),
                "trigger": (args.trigger or base_settings.trigger_timeframe).upper(),
            },
        },
        "runs": [],
        "ranking": [],
        "snapshot_cache": shared_snapshot_cache.stats(),
    }

    for history_limit in history_limits:
        for evaluation_step in evaluation_steps:
            scenarios = _filter_scenarios(
                _build_scenarios(mode=scenario_mode, bonus_values=bonus_values),
                args.scenario_filter,
            )
            for scenario in scenarios:
                print(
                    f"\n=== Structure quality calibration | bars={history_limit} step={evaluation_step} scenario={scenario.name} ===",
                    flush=True,
                )
                settings = _scenario_settings(
                    base_settings,
                    enabled=scenario.enabled,
                    max_bonus=scenario.max_bonus,
                    min_score_for_bonus=args.min_score_for_bonus,
                    allowed_regimes=scenario.allowed_regimes,
                    allowed_pairs=scenario.allowed_pairs,
                    excluded_pairs=scenario.excluded_pairs,
                )
                engine = _build_engine(
                    settings=settings,
                    history_limit=history_limit,
                    evaluation_step=max(1, evaluation_step),
                    max_hold_bars=max(1, args.max_hold_bars),
                    warmup_bars=max(80, args.warmup_bars),
                    cache_only=args.cache_only,
                    ltf=(args.ltf or settings.ltf_timeframe).upper(),
                    htf=(args.htf or settings.htf_timeframe).upper(),
                    trigger=(args.trigger or settings.trigger_timeframe).upper(),
                    snapshot_cache=shared_snapshot_cache,
                    snapshot_cache_settings=snapshot_settings,
                )
                result = engine.run(pairs)
                payload = _result_payload(
                    name=scenario.name,
                    max_bonus=scenario.max_bonus,
                    settings=settings,
                    result=result,
                    history_limit=history_limit,
                    evaluation_step=evaluation_step,
                )
                metrics = payload["metrics"]
                print(
                    "{name} | trades={trades} pf={pf} avg_r={avg_r} dd={dd} win_rate={wr} bonus_avg={bonus}".format(
                        name=scenario.name,
                        trades=metrics.get("trades"),
                        pf=_clean_float(float(metrics.get("profit_factor", 0.0) or 0.0)),
                        avg_r=_clean_float(float(metrics.get("avg_r", 0.0) or 0.0)),
                        dd=_clean_float(float(metrics.get("max_drawdown_r", 0.0) or 0.0)),
                        wr=_clean_float(float(metrics.get("win_rate", 0.0) or 0.0)),
                        bonus=payload["structure_bonus"].get("avg_bonus") if isinstance(payload.get("structure_bonus"), dict) else 0.0,
                    ),
                    flush=True,
                )
                runs = report.get("runs")
                if isinstance(runs, list):
                    runs.append(payload)
                report["snapshot_cache"] = shared_snapshot_cache.stats()
                _refresh_comparisons(report)
                _write_report(output, report)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["snapshot_cache"] = shared_snapshot_cache.stats()
    _refresh_comparisons(report)
    _write_report(output, report)
    print(f"Saved structure quality calibration report: {output}", flush=True)


if __name__ == "__main__":
    main()
