from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable

from backtest.engine import BacktestEngine, BacktestTrade, expectancy_stats
from backtest.news import NeutralNewsFeed
from backtest.smc_research_features import SMCResearchFeatureSettings
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


FeaturePredicate = Callable[[BacktestTrade], bool]


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


def _metrics(trades: list[BacktestTrade], *, baseline_count: int | None = None) -> dict[str, object]:
    values = [float(trade.r_multiple) for trade in trades]
    wins = [value for value in values if value > 0]
    losses = [abs(value) for value in values if value < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    stats = expectancy_stats(values)
    return {
        "trades": len(values),
        "pass_rate": round(len(values) / baseline_count, 6) if baseline_count else 1.0,
        "win_rate": round(len(wins) / len(values), 6) if values else 0.0,
        "profit_factor": _clean_float(gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)),
        "avg_r": round(mean(values), 6) if values else 0.0,
        "median_r": round(median(values), 6) if values else 0.0,
        "avg_win_r": round(stats["avg_win_r"], 6),
        "avg_loss_r": round(stats["avg_loss_r"], 6),
        "payoff_ratio": _clean_float(stats["payoff_ratio"]),
        "expectancy_r": round(stats["expectancy_r"], 6),
        "max_drawdown_r": round(_max_drawdown(values), 6),
    }


def _pair_metrics(trades: list[BacktestTrade], *, baseline_trades: list[BacktestTrade]) -> dict[str, dict[str, object]]:
    baseline_counts: dict[str, int] = {}
    for trade in baseline_trades:
        baseline_counts[trade.pair] = baseline_counts.get(trade.pair, 0) + 1

    grouped: dict[str, list[BacktestTrade]] = {pair: [] for pair in sorted(baseline_counts)}
    for trade in trades:
        grouped.setdefault(trade.pair, []).append(trade)

    return {
        pair: _metrics(pair_trades, baseline_count=baseline_counts.get(pair, 0))
        for pair, pair_trades in sorted(grouped.items())
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denom = math.sqrt(x_var * y_var)
    return numerator / denom if denom > 0 else 0.0


def _feature_value(trade: BacktestTrade, key: str) -> float | None:
    value = trade.smc_features.get(key) if isinstance(trade.smc_features, dict) else None
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_stats(trades: list[BacktestTrade], keys: list[str]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for key in keys:
        rows = [(value, float(trade.r_multiple)) for trade in trades if (value := _feature_value(trade, key)) is not None]
        values = [row[0] for row in rows]
        pnl = [row[1] for row in rows]
        winners = [value for value, r in rows if r > 0]
        losers = [value for value, r in rows if r < 0]
        output[key] = {
            "count": len(values),
            "avg": round(mean(values), 6) if values else 0.0,
            "winner_avg": round(mean(winners), 6) if winners else 0.0,
            "loser_avg": round(mean(losers), 6) if losers else 0.0,
            "winner_minus_loser": round((mean(winners) if winners else 0.0) - (mean(losers) if losers else 0.0), 6),
            "corr_with_r": round(_pearson(values, pnl), 6) if values else 0.0,
        }
    return output


def _build_market_data(settings: Settings, *, history_limit: int, cache_only: bool) -> MarketDataClient:
    return MarketDataClient(
        history_limit=history_limit,
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
    feature_settings: SMCResearchFeatureSettings,
    snapshot_cache: SnapshotCache | None = None,
    snapshot_cache_settings: SnapshotCacheSettings | None = None,
) -> BacktestEngine:
    market_data = _build_market_data(settings, history_limit=history_limit, cache_only=cache_only)
    live_news = NewsFilter(
        blackout_before_minutes=settings.news_blackout_before_minutes,
        blackout_after_minutes=settings.news_blackout_after_minutes,
        surprise_threshold=settings.news_surprise_threshold,
    )
    signal_engine = build_signal_engine(
        market_data=market_data,
        news_filter=live_news,
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
        smc_research_feature_settings=feature_settings,
    )


def _scenario_predicates(
    *,
    structure_threshold: float,
    ob_threshold: float,
    fvg_threshold: float,
) -> dict[str, FeaturePredicate]:
    return {
        "baseline": lambda trade: True,
        "strict_structure_score": lambda trade: (_feature_value(trade, "structure_strict_score") or 0.0) >= structure_threshold,
        "fresh_ob_score": lambda trade: (_feature_value(trade, "fresh_ob_score") or 0.0) >= ob_threshold,
        "relaxed_fvg_score": lambda trade: (_feature_value(trade, "relaxed_fvg_score") or 0.0) >= fvg_threshold,
        "strict_structure_plus_fresh_ob": lambda trade: (
            (_feature_value(trade, "structure_strict_score") or 0.0) >= structure_threshold
            and (_feature_value(trade, "fresh_ob_score") or 0.0) >= ob_threshold
        ),
        "strict_structure_plus_relaxed_fvg": lambda trade: (
            (_feature_value(trade, "structure_strict_score") or 0.0) >= structure_threshold
            and (_feature_value(trade, "relaxed_fvg_score") or 0.0) >= fvg_threshold
        ),
        "fresh_ob_plus_relaxed_fvg": lambda trade: (
            (_feature_value(trade, "fresh_ob_score") or 0.0) >= ob_threshold
            and (_feature_value(trade, "relaxed_fvg_score") or 0.0) >= fvg_threshold
        ),
        "all_features": lambda trade: (
            (_feature_value(trade, "structure_strict_score") or 0.0) >= structure_threshold
            and (_feature_value(trade, "fresh_ob_score") or 0.0) >= ob_threshold
            and (_feature_value(trade, "relaxed_fvg_score") or 0.0) >= fvg_threshold
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run logging-only SMC feature ablation diagnostics.")
    parser.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY")
    parser.add_argument("--history-limit", type=int, default=600)
    parser.add_argument("--history-limits", default=None)
    parser.add_argument("--evaluation-step", type=int, default=2)
    parser.add_argument("--evaluation-steps", default=None)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--warmup-bars", type=int, default=120)
    parser.add_argument("--ltf", default=None)
    parser.add_argument("--htf", default=None)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--structure-threshold", type=float, default=60.0)
    parser.add_argument("--ob-threshold", type=float, default=60.0)
    parser.add_argument("--fvg-threshold", type=float, default=60.0)
    parser.add_argument("--no-snapshot-cache", action="store_true")
    parser.add_argument("--snapshot-cache-size", type=int, default=250000)
    parser.add_argument("--output", default="reports/smc_feature_ablation.json")
    return parser


def _run_ablation(
    *,
    result,
    structure_threshold: float,
    ob_threshold: float,
    fvg_threshold: float,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    trades = result.trades
    predicates = _scenario_predicates(
        structure_threshold=structure_threshold,
        ob_threshold=ob_threshold,
        fvg_threshold=fvg_threshold,
    )
    scenarios: dict[str, object] = {}
    baseline_metrics = _metrics(trades)
    for name, predicate in predicates.items():
        selected = [trade for trade in trades if predicate(trade)]
        metrics = _metrics(selected, baseline_count=len(trades))
        scenarios[name] = {
            "metrics": metrics,
            "pair_metrics": _pair_metrics(selected, baseline_trades=trades),
            "delta_vs_baseline": {
                "trades": int(metrics["trades"]) - int(baseline_metrics["trades"]),
                "profit_factor": round(float(metrics["profit_factor"] or 0.0) - float(baseline_metrics["profit_factor"] or 0.0), 6),
                "avg_r": round(float(metrics["avg_r"]) - float(baseline_metrics["avg_r"]), 6),
                "max_drawdown_r": round(float(metrics["max_drawdown_r"]) - float(baseline_metrics["max_drawdown_r"]), 6),
            },
        }
        print(
            f"{name} | trades={metrics['trades']} pass={metrics['pass_rate']} "
            f"pf={metrics['profit_factor']} avg_r={metrics['avg_r']} dd={metrics['max_drawdown_r']}",
            flush=True,
        )

    feature_keys = ["structure_strict_score", "fresh_ob_score", "relaxed_fvg_score"]
    return scenarios, _feature_stats(trades, feature_keys)


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    pairs = _parse_pairs(args.pairs)
    history_limits = _parse_int_csv(args.history_limits) if args.history_limits else [args.history_limit]
    evaluation_steps = _parse_int_csv(args.evaluation_steps) if args.evaluation_steps else [args.evaluation_step]
    feature_settings = SMCResearchFeatureSettings(
        enabled=True,
        structure_scan_bars=settings.smc_structure_scan_bars,
        structure_min_break_pips=settings.smc_structure_min_break_pips,
        structure_level_bucket_pips=settings.smc_structure_level_bucket_pips,
        ob_lookback_bars=settings.smc_ob_lookback_bars,
        ob_max_age_bars=settings.smc_ob_max_age_bars,
        ob_max_width_pips=settings.smc_ob_max_width_pips,
        ob_max_distance_pips=settings.smc_ob_max_distance_pips,
        relaxed_fvg_lookback_bars=settings.smc_relaxed_fvg_lookback_bars,
        relaxed_fvg_min_gap_pips=settings.smc_relaxed_fvg_min_gap_pips,
        relaxed_fvg_max_distance_pips=settings.smc_relaxed_fvg_max_distance_pips,
    ).sanitized()

    snapshot_settings = SnapshotCacheSettings(
        enabled=not args.no_snapshot_cache,
        max_entries=max(1000, int(args.snapshot_cache_size)),
    )
    shared_snapshot_cache = SnapshotCache(snapshot_settings)
    output = Path(args.output)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "settings": {
            "pairs": pairs,
            "history_limits": history_limits,
            "evaluation_steps": evaluation_steps,
            "max_hold_bars": args.max_hold_bars,
            "warmup_bars": args.warmup_bars,
            "thresholds": {
                "structure": args.structure_threshold,
                "fresh_ob": args.ob_threshold,
                "relaxed_fvg": args.fvg_threshold,
            },
            "smc_research_features": asdict(feature_settings),
            "snapshot_cache": asdict(snapshot_settings),
        },
        "runs": [],
        "ranking": [],
        "snapshot_cache": shared_snapshot_cache.stats(),
    }

    for history_limit in history_limits:
        for evaluation_step in evaluation_steps:
            print(f"\n=== SMC feature ablation | bars={history_limit} step={evaluation_step} ===", flush=True)
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
                feature_settings=feature_settings,
                snapshot_cache=shared_snapshot_cache,
                snapshot_cache_settings=snapshot_settings,
            )
            result = engine.run(pairs)
            scenarios, feature_stats = _run_ablation(
                result=result,
                structure_threshold=args.structure_threshold,
                ob_threshold=args.ob_threshold,
                fvg_threshold=args.fvg_threshold,
            )
            run_payload = {
                "history_limit": history_limit,
                "evaluation_step": evaluation_step,
                "baseline_parameters": result.parameters,
                "baseline_pair_rows": result.pair_rows(),
                "scenarios": scenarios,
                "feature_stats": feature_stats,
            }
            report["runs"].append(run_payload)
            ranking_rows = []
            for run in report["runs"]:
                for scenario_name, scenario in run["scenarios"].items():
                    metrics = scenario["metrics"]
                    ranking_rows.append(
                        {
                            "scenario": scenario_name,
                            "history_limit": run["history_limit"],
                            "evaluation_step": run["evaluation_step"],
                            "trades": metrics["trades"],
                            "profit_factor": metrics["profit_factor"],
                            "avg_r": metrics["avg_r"],
                            "max_drawdown_r": metrics["max_drawdown_r"],
                            "pass_rate": metrics["pass_rate"],
                            "candidate_score": round(
                                float(metrics["avg_r"]) * 4.0
                                + min(3.0, float(metrics["profit_factor"] or 0.0))
                                - float(metrics["max_drawdown_r"]) * 0.05
                                - (0.25 if int(metrics["trades"]) < 10 else 0.0),
                                6,
                            ),
                        }
                    )
            report["ranking"] = sorted(ranking_rows, key=lambda row: float(row["candidate_score"]), reverse=True)
            report["snapshot_cache"] = shared_snapshot_cache.stats()
            _write_report(output, report)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["snapshot_cache"] = shared_snapshot_cache.stats()
    _write_report(output, report)
    print(f"Saved SMC feature ablation report: {output}", flush=True)


if __name__ == "__main__":
    main()
