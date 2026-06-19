from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

import pandas as pd

from analytics.monte_carlo import MonteCarloSettings, run_monte_carlo
from backtest.engine import BacktestEngine, BacktestPairReport, BacktestRunResult, expectancy_stats
from backtest.news import NeutralNewsFeed
from backtest.portfolio_layer import PortfolioLayerState
from backtest.risk import EquityProtectionState
from backtest.snapshot_cache import SnapshotCache, SnapshotCacheSettings
from backtest_runner import (
    build_atr_risk_settings,
    build_equity_protection_settings,
    build_execution_settings,
    build_exit_settings,
    build_meta_label_settings,
    build_portfolio_layer_settings,
    build_smc_research_feature_settings,
    build_signal_engine,
    build_sizing_settings,
)
from config import Settings
from data.market_data import MarketDataCacheConfig, MarketDataClient
from execution.news import NewsFilter


@dataclass(frozen=True)
class CalibrationScenario:
    name: str
    enable_adaptive_weights: bool = False
    adaptive_weights_preset: str = "default"
    enable_exit_engine: bool = False
    enable_adaptive_sizing: bool = False
    sizing_min_multiplier: float = 0.40
    sizing_max_multiplier: float = 1.50
    enable_meta_label: bool = False
    meta_label_mode: str = "analysis_only"
    meta_label_probability_threshold: float = 0.55
    meta_label_enable_size_adjustment: bool = False
    enable_portfolio_layer: bool = False
    portfolio_layer_mode: str = "analysis_only"
    portfolio_layer_max_sleeve_concentration: float = 0.55


SCENARIOS: dict[str, CalibrationScenario] = {
    "baseline": CalibrationScenario(name="baseline"),
    "exit_v2": CalibrationScenario(name="exit_v2", enable_exit_engine=True),
    "weights_v1": CalibrationScenario(
        name="weights_v1",
        enable_adaptive_weights=True,
        adaptive_weights_preset="effectiveness_v1",
    ),
    "weights_v1_sizing": CalibrationScenario(
        name="weights_v1_sizing",
        enable_adaptive_weights=True,
        adaptive_weights_preset="effectiveness_v1",
        enable_adaptive_sizing=True,
    ),
    "weights_v1_portfolio_c045": CalibrationScenario(
        name="weights_v1_portfolio_c045",
        enable_adaptive_weights=True,
        adaptive_weights_preset="effectiveness_v1",
        enable_portfolio_layer=True,
        portfolio_layer_mode="apply",
        portfolio_layer_max_sleeve_concentration=0.45,
    ),
    "sizing_only": CalibrationScenario(
        name="sizing_only",
        enable_adaptive_sizing=True,
    ),
    "exit_sizing": CalibrationScenario(
        name="exit_sizing",
        enable_exit_engine=True,
        enable_adaptive_sizing=True,
    ),
    "portfolio_c045": CalibrationScenario(
        name="portfolio_c045",
        enable_portfolio_layer=True,
        portfolio_layer_mode="apply",
        portfolio_layer_max_sleeve_concentration=0.45,
    ),
    "exit_portfolio_c045": CalibrationScenario(
        name="exit_portfolio_c045",
        enable_exit_engine=True,
        enable_portfolio_layer=True,
        portfolio_layer_mode="apply",
        portfolio_layer_max_sleeve_concentration=0.45,
    ),
    "tier2_meta_052": CalibrationScenario(
        name="tier2_meta_052",
        enable_exit_engine=True,
        enable_adaptive_sizing=True,
        enable_meta_label=True,
        meta_label_mode="hard_gate",
        meta_label_probability_threshold=0.52,
        meta_label_enable_size_adjustment=True,
    ),
    "tier2_meta_055": CalibrationScenario(
        name="tier2_meta_055",
        enable_exit_engine=True,
        enable_adaptive_sizing=True,
        enable_meta_label=True,
        meta_label_mode="hard_gate",
        meta_label_probability_threshold=0.55,
        meta_label_enable_size_adjustment=True,
    ),
    "tier2_meta_058": CalibrationScenario(
        name="tier2_meta_058",
        enable_exit_engine=True,
        enable_adaptive_sizing=True,
        enable_meta_label=True,
        meta_label_mode="hard_gate",
        meta_label_probability_threshold=0.58,
        meta_label_enable_size_adjustment=True,
    ),
    "tier3_c045": CalibrationScenario(
        name="tier3_c045",
        enable_exit_engine=True,
        enable_adaptive_sizing=True,
        enable_meta_label=True,
        meta_label_mode="hard_gate",
        meta_label_probability_threshold=0.55,
        meta_label_enable_size_adjustment=True,
        enable_portfolio_layer=True,
        portfolio_layer_mode="apply",
        portfolio_layer_max_sleeve_concentration=0.45,
    ),
    "tier3_c055": CalibrationScenario(
        name="tier3_c055",
        enable_exit_engine=True,
        enable_adaptive_sizing=True,
        enable_meta_label=True,
        meta_label_mode="hard_gate",
        meta_label_probability_threshold=0.55,
        meta_label_enable_size_adjustment=True,
        enable_portfolio_layer=True,
        portfolio_layer_mode="apply",
        portfolio_layer_max_sleeve_concentration=0.55,
    ),
}


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv(raw)]


def _clean_pairs(raw: str) -> list[str]:
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


def _scenario_settings(base: Settings, scenario: CalibrationScenario) -> Settings:
    return replace(
        base,
        enable_adaptive_weights=scenario.enable_adaptive_weights,
        adaptive_weights_preset=scenario.adaptive_weights_preset,
        enable_exit_engine=scenario.enable_exit_engine,
        enable_adaptive_sizing=scenario.enable_adaptive_sizing,
        sizing_min_multiplier=scenario.sizing_min_multiplier,
        sizing_max_multiplier=scenario.sizing_max_multiplier,
        enable_meta_label=scenario.enable_meta_label,
        meta_label_mode=scenario.meta_label_mode,
        meta_label_probability_threshold=scenario.meta_label_probability_threshold,
        meta_label_enable_size_adjustment=scenario.meta_label_enable_size_adjustment,
        enable_portfolio_layer=scenario.enable_portfolio_layer,
        portfolio_layer_mode=scenario.portfolio_layer_mode,
        portfolio_layer_max_sleeve_concentration=scenario.portfolio_layer_max_sleeve_concentration,
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
) -> BacktestEngine:
    cache_mode = "cache_only" if cache_only else settings.market_data_cache_mode
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
            mode=cache_mode,
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
        snapshot_cache_settings=SnapshotCacheSettings(
            enabled=settings.enable_backtest_snapshot_cache,
            max_entries=settings.backtest_snapshot_cache_max_entries,
        ),
        snapshot_cache=snapshot_cache,
        smc_research_feature_settings=build_smc_research_feature_settings(settings),
    )


def _compact_metrics(result: BacktestRunResult) -> dict[str, object]:
    metrics = result.overall_metrics()
    keys = (
        "trades",
        "win_rate",
        "profit_factor",
        "avg_r",
        "expectancy_r",
        "avg_win_r",
        "avg_loss_r",
        "payoff_ratio",
        "max_drawdown_r",
        "acceptance_rate",
        "avg_sizing_multiplier",
        "avg_meta_probability",
        "avg_portfolio_multiplier",
        "avg_exit_target_rr",
    )
    return {key: metrics.get(key, 0.0) for key in keys}


def _regime_metrics(result: BacktestRunResult) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for trade in result.trades:
        grouped.setdefault((trade.regime_label or "UNKNOWN").upper(), []).append(float(trade.r_multiple))

    output: dict[str, dict[str, float]] = {}
    for regime, values in grouped.items():
        stats = expectancy_stats(values)
        gross_profit = sum(value for value in values if value > 0)
        gross_loss = abs(sum(value for value in values if value < 0))
        output[regime] = {
            "trades": float(len(values)),
            "win_rate": sum(1 for value in values if value > 0) / len(values),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
            "avg_r": mean(values),
            "expectancy_r": stats["expectancy_r"],
            "max_drawdown_r": _max_drawdown(values),
        }
    return output


def _max_drawdown(values: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _result_payload(
    *,
    result: BacktestRunResult,
    scenario: CalibrationScenario,
    history_limit: int,
    evaluation_step: int,
    mc_iterations: int,
    mc_seed: int,
    mc_ruin_dd: float,
) -> dict[str, object]:
    r_values = [float(trade.r_multiple) for trade in result.trades]
    return {
        "scenario": scenario.name,
        "scenario_settings": asdict(scenario),
        "history_limit": history_limit,
        "evaluation_step": evaluation_step,
        "metrics": _compact_metrics(result),
        "regimes": _regime_metrics(result),
        "monte_carlo": run_monte_carlo(
            r_values,
            MonteCarloSettings(iterations=mc_iterations, seed=mc_seed, ruin_drawdown_r=mc_ruin_dd),
        ),
        "pairs": result.pair_rows(),
        "snapshot_cache": result.parameters.get("snapshot_cache_stats", {}),
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
    }


def _slice_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame.index >= start) & (frame.index < end)].copy()


def _load_frames(
    engine: BacktestEngine,
    pairs: list[str],
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], dict[str, str]]:
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    errors: dict[str, str] = {}
    for pair in pairs:
        try:
            frames[pair] = engine.load_pair_frames(pair)
        except Exception as exc:
            errors[pair] = str(exc)
    return frames, errors


def _common_bounds(frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for ltf, htf, trigger in frames.values():
        if ltf.empty or htf.empty or trigger.empty:
            continue
        starts.append(max(ltf.index.min(), htf.index.min(), trigger.index.min()))
        ends.append(min(ltf.index.max(), htf.index.max(), trigger.index.max()))
    if not starts or not ends:
        return None
    start = max(starts)
    end = min(ends)
    return (start, end) if start < end else None


def _run_segment(
    engine: BacktestEngine,
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mode: str,
) -> BacktestRunResult:
    if hasattr(engine.signal_engine, "reset_release_state"):
        engine.signal_engine.reset_release_state()
    reports: list[BacktestPairReport] = []
    portfolio_state = PortfolioLayerState(engine.portfolio_layer_settings)
    equity_state = EquityProtectionState(engine.equity_protection_settings) if engine.equity_protection_settings.enabled else None
    started_at = datetime.now(timezone.utc)

    universe = set(frames.keys())
    for pair in sorted(frames):
        ltf, htf, trigger = frames[pair]
        reference_pair = engine.signal_engine._resolve_smt_reference_pair(pair, universe)
        reference_trigger = None
        if reference_pair is not None and reference_pair in frames:
            reference_trigger = _slice_frame(frames[reference_pair][2], start, end)
        reports.append(
            engine.run_pair_from_frames(
                pair,
                _slice_frame(ltf, start, end),
                _slice_frame(htf, start, end),
                _slice_frame(trigger, start, end),
                reference_pair=reference_pair,
                reference_trigger=reference_trigger,
                equity_state=equity_state,
                portfolio_state=portfolio_state,
            )
        )

    return BacktestRunResult(
        pair_reports=reports,
        parameters={"mode": mode, "window_start": start.isoformat(), "window_end": end.isoformat()},
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        news_mode=engine.news_feed.__class__.__name__,
    )


def _walk_forward_days(
    *,
    engine: BacktestEngine,
    pairs: list[str],
    train_days: int,
    test_days: int,
    step_days: int,
    mc_iterations: int,
    mc_seed: int,
    mc_ruin_dd: float,
) -> dict[str, object]:
    frames, errors = _load_frames(engine, pairs)
    bounds = _common_bounds(frames)
    if bounds is None:
        return {"window_count": 0, "errors": errors, "windows": []}

    cursor, global_end = bounds
    windows: list[dict[str, object]] = []
    idx = 1
    while True:
        train_end = cursor + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > global_end:
            break
        train_result = _run_segment(engine, frames, start=cursor, end=train_end, mode="wf_train")
        test_result = _run_segment(engine, frames, start=train_end, end=test_end, mode="wf_test")
        test_r = [float(trade.r_multiple) for trade in test_result.trades]
        windows.append(
            {
                "window_index": idx,
                "train_start": cursor.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": train_end.isoformat(),
                "test_end": test_end.isoformat(),
                "train_metrics": _compact_metrics(train_result),
                "test_metrics": _compact_metrics(test_result),
                "test_monte_carlo": run_monte_carlo(
                    test_r,
                    MonteCarloSettings(iterations=mc_iterations, seed=mc_seed + idx, ruin_drawdown_r=mc_ruin_dd),
                ),
            }
        )
        idx += 1
        cursor = cursor + pd.Timedelta(days=step_days)

    test_metrics = [window["test_metrics"] for window in windows]
    return {
        "window_count": len(windows),
        "errors": errors,
        "summary": {
            "test_avg_pf": _avg_metric(test_metrics, "profit_factor"),
            "test_avg_r": _avg_metric(test_metrics, "avg_r"),
            "test_avg_drawdown_r": _avg_metric(test_metrics, "max_drawdown_r"),
            "test_total_trades": sum(int(row.get("trades", 0)) for row in test_metrics),
        },
        "windows": windows,
    }


def _run_segment_with_context(
    engine: BacktestEngine,
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *,
    evaluation_start: pd.Timestamp,
    end: pd.Timestamp,
    mode: str,
) -> BacktestRunResult:
    if hasattr(engine.signal_engine, "reset_release_state"):
        engine.signal_engine.reset_release_state()
    reports: list[BacktestPairReport] = []
    portfolio_state = PortfolioLayerState(engine.portfolio_layer_settings)
    equity_state = EquityProtectionState(engine.equity_protection_settings) if engine.equity_protection_settings.enabled else None
    started_at = datetime.now(timezone.utc)
    universe = set(frames.keys())

    for pair in sorted(frames):
        ltf, htf, trigger = frames[pair]
        reference_pair = engine.signal_engine._resolve_smt_reference_pair(pair, universe)
        reference_trigger = None
        if reference_pair is not None and reference_pair in frames:
            reference_trigger = frames[reference_pair][2][frames[reference_pair][2].index < end].copy()
        reports.append(
            engine.run_pair_from_frames(
                pair,
                ltf[ltf.index < end].copy(),
                htf[htf.index < end].copy(),
                trigger[trigger.index < end].copy(),
                reference_pair=reference_pair,
                reference_trigger=reference_trigger,
                equity_state=equity_state,
                portfolio_state=portfolio_state,
                evaluation_start_time=evaluation_start,
            )
        )

    return BacktestRunResult(
        pair_reports=reports,
        parameters={"mode": mode, "evaluation_start": evaluation_start.isoformat(), "window_end": end.isoformat()},
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        news_mode=engine.news_feed.__class__.__name__,
    )


def _walk_forward_bars(
    *,
    engine: BacktestEngine,
    pairs: list[str],
    train_bars: int,
    test_bars: int,
    step_bars: int,
    mc_iterations: int,
    mc_seed: int,
    mc_ruin_dd: float,
) -> dict[str, object]:
    frames, errors = _load_frames(engine, pairs)
    if not frames:
        return {"window_count": 0, "errors": errors, "windows": []}

    anchor_pair = sorted(frames)[0]
    anchor_trigger = frames[anchor_pair][2].sort_index()
    if anchor_trigger.empty:
        return {"window_count": 0, "errors": errors, "windows": []}

    min_train = max(train_bars, engine.warmup_bars + engine.max_hold_bars + 1)
    cursor = min_train
    windows: list[dict[str, object]] = []
    idx = 1
    while cursor + test_bars < len(anchor_trigger):
        train_start_idx = max(0, cursor - train_bars)
        train_start = anchor_trigger.index[train_start_idx]
        test_start = anchor_trigger.index[cursor]
        test_end = anchor_trigger.index[min(cursor + test_bars, len(anchor_trigger) - 1)]

        train_result = _run_segment_with_context(
            engine,
            frames,
            evaluation_start=train_start,
            end=test_start,
            mode="bar_wf_train",
        )
        test_result = _run_segment_with_context(
            engine,
            frames,
            evaluation_start=test_start,
            end=test_end,
            mode="bar_wf_test",
        )
        test_r = [float(trade.r_multiple) for trade in test_result.trades]
        windows.append(
            {
                "window_index": idx,
                "train_start": train_start.isoformat(),
                "train_end": test_start.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "train_metrics": _compact_metrics(train_result),
                "test_metrics": _compact_metrics(test_result),
                "test_monte_carlo": run_monte_carlo(
                    test_r,
                    MonteCarloSettings(iterations=mc_iterations, seed=mc_seed + idx, ruin_drawdown_r=mc_ruin_dd),
                ),
            }
        )
        idx += 1
        cursor += max(1, step_bars)

    test_metrics = [window["test_metrics"] for window in windows]
    return {
        "window_count": len(windows),
        "errors": errors,
        "summary": {
            "test_avg_pf": _avg_metric(test_metrics, "profit_factor"),
            "test_avg_r": _avg_metric(test_metrics, "avg_r"),
            "test_avg_drawdown_r": _avg_metric(test_metrics, "max_drawdown_r"),
            "test_total_trades": sum(int(row.get("trades", 0)) for row in test_metrics),
        },
        "windows": windows,
    }


def _avg_metric(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return mean(values) if values else 0.0


def _score_candidate(row: dict[str, object]) -> float:
    metrics = row["metrics"]
    mc = row["monte_carlo"]
    trades = float(metrics.get("trades", 0))
    pf = min(3.0, float(metrics.get("profit_factor", 0.0)))
    avg_r = float(metrics.get("avg_r", 0.0))
    dd = float(metrics.get("max_drawdown_r", 0.0))
    positive_prob = float((mc.get("positive_terminal_probability", 0.0) if isinstance(mc, dict) else 0.0))
    trade_penalty = 0.25 if trades < 10 else 0.0
    return round((pf - 1.0) + (avg_r * 4.0) + positive_prob - (dd * 0.05) - trade_penalty, 6)


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 999.0 if value > 0 else -999.0
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Tier2/Tier3 full-fidelity calibration diagnostics.")
    parser.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY")
    parser.add_argument("--history-limits", default="1200")
    parser.add_argument("--evaluation-steps", default="1,2,3")
    parser.add_argument("--scenarios", default="baseline,exit_v2,tier2_meta_052,tier2_meta_055,tier2_meta_058,tier3_c045,tier3_c055")
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--warmup-bars", type=int, default=120)
    parser.add_argument("--ltf", default=None)
    parser.add_argument("--htf", default=None)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--mc-iterations", type=int, default=500)
    parser.add_argument("--mc-seed", type=int, default=42)
    parser.add_argument("--mc-ruin-dd", type=float, default=10.0)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--bar-walk-forward", action="store_true")
    parser.add_argument("--wf-train-days", type=int, default=20)
    parser.add_argument("--wf-test-days", type=int, default=7)
    parser.add_argument("--wf-step-days", type=int, default=7)
    parser.add_argument("--wf-train-bars", type=int, default=600)
    parser.add_argument("--wf-test-bars", type=int, default=240)
    parser.add_argument("--wf-step-bars", type=int, default=240)
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for interactive runs; 0 means no cap.")
    parser.add_argument("--no-snapshot-cache", action="store_true")
    parser.add_argument("--snapshot-cache-size", type=int, default=None)
    parser.add_argument("--output", default="reports/tier_calibration.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    pairs = _clean_pairs(args.pairs)
    history_limits = _parse_int_csv(args.history_limits)
    evaluation_steps = _parse_int_csv(args.evaluation_steps)
    scenario_names = _parse_csv(args.scenarios)
    ltf = (args.ltf or base_settings.ltf_timeframe).upper()
    htf = (args.htf or base_settings.htf_timeframe).upper()
    trigger = (args.trigger or base_settings.trigger_timeframe).upper()
    output_path = Path(args.output)
    snapshot_settings = SnapshotCacheSettings(
        enabled=not args.no_snapshot_cache and base_settings.enable_backtest_snapshot_cache,
        max_entries=args.snapshot_cache_size or base_settings.backtest_snapshot_cache_max_entries,
    )
    shared_snapshot_cache = SnapshotCache(snapshot_settings)

    unknown = [name for name in scenario_names if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")

    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "pairs": pairs,
        "history_limits": history_limits,
        "evaluation_steps": evaluation_steps,
        "scenarios": scenario_names,
        "runs": [],
        "ranking": [],
        "snapshot_cache": shared_snapshot_cache.stats(),
    }

    run_count = 0
    for history_limit in history_limits:
        for evaluation_step in evaluation_steps:
            for scenario_name in scenario_names:
                if args.max_runs and run_count >= args.max_runs:
                    break
                scenario = SCENARIOS[scenario_name]
                settings = _scenario_settings(base_settings, scenario)
                engine = _build_engine(
                    settings=settings,
                    history_limit=history_limit,
                    evaluation_step=evaluation_step,
                    max_hold_bars=args.max_hold_bars,
                    warmup_bars=args.warmup_bars,
                    cache_only=args.cache_only,
                    ltf=ltf,
                    htf=htf,
                    trigger=trigger,
                    snapshot_cache=shared_snapshot_cache,
                )
                result = engine.run(pairs)
                row = _result_payload(
                    result=result,
                    scenario=scenario,
                    history_limit=history_limit,
                    evaluation_step=evaluation_step,
                    mc_iterations=args.mc_iterations,
                    mc_seed=args.mc_seed + run_count,
                    mc_ruin_dd=args.mc_ruin_dd,
                )
                if args.walk_forward:
                    row["walk_forward"] = _walk_forward_days(
                        engine=engine,
                        pairs=pairs,
                        train_days=args.wf_train_days,
                        test_days=args.wf_test_days,
                        step_days=args.wf_step_days,
                        mc_iterations=args.mc_iterations,
                        mc_seed=args.mc_seed + run_count + 1000,
                        mc_ruin_dd=args.mc_ruin_dd,
                    )
                if args.bar_walk_forward:
                    row["bar_walk_forward"] = _walk_forward_bars(
                        engine=engine,
                        pairs=pairs,
                        train_bars=args.wf_train_bars,
                        test_bars=args.wf_test_bars,
                        step_bars=args.wf_step_bars,
                        mc_iterations=args.mc_iterations,
                        mc_seed=args.mc_seed + run_count + 2000,
                        mc_ruin_dd=args.mc_ruin_dd,
                    )
                row["candidate_score"] = _score_candidate(row)
                report["runs"].append(row)
                run_count += 1

                report["ranking"] = sorted(
                    [
                        {
                            "scenario": item["scenario"],
                            "history_limit": item["history_limit"],
                            "evaluation_step": item["evaluation_step"],
                            "candidate_score": item["candidate_score"],
                            "metrics": item["metrics"],
                        }
                        for item in report["runs"]
                    ],
                    key=lambda item: float(item["candidate_score"]),
                    reverse=True,
                )
                report["snapshot_cache"] = shared_snapshot_cache.stats()
                _write_report(output_path, report)
                print(
                    "{scenario} | bars={bars} step={step} trades={trades} pf={pf:.2f} avg_r={avg_r:.3f} dd={dd:.2f} score={score:.3f}".format(
                        scenario=scenario.name,
                        bars=history_limit,
                        step=evaluation_step,
                        trades=int(row["metrics"].get("trades", 0)),
                        pf=float(row["metrics"].get("profit_factor", 0.0)),
                        avg_r=float(row["metrics"].get("avg_r", 0.0)),
                        dd=float(row["metrics"].get("max_drawdown_r", 0.0)),
                        score=float(row["candidate_score"]),
                    ),
                    flush=True,
                )
            if args.max_runs and run_count >= args.max_runs:
                break
        if args.max_runs and run_count >= args.max_runs:
            break

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["snapshot_cache"] = shared_snapshot_cache.stats()
    _write_report(output_path, report)
    print(f"\nSaved calibration report: {output_path}")


if __name__ == "__main__":
    main()
