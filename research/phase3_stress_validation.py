from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from analytics.monte_carlo import MonteCarloSettings, run_monte_carlo
from backtest.engine import BacktestAccountSettings, BacktestEngine, BacktestRunResult, account_money_stats, expectancy_stats
from backtest.execution import RealisticExecutionSettings
from backtest.news import NeutralNewsFeed
from backtest.snapshot_cache import SnapshotCache, SnapshotCacheSettings
from backtest.trade_cache import BacktestTradeCache, TradeCacheSettings
from backtest_runner import (
    build_atr_risk_settings,
    build_equity_protection_settings,
    build_exit_settings,
    build_meta_label_settings,
    build_portfolio_layer_settings,
    build_signal_engine,
    build_sizing_settings,
    build_smc_research_feature_settings,
)
from config import Settings
from data.market_data import MarketDataCacheConfig, MarketDataClient
from execution.news import NewsFilter


VALIDATED_EXPANSION_PAIR_PROFILES: dict[str, dict[str, object]] = {
    "EURUSD": {
        "min_score": 80,
        "evaluation_step": 2,
        "session_windows_utc": "07-16",
        "regime_blocklist": "TREND",
    },
    "EURJPY": {
        "min_score": 78,
        "evaluation_step": 3,
        "session_windows_utc": "07-16",
        "regime_blocklist": "TREND",
    },
    "CADJPY": {
        "min_score": 80,
        "evaluation_step": 3,
        "session_windows_utc": "07-16",
        "regime_blocklist": "TREND",
    },
}


STRICT_LTF_STEP3_PAIR_PROFILES: dict[str, dict[str, object]] = {
    pair: {**profile, "evaluation_step": 3}
    for pair, profile in VALIDATED_EXPANSION_PAIR_PROFILES.items()
}


@dataclass(frozen=True)
class StressPreset:
    name: str
    description: str
    execution_settings: RealisticExecutionSettings


def stress_presets(seed: int | None) -> dict[str, StressPreset]:
    return {
        "ideal": StressPreset(
            name="ideal",
            description="No spread, slippage, delay, or partial-fill friction.",
            execution_settings=RealisticExecutionSettings(enabled=False, random_seed=seed),
        ),
        "mild": StressPreset(
            name="mild",
            description="Light live-friction model for liquid sessions.",
            execution_settings=RealisticExecutionSettings(
                enabled=True,
                spread_default_pips=0.7,
                spread_by_pair={"EURUSD": 0.6, "EURJPY": 1.1, "CADJPY": 1.3, "USDJPY": 0.8},
                slippage_mode="random",
                max_slippage_pips=0.25,
                execution_delay_bars=0,
                partial_fill_probability=0.98,
                partial_fill_min_ratio=0.85,
                limit_touch_tolerance_pips=0.3,
                apply_spread_to_limit=True,
                random_seed=seed,
            ),
        ),
        "moderate": StressPreset(
            name="moderate",
            description="Moderate execution stress used for live-candidate validation.",
            execution_settings=RealisticExecutionSettings(
                enabled=True,
                spread_default_pips=1.0,
                spread_by_pair={"EURUSD": 0.8, "EURJPY": 1.5, "CADJPY": 1.8, "USDJPY": 1.0},
                slippage_mode="random",
                max_slippage_pips=0.5,
                execution_delay_bars=1,
                partial_fill_probability=0.95,
                partial_fill_min_ratio=0.7,
                limit_touch_tolerance_pips=0.2,
                apply_spread_to_limit=True,
                random_seed=seed,
            ),
        ),
        "harsh": StressPreset(
            name="harsh",
            description="Harsh spread/slippage/delay model for robustness stress testing.",
            execution_settings=RealisticExecutionSettings(
                enabled=True,
                spread_default_pips=1.5,
                spread_by_pair={"EURUSD": 1.2, "EURJPY": 2.2, "CADJPY": 2.6, "USDJPY": 1.6},
                slippage_mode="volatility",
                max_slippage_pips=1.0,
                execution_delay_bars=2,
                partial_fill_probability=0.85,
                partial_fill_min_ratio=0.5,
                limit_touch_tolerance_pips=0.0,
                apply_spread_to_limit=True,
                random_seed=seed,
            ),
        ),
    }


def parse_pairs(raw: str) -> list[str]:
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


def parse_names(raw: str, available: Iterable[str]) -> list[str]:
    known = list(available)
    text = (raw or "").strip().lower()
    if text == "all":
        return known
    names: list[str] = []
    for item in text.split(","):
        name = item.strip()
        if name and name not in names:
            names.append(name)
    return names


def safe_float(value: object, default: float = 0.0) -> float:
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


def safe_settings_payload(settings: Settings) -> dict[str, object]:
    payload = asdict(settings)
    for key in (
        "telegram_bot_token",
        "telegram_chat_id",
        "mt5_password",
        "backtest_trade_cache_dir",
        "enable_backtest_trade_cache",
    ):
        payload.pop(key, None)
    return payload


def account_settings(args: argparse.Namespace) -> BacktestAccountSettings:
    return BacktestAccountSettings(
        enabled=not args.disable_account_report,
        starting_balance=args.account_balance,
        risk_per_trade=args.risk_per_trade,
        currency=args.account_currency,
    ).sanitized()


def trade_cache_settings(settings: Settings, args: argparse.Namespace) -> TradeCacheSettings:
    return TradeCacheSettings(
        enabled=settings.enable_backtest_trade_cache or bool(args.trade_cache),
        cache_dir=args.trade_cache_dir or settings.backtest_trade_cache_dir,
        version=settings.backtest_trade_cache_version,
    ).sanitized()


def snapshot_cache_settings(settings: Settings, args: argparse.Namespace) -> SnapshotCacheSettings:
    return SnapshotCacheSettings(
        enabled=not args.no_snapshot_cache and settings.enable_backtest_snapshot_cache,
        max_entries=max(1000, int(args.snapshot_cache_size)),
    ).sanitized()


def load_pair_profiles(raw: str | None) -> dict[str, object] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --pair-profiles-json: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--pair-profiles-json must be a JSON object")
    return payload


def apply_signal_profile(settings: Settings, args: argparse.Namespace) -> Settings:
    updates: dict[str, object] = {}
    if args.signal_profile == "strict_ltf_only":
        updates.update(
            {
                "enable_strict_ltf_direction_gate": True,
                "enable_market_fallback_entry": True,
                "market_fallback_min_trigger_strength": 0,
                "market_fallback_require_displacement": False,
                "enable_pip_aware_liquidity": False,
                "enable_structure_quality_scoring": False,
                "structure_quality_replaces_raw_structure_score": False,
            }
        )
    elif args.signal_profile != "current":
        raise SystemExit(f"Unknown signal profile: {args.signal_profile}")

    pair_profiles = None
    if args.pair_profile_preset == "validated_expansion_v1":
        pair_profiles = VALIDATED_EXPANSION_PAIR_PROFILES
    elif args.pair_profile_preset == "strict_ltf_step3_v1":
        pair_profiles = STRICT_LTF_STEP3_PAIR_PROFILES
    elif args.pair_profile_preset != "current":
        raise SystemExit(f"Unknown pair profile preset: {args.pair_profile_preset}")
    pair_profiles = load_pair_profiles(args.pair_profiles_json) or pair_profiles
    if pair_profiles is not None:
        updates.update(
            {
                "enable_pair_profiles": True,
                "pair_profiles": pair_profiles,
                "pair_profiles_backtest_only": True,
                "allow_live_pair_profiles": False,
            }
        )

    return replace(settings, **updates) if updates else settings


def resolve_end_time(settings: Settings, args: argparse.Namespace) -> str | None:
    raw = str(args.end_time or settings.backtest_end_time or "").strip()
    return raw or None


def build_engine(
    *,
    settings: Settings,
    preset: StressPreset,
    args: argparse.Namespace,
    pairs: list[str],
    account: BacktestAccountSettings,
    snapshot_cache: SnapshotCache,
    snapshot_settings: SnapshotCacheSettings,
) -> BacktestEngine:
    cache_mode = "cache_only" if args.cache_only else settings.market_data_cache_mode
    if args.refresh_cache:
        cache_mode = "refresh"
    market_data = MarketDataClient(
        history_limit=max(settings.history_limit, args.history_limit),
        data_source=settings.data_source,
        mt5_login=settings.mt5_login,
        mt5_password=settings.mt5_password,
        mt5_server=settings.mt5_server,
        mt5_path=settings.mt5_path,
        cache_config=MarketDataCacheConfig(
            enabled=settings.market_data_cache_enabled,
            cache_dir=args.cache_dir or settings.market_data_cache_dir,
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
        htf=(args.htf or settings.htf_timeframe).upper(),
        ltf=(args.ltf or settings.ltf_timeframe).upper(),
        trigger=(args.trigger or settings.trigger_timeframe).upper(),
        enable_shadow_scoring=True,
        enable_mitigation_entry=settings.enable_mitigation_entry,
    )
    return BacktestEngine(
        market_data=market_data,
        signal_engine=signal_engine,
        history_limit=max(150, int(args.history_limit)),
        max_hold_bars=max(1, int(args.max_hold_bars)),
        warmup_bars=max(80, int(args.warmup_bars)),
        evaluation_step=max(1, int(args.evaluation_step)),
        news_feed=NeutralNewsFeed(),
        execution_settings=preset.execution_settings,
        atr_risk_settings=build_atr_risk_settings(settings),
        equity_protection_settings=build_equity_protection_settings(settings),
        exit_settings=build_exit_settings(settings),
        sizing_settings=build_sizing_settings(settings),
        meta_label_settings=build_meta_label_settings(settings),
        portfolio_layer_settings=build_portfolio_layer_settings(settings),
        snapshot_cache_settings=snapshot_settings,
        snapshot_cache=snapshot_cache,
        smc_research_feature_settings=build_smc_research_feature_settings(settings),
        account_settings=account,
        end_time=resolve_end_time(settings, args),
    )


def cache_key_payload(
    *,
    settings: Settings,
    preset: StressPreset,
    args: argparse.Namespace,
    pairs: list[str],
    account: BacktestAccountSettings,
) -> dict[str, object]:
    cache_mode = "cache_only" if args.cache_only else settings.market_data_cache_mode
    if args.refresh_cache:
        cache_mode = "refresh"
    return {
        "runner": "research.phase3_stress_validation",
        "stress_preset": preset.name,
        "stress_settings": asdict(preset.execution_settings.sanitized()),
        "signal_profile": args.signal_profile,
        "pair_profile_preset": args.pair_profile_preset,
        "pairs": pairs,
        "timeframes": {
            "ltf": (args.ltf or settings.ltf_timeframe).upper(),
            "htf": (args.htf or settings.htf_timeframe).upper(),
            "trigger": (args.trigger or settings.trigger_timeframe).upper(),
        },
        "history_limit": max(150, int(args.history_limit)),
        "max_hold_bars": max(1, int(args.max_hold_bars)),
        "warmup_bars": max(80, int(args.warmup_bars)),
        "evaluation_step": max(1, int(args.evaluation_step)),
        "end_time": resolve_end_time(settings, args),
        "cache_mode": cache_mode,
        "account": asdict(account),
        "settings": safe_settings_payload(settings),
        "exit_settings": asdict(build_exit_settings(settings).sanitized()),
        "sizing_settings": asdict(build_sizing_settings(settings)),
        "meta_label_settings": asdict(build_meta_label_settings(settings)),
        "portfolio_layer_settings": asdict(build_portfolio_layer_settings(settings)),
    }


def compact_metrics(result: BacktestRunResult) -> dict[str, object]:
    metrics = result.overall_metrics()
    keys = (
        "trades",
        "wins",
        "losses",
        "breakeven",
        "win_rate",
        "avg_r",
        "expectancy_r",
        "avg_win_r",
        "avg_loss_r",
        "payoff_ratio",
        "profit_factor",
        "max_drawdown_r",
        "fill_rate",
        "avg_spread_pips",
        "avg_slippage_pips",
        "total_spread_cost_r",
        "total_slippage_cost_r",
        "avg_delay_cost_r",
        "starting_balance",
        "risk_per_trade",
        "net_pnl_usd",
        "final_balance_usd",
        "max_drawdown_usd",
        "max_drawdown_pct",
        "roi_pct",
        "expectancy_usd",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def pair_metrics(result: BacktestRunResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in result.pair_rows():
        rows.append(
            {
                key: row.get(key)
                for key in (
                    "pair",
                    "trades",
                    "win_rate",
                    "avg_r",
                    "expectancy_r",
                    "profit_factor",
                    "max_drawdown_r",
                    "net_pnl_usd",
                    "final_balance_usd",
                    "max_drawdown_usd",
                    "fill_rate",
                    "avg_spread_pips",
                    "avg_slippage_pips",
                    "total_spread_cost_r",
                    "total_slippage_cost_r",
                    "avg_delay_cost_r",
                    "rejections",
                )
                if key in row
            }
        )
    return rows


def regime_metrics(result: BacktestRunResult, account: BacktestAccountSettings) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    for trade in result.trades:
        key = (trade.regime_label or "UNKNOWN").upper()
        grouped.setdefault(key, []).append(float(trade.r_multiple))
    payload: dict[str, dict[str, object]] = {}
    for regime, values in sorted(grouped.items()):
        expectancy = expectancy_stats(values)
        gross_profit = sum(value for value in values if value > 0)
        gross_loss = abs(sum(value for value in values if value < 0))
        payload[regime] = {
            "trades": len(values),
            "win_rate": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
            "avg_r": sum(values) / len(values) if values else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
            **expectancy,
            **account_money_stats(values, account),
        }
    return payload


def monte_carlo_payload(result: BacktestRunResult, args: argparse.Namespace, account: BacktestAccountSettings) -> dict[str, object]:
    settings = MonteCarloSettings(
        iterations=max(100, int(args.mc_iterations)),
        seed=int(args.mc_seed),
        ruin_drawdown_r=max(0.1, float(args.mc_ruin_dd)),
    )
    payload = run_monte_carlo([trade.r_multiple for trade in result.trades], settings)
    terminal = payload.get("terminal_r")
    drawdown = payload.get("max_drawdown_r")
    if account.enabled and isinstance(terminal, dict):
        payload["terminal_usd"] = {
            key: round(float(value) * account.risk_per_trade, 2)
            for key, value in terminal.items()
        }
    if account.enabled and isinstance(drawdown, dict):
        payload["max_drawdown_usd"] = {
            key: round(float(value) * account.risk_per_trade, 2)
            for key, value in drawdown.items()
        }
    return payload


def candidate_score(metrics: dict[str, object], mc: dict[str, object]) -> float:
    pf = min(5.0, safe_float(metrics.get("profit_factor")))
    avg_r = safe_float(metrics.get("avg_r"))
    dd_r = safe_float(metrics.get("max_drawdown_r"))
    trades = safe_float(metrics.get("trades"))
    terminal = mc.get("terminal_r", {}) if isinstance(mc.get("terminal_r"), dict) else {}
    p05 = safe_float(terminal.get("p05")) if isinstance(terminal, dict) else 0.0
    positive = safe_float(mc.get("positive_terminal_probability"))
    trade_penalty = 0.0 if trades >= 20 else (20.0 - trades) * 0.15
    return round(pf + avg_r * 4.0 + p05 * 0.15 + positive - dd_r * 0.25 - trade_penalty, 6)


def result_payload(
    *,
    preset: StressPreset,
    result: BacktestRunResult,
    account: BacktestAccountSettings,
    args: argparse.Namespace,
) -> dict[str, object]:
    metrics = compact_metrics(result)
    mc = monte_carlo_payload(result, args, account)
    return {
        "scenario": preset.name,
        "description": preset.description,
        "metrics": metrics,
        "candidate_score": candidate_score(metrics, mc),
        "monte_carlo": mc,
        "pairs": pair_metrics(result),
        "regimes": regime_metrics(result, account),
        "execution_settings": asdict(preset.execution_settings.sanitized()),
        "snapshot_cache": result.parameters.get("snapshot_cache_stats", {}),
        "trade_cache": {
            "status": result.parameters.get("trade_cache_status"),
            "key": result.parameters.get("trade_cache_key"),
            "path": result.parameters.get("trade_cache_path"),
        },
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
    }


def write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, default=str, allow_nan=False), encoding="utf-8")


def print_line(payload: dict[str, object]) -> None:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    mc = payload.get("monte_carlo", {}) if isinstance(payload.get("monte_carlo"), dict) else {}
    terminal = mc.get("terminal_r", {}) if isinstance(mc.get("terminal_r"), dict) else {}
    print(
        "{name:10s} trades={trades:>3} pf={pf:>5} avg_r={avg_r:>7} net=${net:>8} dd=${dd:>7} "
        "mc_p05={p05:>7} pos={pos:>6} score={score}".format(
            name=str(payload.get("scenario")),
            trades=int(safe_float(metrics.get("trades"))),
            pf=f"{safe_float(metrics.get('profit_factor')):.2f}",
            avg_r=f"{safe_float(metrics.get('avg_r')):.3f}",
            net=f"{safe_float(metrics.get('net_pnl_usd')):+.2f}",
            dd=f"{safe_float(metrics.get('max_drawdown_usd')):.2f}",
            p05=f"{safe_float(terminal.get('p05')):.3f}",
            pos=f"{safe_float(mc.get('positive_terminal_probability')):.1%}",
            score=f"{safe_float(payload.get('candidate_score')):.3f}",
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 fast stress validation runner.")
    parser.add_argument("--pairs", default="EURUSD,EURJPY,CADJPY")
    parser.add_argument("--history-limit", type=int, default=3000)
    parser.add_argument("--evaluation-step", type=int, default=3)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    parser.add_argument("--warmup-bars", type=int, default=120)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--ltf", default=None)
    parser.add_argument("--htf", default=None)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--signal-profile", default="current", choices=("current", "strict_ltf_only"))
    parser.add_argument(
        "--pair-profile-preset",
        default="current",
        choices=("current", "validated_expansion_v1", "strict_ltf_step3_v1"),
    )
    parser.add_argument("--pair-profiles-json", default=None)
    parser.add_argument("--stress-presets", default="ideal,moderate,harsh")
    parser.add_argument("--account-balance", type=float, default=1000.0)
    parser.add_argument("--risk-per-trade", type=float, default=50.0)
    parser.add_argument("--account-currency", default="USD")
    parser.add_argument("--disable-account-report", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-snapshot-cache", action="store_true")
    parser.add_argument("--snapshot-cache-size", type=int, default=250000)
    parser.add_argument("--trade-cache", action="store_true")
    parser.add_argument("--refresh-trade-cache", action="store_true")
    parser.add_argument("--trade-cache-dir", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--mc-iterations", type=int, default=5000)
    parser.add_argument("--mc-seed", type=int, default=42)
    parser.add_argument("--mc-ruin-dd", type=float, default=10.0)
    parser.add_argument("--export-run-artifacts", action="store_true")
    parser.add_argument("--output", default="reports/phase3_stress_validation.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = apply_signal_profile(base_settings, args)
    pairs = parse_pairs(args.pairs)
    account = account_settings(args)
    presets = stress_presets(args.random_seed)
    selected_names = parse_names(args.stress_presets, presets)
    missing = [name for name in selected_names if name not in presets]
    if missing:
        raise SystemExit(f"Unknown stress presets: {', '.join(missing)}")
    selected = [presets[name] for name in selected_names]
    if not selected:
        raise SystemExit("No stress presets selected")

    output = Path(args.output)
    trade_cache = BacktestTradeCache(trade_cache_settings(settings, args))
    snapshot_settings = snapshot_cache_settings(settings, args)
    snapshot_cache = SnapshotCache(snapshot_settings)
    report: dict[str, object] = {
        "runner": "research.phase3_stress_validation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "settings": {
            "pairs": pairs,
            "history_limit": max(150, int(args.history_limit)),
            "evaluation_step": max(1, int(args.evaluation_step)),
            "signal_profile": args.signal_profile,
            "pair_profile_preset": args.pair_profile_preset,
            "pair_profiles": settings.pair_profiles if settings.enable_pair_profiles else {},
            "stress_presets": [preset.name for preset in selected],
            "account": asdict(account),
            "monte_carlo": {
                "iterations": max(100, int(args.mc_iterations)),
                "seed": int(args.mc_seed),
                "ruin_drawdown_r": max(0.1, float(args.mc_ruin_dd)),
            },
            "cache_only": bool(args.cache_only),
            "end_time": resolve_end_time(settings, args),
            "trade_cache": asdict(trade_cache.settings),
            "snapshot_cache": asdict(snapshot_settings),
        },
        "runs": [],
        "ranking": [],
        "snapshot_cache": snapshot_cache.stats(),
    }

    print(
        "Phase 3 stress validation | pairs={pairs} bars={bars} step={step} profile={profile} presets={presets}".format(
            pairs=",".join(pairs),
            bars=max(150, int(args.history_limit)),
            step=max(1, int(args.evaluation_step)),
            profile=args.signal_profile,
            presets=",".join(preset.name for preset in selected),
        ),
        flush=True,
    )

    for preset in selected:
        print(f"\n=== {preset.name} ===", flush=True)
        engine = build_engine(
            settings=settings,
            preset=preset,
            args=args,
            pairs=pairs,
            account=account,
            snapshot_cache=snapshot_cache,
            snapshot_settings=snapshot_settings,
        )
        key_payload = cache_key_payload(
            settings=settings,
            preset=preset,
            args=args,
            pairs=pairs,
            account=account,
        )
        cache_key = trade_cache.build_key(key_payload)
        result = None
        if trade_cache.enabled and not args.refresh_trade_cache:
            result = trade_cache.load(cache_key)
            if result is not None:
                print(f"Trade cache: HIT {result.parameters.get('trade_cache_path')}", flush=True)
        if result is None:
            result = engine.run(pairs)
            if trade_cache.enabled:
                path = trade_cache.store(
                    cache_key,
                    key_payload,
                    result,
                    metadata={"runner": "research.phase3_stress_validation", "stress_preset": preset.name},
                )
                if path is not None:
                    print(f"Trade cache: STORED {path}", flush=True)
        if args.export_run_artifacts:
            artifact_dir = output.with_suffix("").parent / output.with_suffix("").name / preset.name
            result.export(artifact_dir)

        payload = result_payload(preset=preset, result=result, account=account, args=args)
        runs = report.get("runs")
        if isinstance(runs, list):
            runs.append(payload)
        report["snapshot_cache"] = snapshot_cache.stats()
        report["ranking"] = sorted(
            [
                {
                    "scenario": item.get("scenario"),
                    "candidate_score": item.get("candidate_score"),
                    "metrics": item.get("metrics"),
                    "monte_carlo": item.get("monte_carlo"),
                }
                for item in report.get("runs", [])
                if isinstance(item, dict)
            ],
            key=lambda item: safe_float(item.get("candidate_score")),
            reverse=True,
        )
        print_line(payload)
        write_report(output, report)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["snapshot_cache"] = snapshot_cache.stats()
    write_report(output, report)
    print(f"\nSaved Phase 3 stress report: {output}", flush=True)
    print("\nRANKING", flush=True)
    for row in report.get("ranking", []):
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        mc = row.get("monte_carlo", {}) if isinstance(row.get("monte_carlo"), dict) else {}
        terminal = mc.get("terminal_r", {}) if isinstance(mc.get("terminal_r"), dict) else {}
        print(
            "{name:10s} score={score:.3f} pf={pf:.2f} avg_r={avg_r:.3f} net=${net:+.2f} dd=${dd:.2f} mc_p05={p05:.3f}".format(
                name=str(row.get("scenario")),
                score=safe_float(row.get("candidate_score")),
                pf=safe_float(metrics.get("profit_factor")),
                avg_r=safe_float(metrics.get("avg_r")),
                net=safe_float(metrics.get("net_pnl_usd")),
                dd=safe_float(metrics.get("max_drawdown_usd")),
                p05=safe_float(terminal.get("p05")) if isinstance(terminal, dict) else 0.0,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
