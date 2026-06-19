from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from backtest.engine import BacktestAccountSettings, BacktestEngine, BacktestRunResult, BacktestTrade, account_money_stats, expectancy_stats
from backtest.exit_engine import AdaptiveExitSettings, ExitProfile, PartialRTarget, default_regime_profiles
from backtest.news import NeutralNewsFeed
from backtest.snapshot_cache import SnapshotCache, SnapshotCacheSettings
from backtest.trade_cache import BacktestTradeCache, TradeCacheSettings
from backtest_runner import (
    build_atr_risk_settings,
    build_equity_protection_settings,
    build_execution_settings,
    build_meta_label_settings,
    build_portfolio_layer_settings,
    build_signal_engine,
    build_sizing_settings,
)
from config import Settings
from data.market_data import MarketDataCacheConfig, MarketDataClient
from execution.news import NewsFilter


@dataclass(frozen=True)
class ExitCalibrationScenario:
    name: str
    description: str
    exit_settings: AdaptiveExitSettings


def _pt(r_multiple: float, fraction: float) -> PartialRTarget:
    return PartialRTarget(r_multiple=r_multiple, fraction=fraction).sanitized()


def _profile(
    *,
    target_rr: float,
    partials: tuple[tuple[float, float], ...],
    break_even_r: float,
    trailing_enabled: bool,
    trailing_start_r: float,
    trailing_lookback_bars: int,
    time_stop_bars: int,
) -> ExitProfile:
    return ExitProfile(
        target_rr=target_rr,
        partial_targets=tuple(_pt(r, fraction) for r, fraction in partials),
        break_even_r=break_even_r,
        trailing_enabled=trailing_enabled,
        trailing_start_r=trailing_start_r,
        trailing_lookback_bars=trailing_lookback_bars,
        time_stop_bars=time_stop_bars,
    ).sanitized()


def _profiles(
    *,
    trend: ExitProfile,
    expansion: ExitProfile | None = None,
    range_: ExitProfile,
    contraction: ExitProfile | None = None,
    neutral: ExitProfile | None = None,
) -> dict[str, ExitProfile]:
    return {
        "trend": trend.sanitized(),
        "expansion": (expansion or trend).sanitized(),
        "range": range_.sanitized(),
        "contraction": (contraction or range_).sanitized(),
        "neutral": (neutral or range_).sanitized(),
    }


def build_exit_scenarios() -> dict[str, ExitCalibrationScenario]:
    default_profiles = default_regime_profiles()
    balanced_trend = _profile(
        target_rr=2.4,
        partials=((1.0, 0.35), (1.8, 0.25)),
        break_even_r=1.1,
        trailing_enabled=True,
        trailing_start_r=1.5,
        trailing_lookback_bars=8,
        time_stop_bars=60,
    )
    balanced_range = _profile(
        target_rr=1.35,
        partials=((0.8, 0.55),),
        break_even_r=0.8,
        trailing_enabled=True,
        trailing_start_r=1.1,
        trailing_lookback_bars=5,
        time_stop_bars=24,
    )
    fast_range = _profile(
        target_rr=1.15,
        partials=((0.7, 0.65),),
        break_even_r=0.65,
        trailing_enabled=False,
        trailing_start_r=99.0,
        trailing_lookback_bars=4,
        time_stop_bars=18,
    )
    soft_range = _profile(
        target_rr=1.45,
        partials=((0.9, 0.45),),
        break_even_r=0.95,
        trailing_enabled=True,
        trailing_start_r=1.25,
        trailing_lookback_bars=6,
        time_stop_bars=30,
    )
    runner_trend = _profile(
        target_rr=3.4,
        partials=((1.2, 0.30), (2.2, 0.20)),
        break_even_r=1.35,
        trailing_enabled=True,
        trailing_start_r=1.8,
        trailing_lookback_bars=12,
        time_stop_bars=84,
    )
    runner_expansion = _profile(
        target_rr=3.8,
        partials=((1.2, 0.25), (2.4, 0.20)),
        break_even_r=1.4,
        trailing_enabled=True,
        trailing_start_r=1.9,
        trailing_lookback_bars=14,
        time_stop_bars=96,
    )
    late_be_trend = _profile(
        target_rr=3.0,
        partials=((1.5, 0.35),),
        break_even_r=1.6,
        trailing_enabled=True,
        trailing_start_r=2.0,
        trailing_lookback_bars=10,
        time_stop_bars=72,
    )
    no_partial_trend = _profile(
        target_rr=2.8,
        partials=(),
        break_even_r=1.4,
        trailing_enabled=True,
        trailing_start_r=1.8,
        trailing_lookback_bars=10,
        time_stop_bars=72,
    )
    no_partial_range = _profile(
        target_rr=1.35,
        partials=(),
        break_even_r=0.9,
        trailing_enabled=False,
        trailing_start_r=99.0,
        trailing_lookback_bars=5,
        time_stop_bars=24,
    )

    scenarios = {
        "baseline_legacy": ExitCalibrationScenario(
            name="baseline_legacy",
            description="Current signal-engine exit plan, adaptive exit disabled.",
            exit_settings=AdaptiveExitSettings(enabled=False),
        ),
        "exit_default": ExitCalibrationScenario(
            name="exit_default",
            description="Existing regime-aware exit-v2 defaults.",
            exit_settings=AdaptiveExitSettings(enabled=True, regime_profiles=default_profiles),
        ),
        "hybrid_balanced": ExitCalibrationScenario(
            name="hybrid_balanced",
            description="Moderate trend runners and controlled range monetization.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=balanced_trend, expansion=runner_trend, range_=balanced_range),
            ),
        ),
        "range_fast": ExitCalibrationScenario(
            name="range_fast",
            description="Fast range exits, no trailing; designed to reduce low-R chop exposure.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=balanced_trend, expansion=runner_trend, range_=fast_range),
            ),
        ),
        "range_soft": ExitCalibrationScenario(
            name="range_soft",
            description="Range still quick, but less aggressive than range_fast.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=balanced_trend, expansion=runner_trend, range_=soft_range),
            ),
        ),
        "trend_runner": ExitCalibrationScenario(
            name="trend_runner",
            description="Higher trend/expansion asymmetry with smaller partials and later BE.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=runner_trend, expansion=runner_expansion, range_=balanced_range),
            ),
        ),
        "late_be_runner": ExitCalibrationScenario(
            name="late_be_runner",
            description="Avoid premature BE; lets winners breathe more before risk removal.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=late_be_trend, expansion=runner_expansion, range_=soft_range),
            ),
        ),
        "no_partials_runner": ExitCalibrationScenario(
            name="no_partials_runner",
            description="Pure full-position exits; tests whether partials are capping expectancy.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=no_partial_trend, expansion=runner_expansion, range_=no_partial_range),
            ),
        ),
        "atr_trailing": ExitCalibrationScenario(
            name="atr_trailing",
            description="Balanced exits plus ATR trailing in continuation regimes.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=balanced_trend, expansion=runner_trend, range_=balanced_range),
                atr_trailing_enabled=True,
                atr_trailing_period=14,
                atr_trailing_multiplier=1.4,
            ),
        ),
        "liquidity_trailing": ExitCalibrationScenario(
            name="liquidity_trailing",
            description="Balanced exits plus liquidity-style trailing anchors.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=balanced_trend, expansion=runner_trend, range_=balanced_range),
                liquidity_trailing_enabled=True,
                liquidity_lookback_bars=8,
                liquidity_buffer_pips=1.0,
            ),
        ),
        "volatility_rr": ExitCalibrationScenario(
            name="volatility_rr",
            description="Balanced exits with target RR adjusted by realized volatility ratio.",
            exit_settings=AdaptiveExitSettings(
                enabled=True,
                regime_profiles=_profiles(trend=balanced_trend, expansion=runner_trend, range_=balanced_range),
                volatility_rr_enabled=True,
                volatility_rr_floor=0.85,
                volatility_rr_cap=1.30,
            ),
        ),
    }
    for floor in (0.75, 0.85, 0.95):
        for cap in (1.20, 1.30, 1.40):
            for liquidity_enabled in (False, True):
                suffix = "_liq" if liquidity_enabled else ""
                name = f"vol_rr_f{int(floor * 100):03d}_c{int(cap * 100):03d}{suffix}"
                scenarios[name] = ExitCalibrationScenario(
                    name=name,
                    description=(
                        f"Volatility RR grid floor={floor:.2f} cap={cap:.2f} "
                        f"liquidity_trailing={liquidity_enabled}."
                    ),
                    exit_settings=AdaptiveExitSettings(
                        enabled=True,
                        regime_profiles=_profiles(trend=balanced_trend, expansion=runner_trend, range_=balanced_range),
                        volatility_rr_enabled=True,
                        volatility_rr_floor=floor,
                        volatility_rr_cap=cap,
                        liquidity_trailing_enabled=liquidity_enabled,
                        liquidity_lookback_bars=8,
                        liquidity_buffer_pips=1.0,
                    ),
                )
    return {name: scenario for name, scenario in scenarios.items()}


def _parse_pairs(raw: str) -> list[str]:
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _select_scenarios(raw: str, scenarios: dict[str, ExitCalibrationScenario]) -> tuple[list[ExitCalibrationScenario], list[str]]:
    if raw == "all":
        names = list(scenarios)
    else:
        names = []
        for item in _parse_csv(raw):
            if item == "volatility_grid":
                names.extend(name for name in scenarios if name.startswith("vol_rr_f"))
            else:
                names.append(item)

    selected: list[ExitCalibrationScenario] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        scenario = scenarios.get(name)
        if scenario is None:
            missing.append(name)
        else:
            selected.append(scenario)
    return selected, missing


def _safe_float(value: object, default: float = 0.0) -> float:
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
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 999.0 if value > 0 else -999.0
        return round(value, 6)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _account_settings(args: argparse.Namespace) -> BacktestAccountSettings:
    return BacktestAccountSettings(
        enabled=not args.disable_account_report,
        starting_balance=args.account_balance,
        risk_per_trade=args.risk_per_trade,
        currency=args.account_currency,
    ).sanitized()


def _safe_settings_payload(settings: Settings) -> dict[str, object]:
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


def _trade_cache_settings(base_settings: Settings, args: argparse.Namespace) -> TradeCacheSettings:
    enabled = base_settings.enable_backtest_trade_cache or bool(args.trade_cache)
    cache_dir = args.trade_cache_dir or base_settings.backtest_trade_cache_dir
    return TradeCacheSettings(
        enabled=enabled,
        cache_dir=cache_dir,
        version=base_settings.backtest_trade_cache_version,
    ).sanitized()


def _resolve_end_time(base_settings: Settings, args: argparse.Namespace) -> str | None:
    raw = str(args.end_time or base_settings.backtest_end_time or "").strip()
    return raw or None


def _scenario_cache_key_payload(
    *,
    base_settings: Settings,
    scenario: ExitCalibrationScenario,
    pairs: list[str],
    account: BacktestAccountSettings,
    args: argparse.Namespace,
    end_time: str | None,
) -> dict[str, object]:
    return {
        "runner": "research.exit_calibration",
        "scenario": scenario.name,
        "pairs": pairs,
        "history_limit": max(150, int(args.history_limit)),
        "evaluation_step": max(1, int(args.evaluation_step)),
        "max_hold_bars": max(1, int(args.max_hold_bars)),
        "warmup_bars": max(80, int(args.warmup_bars)),
        "timeframes": {
            "ltf": (args.ltf or base_settings.ltf_timeframe).upper(),
            "htf": (args.htf or base_settings.htf_timeframe).upper(),
            "trigger": (args.trigger or base_settings.trigger_timeframe).upper(),
        },
        "cache_only": bool(args.cache_only),
        "end_time": end_time,
        "account": asdict(account),
        "settings": _safe_settings_payload(base_settings),
        "exit_settings": asdict(scenario.exit_settings.sanitized()),
    }


def _build_engine(
    *,
    base_settings: Settings,
    scenario: ExitCalibrationScenario,
    history_limit: int,
    evaluation_step: int,
    max_hold_bars: int,
    warmup_bars: int,
    cache_only: bool,
    ltf: str,
    htf: str,
    trigger: str,
    snapshot_cache: SnapshotCache | None,
    snapshot_settings: SnapshotCacheSettings,
    account_settings: BacktestAccountSettings,
    end_time: str | None,
) -> BacktestEngine:
    cache_mode = "cache_only" if cache_only else base_settings.market_data_cache_mode
    market_data = MarketDataClient(
        history_limit=max(base_settings.history_limit, history_limit),
        data_source=base_settings.data_source,
        mt5_login=base_settings.mt5_login,
        mt5_password=base_settings.mt5_password,
        mt5_server=base_settings.mt5_server,
        mt5_path=base_settings.mt5_path,
        cache_config=MarketDataCacheConfig(
            enabled=base_settings.market_data_cache_enabled,
            cache_dir=base_settings.market_data_cache_dir,
            ttl_hours=base_settings.market_data_cache_ttl_hours,
            mode=cache_mode,
        ),
    )
    news_filter = NewsFilter(
        blackout_before_minutes=base_settings.news_blackout_before_minutes,
        blackout_after_minutes=base_settings.news_blackout_after_minutes,
        surprise_threshold=base_settings.news_surprise_threshold,
    )
    signal_engine = build_signal_engine(
        market_data=market_data,
        news_filter=news_filter,
        settings=base_settings,
        htf=htf,
        ltf=ltf,
        trigger=trigger,
        enable_shadow_scoring=True,
        enable_mitigation_entry=base_settings.enable_mitigation_entry,
    )
    return BacktestEngine(
        market_data=market_data,
        signal_engine=signal_engine,
        history_limit=history_limit,
        max_hold_bars=max_hold_bars,
        warmup_bars=warmup_bars,
        evaluation_step=evaluation_step,
        news_feed=NeutralNewsFeed(),
        execution_settings=build_execution_settings(base_settings),
        atr_risk_settings=build_atr_risk_settings(base_settings),
        equity_protection_settings=build_equity_protection_settings(base_settings),
        exit_settings=scenario.exit_settings,
        sizing_settings=build_sizing_settings(base_settings),
        meta_label_settings=build_meta_label_settings(base_settings),
        portfolio_layer_settings=build_portfolio_layer_settings(base_settings),
        snapshot_cache_settings=snapshot_settings,
        snapshot_cache=snapshot_cache,
        account_settings=account_settings,
        end_time=end_time,
    )


def _max_drawdown(values: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return dd


def _group_metrics(trades: Iterable[BacktestTrade], account: BacktestAccountSettings) -> dict[str, object]:
    values = [float(trade.r_multiple) for trade in trades]
    stats = expectancy_stats(values)
    wins = sum(1 for value in values if value > 0)
    losses = sum(1 for value in values if value < 0)
    gross_profit = sum(value for value in values if value > 0)
    gross_loss = abs(sum(value for value in values if value < 0))
    metrics = {
        "trades": len(values),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(values), 6) if values else 0.0,
        "avg_r": round(sum(values) / len(values), 6) if values else 0.0,
        "expectancy_r": round(float(stats["expectancy_r"]), 6),
        "avg_win_r": round(float(stats["avg_win_r"]), 6),
        "avg_loss_r": round(float(stats["avg_loss_r"]), 6),
        "payoff_ratio": gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "max_drawdown_r": round(_max_drawdown(values), 6),
    }
    metrics.update(account_money_stats(values, account))
    return metrics


def _pair_metrics(result: BacktestRunResult) -> list[dict[str, object]]:
    return result.pair_rows()


def _regime_metrics(result: BacktestRunResult, account: BacktestAccountSettings) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[BacktestTrade]] = {}
    for trade in result.trades:
        grouped.setdefault((trade.regime_label or "UNKNOWN").upper(), []).append(trade)
    return {regime: _group_metrics(trades, account) for regime, trades in sorted(grouped.items())}


def _compact_metrics(result: BacktestRunResult) -> dict[str, object]:
    metrics = result.overall_metrics()
    keys = (
        "trades",
        "wins",
        "losses",
        "win_rate",
        "profit_factor",
        "avg_r",
        "expectancy_r",
        "avg_win_r",
        "avg_loss_r",
        "payoff_ratio",
        "max_drawdown_r",
        "avg_score",
        "avg_bars_held",
        "partial_exits",
        "break_even_activations",
        "trailing_activations",
        "atr_trailing_activations",
        "liquidity_trailing_activations",
        "adaptive_exit_trades",
        "avg_exit_target_rr",
        "fill_rate",
        "acceptance_rate",
        "starting_balance",
        "risk_per_trade",
        "risk_per_trade_pct_start",
        "gross_profit_usd",
        "gross_loss_usd",
        "net_pnl_usd",
        "final_balance_usd",
        "min_balance_usd",
        "max_drawdown_usd",
        "max_drawdown_pct",
        "roi_pct",
        "expectancy_usd",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _candidate_score(metrics: dict[str, object]) -> float:
    pf = min(5.0, _safe_float(metrics.get("profit_factor")))
    avg_r = _safe_float(metrics.get("avg_r"))
    net = _safe_float(metrics.get("net_pnl_usd"))
    dd = _safe_float(metrics.get("max_drawdown_usd"))
    trades = _safe_float(metrics.get("trades"))
    trade_penalty = 0.0 if trades >= 20 else (20 - trades) * 0.20
    return round(pf + avg_r * 4.0 + (net / 1000.0) * 2.0 - (dd / 1000.0) * 1.5 - trade_penalty, 6)


def _result_payload(
    *,
    scenario: ExitCalibrationScenario,
    result: BacktestRunResult,
    account: BacktestAccountSettings,
    history_limit: int,
    evaluation_step: int,
) -> dict[str, object]:
    metrics = _compact_metrics(result)
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "history_limit": history_limit,
        "evaluation_step": evaluation_step,
        "metrics": metrics,
        "candidate_score": _candidate_score(metrics),
        "pairs": _pair_metrics(result),
        "regimes": _regime_metrics(result, account),
        "exit_settings": asdict(scenario.exit_settings.sanitized()),
        "snapshot_cache": result.parameters.get("snapshot_cache_stats", {}),
        "trade_cache": {
            "status": result.parameters.get("trade_cache_status"),
            "key": result.parameters.get("trade_cache_key"),
            "path": result.parameters.get("trade_cache_path"),
        },
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
    }


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False), encoding="utf-8")


def _print_line(payload: dict[str, object]) -> None:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    print(
        "{name:22s} trades={trades:>3} pf={pf:>6} avg_r={avg_r:>7} net=${net:>8} final=${final:>8} dd=${dd:>8} roi={roi:>7}% score={score}".format(
            name=str(payload.get("scenario")),
            trades=int(_safe_float(metrics.get("trades"))),
            pf=f"{_safe_float(metrics.get('profit_factor')):.2f}",
            avg_r=f"{_safe_float(metrics.get('avg_r')):.3f}",
            net=f"{_safe_float(metrics.get('net_pnl_usd')):+.2f}",
            final=f"{_safe_float(metrics.get('final_balance_usd')):.2f}",
            dd=f"{_safe_float(metrics.get('max_drawdown_usd')):.2f}",
            roi=f"{_safe_float(metrics.get('roi_pct')):+.2f}",
            score=f"{_safe_float(payload.get('candidate_score')):.3f}",
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate exit profiles with fixed-dollar backtest accounting.")
    parser.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY")
    parser.add_argument("--history-limit", type=int, default=1200)
    parser.add_argument("--evaluation-step", type=int, default=2)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--warmup-bars", type=int, default=120)
    parser.add_argument("--end-time", default=None, help="Optional fixed historical end timestamp, e.g. 2026-06-15T00:00:00Z")
    parser.add_argument("--ltf", default=None)
    parser.add_argument("--htf", default=None)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--scenarios", default="all", help="Comma-separated scenario names or 'all'.")
    parser.add_argument("--account-balance", type=float, default=1000.0)
    parser.add_argument("--risk-per-trade", type=float, default=50.0)
    parser.add_argument("--account-currency", default="USD")
    parser.add_argument("--disable-account-report", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--no-snapshot-cache", action="store_true")
    parser.add_argument("--snapshot-cache-size", type=int, default=250000)
    parser.add_argument("--trade-cache", action="store_true", help="Use persisted trade cache per scenario")
    parser.add_argument("--refresh-trade-cache", action="store_true", help="Ignore persisted trade cache reads and overwrite after running")
    parser.add_argument("--trade-cache-dir", default=None)
    parser.add_argument("--output", default="reports/exit_calibration.json")
    parser.add_argument("--list-scenarios", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = build_exit_scenarios()
    if args.list_scenarios:
        for scenario in scenarios.values():
            print(f"{scenario.name}: {scenario.description}")
        return

    selected, missing = _select_scenarios(args.scenarios, scenarios)
    if missing:
        raise SystemExit(f"Unknown scenarios: {', '.join(missing)}")
    if not selected:
        raise SystemExit("No scenarios selected")

    base_settings = Settings.from_env()
    pairs = _parse_pairs(args.pairs)
    account = _account_settings(args)
    end_time = _resolve_end_time(base_settings, args)
    trade_cache = BacktestTradeCache(_trade_cache_settings(base_settings, args))
    snapshot_settings = SnapshotCacheSettings(
        enabled=not args.no_snapshot_cache and base_settings.enable_backtest_snapshot_cache,
        max_entries=max(1000, int(args.snapshot_cache_size)),
    )
    snapshot_cache = SnapshotCache(snapshot_settings)
    output = Path(args.output)
    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "settings": {
            "pairs": pairs,
            "history_limit": args.history_limit,
            "evaluation_step": args.evaluation_step,
            "max_hold_bars": args.max_hold_bars,
            "warmup_bars": args.warmup_bars,
            "cache_only": args.cache_only,
            "end_time": end_time,
            "trade_cache": asdict(trade_cache.settings),
            "snapshot_cache": asdict(snapshot_settings),
            "account": asdict(account),
            "timeframes": {
                "ltf": (args.ltf or base_settings.ltf_timeframe).upper(),
                "htf": (args.htf or base_settings.htf_timeframe).upper(),
                "trigger": (args.trigger or base_settings.trigger_timeframe).upper(),
            },
        },
        "runs": [],
        "ranking": [],
        "snapshot_cache": snapshot_cache.stats(),
    }

    print(
        f"Exit calibration | pairs={','.join(pairs)} bars={args.history_limit} step={args.evaluation_step} "
        f"account={account.starting_balance:.2f} {account.currency} risk={account.risk_per_trade:.2f} {account.currency}",
        flush=True,
    )
    for scenario in selected:
        print(f"\n=== {scenario.name} ===", flush=True)
        engine = _build_engine(
            base_settings=base_settings,
            scenario=scenario,
            history_limit=max(150, args.history_limit),
            evaluation_step=max(1, args.evaluation_step),
            max_hold_bars=max(1, args.max_hold_bars),
            warmup_bars=max(80, args.warmup_bars),
            cache_only=bool(args.cache_only),
            ltf=(args.ltf or base_settings.ltf_timeframe).upper(),
            htf=(args.htf or base_settings.htf_timeframe).upper(),
            trigger=(args.trigger or base_settings.trigger_timeframe).upper(),
            snapshot_cache=snapshot_cache,
            snapshot_settings=snapshot_settings,
            account_settings=account,
            end_time=end_time,
        )
        cache_key_payload = _scenario_cache_key_payload(
            base_settings=base_settings,
            scenario=scenario,
            pairs=pairs,
            account=account,
            args=args,
            end_time=end_time,
        )
        cache_key = trade_cache.build_key(cache_key_payload)
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
                    cache_key_payload,
                    result,
                    metadata={"runner": "research.exit_calibration", "scenario": scenario.name},
                )
                if path is not None:
                    print(f"Trade cache: STORED {path}", flush=True)
        payload = _result_payload(
            scenario=scenario,
            result=result,
            account=account,
            history_limit=max(150, args.history_limit),
            evaluation_step=max(1, args.evaluation_step),
        )
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
                }
                for item in report.get("runs", [])
                if isinstance(item, dict)
            ],
            key=lambda item: _safe_float(item.get("candidate_score")),
            reverse=True,
        )
        _print_line(payload)
        _write_report(output, report)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["snapshot_cache"] = snapshot_cache.stats()
    _write_report(output, report)
    print(f"\nSaved exit calibration report: {output}", flush=True)
    print("\nTOP SCENARIOS", flush=True)
    for row in report.get("ranking", [])[:10]:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        print(
            "{name:22s} final=${final:.2f} net=${net:+.2f} dd=${dd:.2f} pf={pf:.2f} avg_r={avg_r:.3f} trades={trades}".format(
                name=str(row.get("scenario")),
                final=_safe_float(metrics.get("final_balance_usd")),
                net=_safe_float(metrics.get("net_pnl_usd")),
                dd=_safe_float(metrics.get("max_drawdown_usd")),
                pf=_safe_float(metrics.get("profit_factor")),
                avg_r=_safe_float(metrics.get("avg_r")),
                trades=int(_safe_float(metrics.get("trades"))),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
