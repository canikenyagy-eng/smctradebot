from __future__ import annotations

import argparse
import logging
import json
from dataclasses import asdict
from pathlib import Path

from analytics.feature_analysis import (
    compute_avg_contribution,
    compute_correlation_with_pnl,
    compute_win_vs_loss_contribution,
)
from analytics.monte_carlo import MonteCarloSettings, run_monte_carlo
from backtest.exit_engine import AdaptiveExitSettings
from backtest.execution import RealisticExecutionSettings
from backtest.engine import BacktestAccountSettings, BacktestEngine
from backtest.meta_label import MetaLabelSettings
from backtest.news import HistoricalNewsFeed, NeutralNewsFeed
from backtest.portfolio_layer import PortfolioLayerSettings
from backtest.risk import ATRRiskSettings, EquityProtectionSettings
from backtest.smc_research_features import SMCResearchFeatureSettings
from backtest.sizing import AdaptiveSizingSettings
from backtest.snapshot_cache import SnapshotCacheSettings
from backtest.trade_cache import BacktestTradeCache, TradeCacheSettings
from backtest.validation import (
    BacktestValidationResult,
    BacktestValidationRunner,
    build_score_distribution_report,
    build_validation_score_distribution_report,
)
from backtest.walk_forward import WalkForwardResult, WalkForwardRunner
from config import Settings
from core.regime_analytics import analyze_regime_performance_from_run, export_regime_report
from core.signal_engine import SignalEngine
from core.meta_optimizer import (
    MetaOptimizer,
    OptimizationConfig,
    run_meta_optimization,
    MetaOptimizationResult,
)
from data.market_data import MarketDataCacheConfig, MarketDataClient
from execution.news import NewsFilter


def _parse_pairs(raw: str) -> list[str]:
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an SMC backtest on Forex pairs.")
    parser.add_argument("--pairs", default=None, help="Comma-separated pairs, e.g. EURUSD,GBPUSD")
    parser.add_argument("--ltf", default=None, help="Lower timeframe, e.g. M15")
    parser.add_argument("--htf", default=None, help="Higher timeframe, e.g. H1")
    parser.add_argument("--trigger", default=None, help="Trigger timeframe, e.g. M5")
    parser.add_argument("--history-limit", type=int, default=3000, help="Historical bars to load per timeframe")
    parser.add_argument("--max-hold-bars", type=int, default=48, help="Maximum bars to hold each trade")
    parser.add_argument("--warmup-bars", type=int, default=120, help="Warmup bars before evaluation starts")
    parser.add_argument("--evaluation-step", type=int, default=None, help="Backtest-only bar step between signal evaluations; 1 preserves exact behavior")
    parser.add_argument("--end-time", default=None, help="Optional fixed historical end timestamp, e.g. 2026-06-15T00:00:00Z")
    parser.add_argument("--account-balance", type=float, default=None, help="Backtest starting balance for USD accounting")
    parser.add_argument("--risk-per-trade", type=float, default=None, help="Fixed cash risk per 1R trade for backtest accounting")
    parser.add_argument("--account-currency", default=None, help="Backtest account currency label, e.g. USD")
    parser.add_argument("--disable-account-report", action="store_true", help="Disable fixed-cash account reporting")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--wf-train-months", type=int, default=None, help="Walk-forward train window in months")
    parser.add_argument("--wf-test-months", type=int, default=None, help="Walk-forward test window in months")
    parser.add_argument("--wf-step-months", type=int, default=None, help="Walk-forward step size in months")
    parser.add_argument("--meta-optimize", action="store_true", help="Run meta-optimization on training folds (recommendations only)")
    parser.add_argument("--complexity-penalty", type=float, default=0.1, help="Complexity penalty for meta-optimization")
    parser.add_argument("--analyze-scores", action="store_true", help="Print score distribution and feature contribution analytics")
    parser.add_argument("--dynamic-threshold-analysis", action="store_true", help="Print rolling percentile threshold analytics from historical score stream")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo bootstrap analysis on backtest R-multiples")
    parser.add_argument("--mc-iterations", type=int, default=2000, help="Monte Carlo bootstrap iterations")
    parser.add_argument("--mc-seed", type=int, default=42, help="Monte Carlo random seed")
    parser.add_argument("--mc-ruin-dd", type=float, default=10.0, help="Drawdown threshold in R for risk-of-ruin estimate")
    parser.add_argument("--news-csv", default=None, help="Optional historical news CSV for backtest filtering")
    parser.add_argument("--output-dir", default=None, help="Optional export directory")
    parser.add_argument("--cache-only", action="store_true", help="Use only local OHLCV cache; do not fetch Yahoo data")
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh OHLCV cache from Yahoo before running")
    parser.add_argument("--cache-dir", default=None, help="Override local OHLCV cache directory")
    parser.add_argument("--trade-cache", action="store_true", help="Use persisted trade cache for identical backtest runs")
    parser.add_argument("--refresh-trade-cache", action="store_true", help="Ignore persisted trade cache reads and overwrite it after the run")
    parser.add_argument("--trade-cache-dir", default=None, help="Override persisted trade cache directory")
    parser.add_argument("--validate-shadow", action="store_true", help="Compare shadow scoring disabled vs enabled on the same historical data")
    parser.add_argument("--validate-mitigation-entry", action="store_true", help="Compare market entry vs mitigation limit entry on the same historical data")
    parser.add_argument("--validation-output-dir", default=None, help="Optional export directory for validation results")
    parser.add_argument("--no-export", action="store_true", help="Do not write CSV/JSON export files")
    parser.add_argument("--export-meta-report", action="store_true", help="Export meta-optimization report to JSON")
    return parser


def format_number(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def print_report(result) -> None:
    overall = result.overall_metrics()
    print()
    print("BACKTEST SUMMARY")
    print(f"News mode: {result.news_mode}")
    print(f"Period: {result.started_at.isoformat()} -> {result.finished_at.isoformat()}")
    print(f"Trigger TF: {result.parameters.get('trigger_timeframe')} | LTF: {result.parameters.get('ltf_timeframe')} | HTF: {result.parameters.get('htf_timeframe')}")
    print(
        "Totals: trades={trades} wins={wins} losses={losses} breakeven={breakeven} "
        "win_rate={win_rate:.1%} avg_r={avg_r:.2f} exp={exp:.2f} avg_win={avg_win:.2f} avg_loss={avg_loss:.2f} pf={pf} max_dd={dd:.2f}R".format(
            trades=overall["trades"],
            wins=overall["wins"],
            losses=overall["losses"],
            breakeven=overall["breakeven"],
            win_rate=overall["win_rate"],
            avg_r=overall["avg_r"],
            exp=float(overall.get("expectancy_r", overall["avg_r"])),
            avg_win=float(overall.get("avg_win_r", 0.0)),
            avg_loss=float(overall.get("avg_loss_r", 0.0)),
            pf=format_number(float(overall["profit_factor"])),
            dd=float(overall["max_drawdown_r"]),
        )
    )
    if overall.get("accounting_enabled"):
        currency = str(overall.get("account_currency", "USD"))
        print(
            "Account: start={start:.2f} {currency} risk/trade={risk:.2f} {currency} final={final:.2f} {currency} "
            "net={net:+.2f} {currency} roi={roi:+.2f}% max_dd={dd:.2f} {currency} ({dd_pct:.2f}%) expectancy={exp:+.2f} {currency}/trade".format(
                start=float(overall.get("starting_balance", 0.0)),
                risk=float(overall.get("risk_per_trade", 0.0)),
                final=float(overall.get("final_balance_usd", 0.0)),
                net=float(overall.get("net_pnl_usd", 0.0)),
                roi=float(overall.get("roi_pct", 0.0)),
                dd=float(overall.get("max_drawdown_usd", 0.0)),
                dd_pct=float(overall.get("max_drawdown_pct", 0.0)),
                exp=float(overall.get("expectancy_usd", 0.0)),
                currency=currency,
            )
        )
    print(
        "Quality: avg_score={score:.1f} avg_shadow={shadow:.1f} avg_hold={hold:.1f} bars limit={limit} market={market} fill_delay={delay:.1f} bars partial={partial} be={be} trail={trail} atr_trail={atr_trail} liq_trail={liq_trail} adaptive_exit={adaptive} target_rr={target_rr:.2f} sizing={sizing:.2f} meta_p={meta_p:.2f} meta_ok={meta_ok} portfolio_mult={portfolio:.2f} tp_hits={tp} sl_hits={sl} timeout={timeout}".format(
            score=float(overall["avg_score"]),
            shadow=float(overall.get("avg_shadow_bonus", 0.0)),
            hold=float(overall["avg_bars_held"]),
            limit=overall.get("limit_entries", 0),
            market=overall.get("market_entries", 0),
            delay=float(overall.get("avg_fill_delay_bars", 0.0)),
            partial=overall.get("partial_exits", 0),
            be=overall.get("break_even_activations", 0),
            trail=overall.get("trailing_activations", 0),
            atr_trail=overall.get("atr_trailing_activations", 0),
            liq_trail=overall.get("liquidity_trailing_activations", 0),
            adaptive=overall.get("adaptive_exit_trades", 0),
            target_rr=float(overall.get("avg_exit_target_rr", 0.0)),
            sizing=float(overall.get("avg_sizing_multiplier", 1.0)),
            meta_p=float(overall.get("avg_meta_probability", 1.0)),
            meta_ok=overall.get("meta_accepted_count", 0),
            portfolio=float(overall.get("avg_portfolio_multiplier", 1.0)),
            tp=overall["tp_hits"],
            sl=overall["sl_hits"],
            timeout=overall["timeout_exits"],
        )
    )
    print(
        "Execution: fill_rate={fill:.1%} partial_fill_rate={pfill:.1%} spread={spread:.2f} pips slippage={slip:.2f} pips spread_cost={spread_cost:.2f}R slippage_cost={slip_cost:.2f}R delay_cost={delay_cost:.2f}R risk_mult={risk_mult:.2f}".format(
            fill=float(overall.get("fill_rate", 0.0)),
            pfill=float(overall.get("partial_fill_rate", 0.0)),
            spread=float(overall.get("avg_spread_pips", 0.0)),
            slip=float(overall.get("avg_slippage_pips", 0.0)),
            spread_cost=float(overall.get("total_spread_cost_r", 0.0)),
            slip_cost=float(overall.get("total_slippage_cost_r", 0.0)),
            delay_cost=float(overall.get("avg_delay_cost_r", 0.0)),
            risk_mult=float(overall.get("avg_risk_multiplier", 1.0)),
        )
    )
    cache_stats = result.parameters.get("snapshot_cache_stats", {}) if hasattr(result, "parameters") else {}
    if isinstance(cache_stats, dict) and cache_stats:
        print(
            "Cache: snapshot_enabled={enabled} entries={entries} hits={hits} misses={misses} stores={stores} skips={skips}".format(
                enabled=cache_stats.get("enabled", False),
                entries=cache_stats.get("entries", 0),
                hits=cache_stats.get("hits", 0),
                misses=cache_stats.get("misses", 0),
                stores=cache_stats.get("stores", 0),
                skips=cache_stats.get("skips", 0),
            )
        )
    print()
    print("PAIR BREAKDOWN")
    for row in result.pair_rows():
        if row["error"]:
            print(f"{row['pair']}: ERROR {row['error']}")
            continue
        pf = format_number(float(row["profit_factor"]))
        print(
            "{pair}: trades={trades} win_rate={win_rate:.1%} avg_r={avg_r:.2f} pf={pf} max_dd={dd:.2f}R "
            "exp={exp:.2f} payoff={payoff} shadow={shadow:.1f} limit={limit} market={market} fill_delay={delay:.1f} partial={partial} be={be} trail={trail} atr={atr} liq={liq} adaptive={adaptive} target_rr={target_rr:.2f} sizing={sizing:.2f} meta_p={meta_p:.2f} portfolio={portfolio:.2f} "
            "fill={fill:.1%} spread={spread:.2f} slip={slip:.2f} pnl={pnl:+.2f} final={final:.2f} accept={acc:.1%} rejects={rej}".format(
                pair=row["pair"],
                trades=row["trades"],
                win_rate=row["win_rate"],
                avg_r=row["avg_r"],
                pf=pf,
                dd=float(row["max_drawdown_r"]),
                exp=float(row.get("expectancy_r", row["avg_r"])),
                payoff=format_number(float(row.get("payoff_ratio", 0.0))),
                shadow=float(row.get("avg_shadow_bonus", 0.0)),
                limit=row.get("limit_entries", 0),
                market=row.get("market_entries", 0),
                delay=float(row.get("avg_fill_delay_bars", 0.0)),
                partial=row.get("partial_exits", 0),
                be=row.get("break_even_activations", 0),
                trail=row.get("trailing_activations", 0),
                atr=row.get("atr_trailing_activations", 0),
                liq=row.get("liquidity_trailing_activations", 0),
                adaptive=row.get("adaptive_exit_trades", 0),
                target_rr=float(row.get("avg_exit_target_rr", 0.0)),
                sizing=float(row.get("avg_sizing_multiplier", 1.0)),
                meta_p=float(row.get("avg_meta_probability", 1.0)),
                portfolio=float(row.get("avg_portfolio_multiplier", 1.0)),
                fill=float(row.get("fill_rate", 0.0)),
                spread=float(row.get("avg_spread_pips", 0.0)),
                slip=float(row.get("avg_slippage_pips", 0.0)),
                pnl=float(row.get("net_pnl_usd", 0.0)),
                final=float(row.get("final_balance_usd", 0.0)),
                acc=row["acceptance_rate"],
                rej=row["rejections"],
            )
        )


def _fmt_delta(value: float | None, *, as_points: bool = False) -> str:
    if value is None:
        return "n/a"

    if as_points:
        points = value * 100.0
        sign = "+" if points > 0 else ""
        return f"{sign}{points:.1f}pp"

    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}"


def _print_validation_line(label: str, metrics: dict[str, object]) -> None:
    print(
            "{label}: trades={trades} win_rate={win_rate:.1%} avg_r={avg_r:.2f} pf={pf} max_dd={dd:.2f}R "
            "exp={exp:.2f} payoff={payoff} shadow_bonus={shadow:.1f} limit={limit} market={market} fill_delay={delay:.1f} partial={partial} be={be} trail={trail}".format(
            label=label,
            trades=metrics["trades"],
            win_rate=metrics["win_rate"],
            avg_r=metrics["avg_r"],
            pf=format_number(float(metrics["profit_factor"])),
            dd=float(metrics["max_drawdown_r"]),
            exp=float(metrics.get("expectancy_r", metrics["avg_r"])),
            payoff=format_number(float(metrics.get("payoff_ratio", 0.0))),
            shadow=float(metrics.get("avg_shadow_bonus", 0.0)),
            limit=metrics.get("limit_entries", 0),
            market=metrics.get("market_entries", 0),
            delay=float(metrics.get("avg_fill_delay_bars", 0.0)),
            partial=metrics.get("partial_exits", 0),
            be=metrics.get("break_even_activations", 0),
            trail=metrics.get("trailing_activations", 0),
        )
    )


def _print_validation_delta(label: str, baseline: dict[str, object], shadow: dict[str, object]) -> None:
    print(
        "{label}: trades={trades} win_rate={win_rate} avg_r={avg_r} pf={pf} max_dd={dd} shadow_bonus={shadow} "
        "limit={limit} market={market} fill_delay={delay} partial={partial} be={be} trail={trail}".format(
            label=label,
            trades=_fmt_delta(float(shadow["trades"]) - float(baseline["trades"])),
            win_rate=_fmt_delta(float(shadow["win_rate"]) - float(baseline["win_rate"]), as_points=True),
            avg_r=_fmt_delta(float(shadow["avg_r"]) - float(baseline["avg_r"])),
            pf=_fmt_delta(float(shadow["profit_factor"]) - float(baseline["profit_factor"])),
            dd=_fmt_delta(float(shadow["max_drawdown_r"]) - float(baseline["max_drawdown_r"])),
            shadow=_fmt_delta(float(shadow.get("avg_shadow_bonus", 0.0)) - float(baseline.get("avg_shadow_bonus", 0.0))),
            limit=_fmt_delta(float(shadow.get("limit_entries", 0)) - float(baseline.get("limit_entries", 0))),
            market=_fmt_delta(float(shadow.get("market_entries", 0)) - float(baseline.get("market_entries", 0))),
            delay=_fmt_delta(float(shadow.get("avg_fill_delay_bars", 0.0)) - float(baseline.get("avg_fill_delay_bars", 0.0))),
            partial=_fmt_delta(float(shadow.get("partial_exits", 0)) - float(baseline.get("partial_exits", 0))),
            be=_fmt_delta(float(shadow.get("break_even_activations", 0)) - float(baseline.get("break_even_activations", 0))),
            trail=_fmt_delta(float(shadow.get("trailing_activations", 0)) - float(baseline.get("trailing_activations", 0))),
        )
    )


def print_validation_report(result: BacktestValidationResult) -> None:
    baseline = result.baseline.overall_metrics()
    shadow = result.shadow.overall_metrics()

    print()
    print("VALIDATION SUMMARY")
    print(f"Period: {result.baseline.started_at.isoformat()} -> {result.shadow.finished_at.isoformat()}")
    print(f"Baseline mode: {result.baseline.parameters.get('mode')} | Shadow mode: {result.shadow.parameters.get('mode')}")
    _print_validation_line("Baseline", baseline)
    _print_validation_line("Shadow", shadow)
    _print_validation_delta("Delta", baseline, shadow)
    print(f"Rejections delta: {result.comparison['rejections_delta']}")
    print()
    print("PAIR VALIDATION")
    for row in result.pair_rows:
        if row.get("baseline_error") or row.get("shadow_error"):
            print(f"{row['pair']}: ERROR baseline={row.get('baseline_error')} shadow={row.get('shadow_error')}")
            continue

        print(
            "{pair}: trades={b_trades}->{s_trades} win_rate={b_wr:.1%}->{s_wr:.1%} avg_r={b_ar:.2f}->{s_ar:.2f} "
            "pf={b_pf}->{s_pf} shadow={b_shadow:.1f}->{s_shadow:.1f} limit={b_limit}->{s_limit} market={b_market}->{s_market} fill_delay={b_delay:.1f}->{s_delay:.1f} partial={b_partial}->{s_partial} be={b_be}->{s_be} trail={b_trail}->{s_trail}".format(
                pair=row["pair"],
                b_trades=row["baseline_trades"],
                s_trades=row["shadow_trades"],
                b_wr=row["baseline_win_rate"],
                s_wr=row["shadow_win_rate"],
                b_ar=row["baseline_avg_r"],
                s_ar=row["shadow_avg_r"],
                b_pf=format_number(float(row["baseline_profit_factor"])),
                s_pf=format_number(float(row["shadow_profit_factor"])),
                b_shadow=float(row.get("baseline_avg_shadow_bonus", 0.0)),
                s_shadow=float(row.get("shadow_avg_shadow_bonus", 0.0)),
                b_limit=row.get("baseline_limit_entries", 0),
                s_limit=row.get("shadow_limit_entries", 0),
                b_market=row.get("baseline_market_entries", 0),
                s_market=row.get("shadow_market_entries", 0),
                b_delay=float(row.get("baseline_avg_fill_delay_bars", 0.0)),
                s_delay=float(row.get("shadow_avg_fill_delay_bars", 0.0)),
                b_partial=row.get("baseline_partial_exits", 0),
                s_partial=row.get("shadow_partial_exits", 0),
                b_be=row.get("baseline_break_even_activations", 0),
                s_be=row.get("shadow_break_even_activations", 0),
                b_trail=row.get("baseline_trailing_activations", 0),
                s_trail=row.get("shadow_trailing_activations", 0),
            )
        )


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_feature_analysis_payload(result) -> dict[str, object]:
    trades = result.trades
    return {
        "sample_size": len(trades),
        "average_contribution": compute_avg_contribution(trades),
        "win_vs_loss_contribution": compute_win_vs_loss_contribution(trades),
        "correlation_with_pnl": compute_correlation_with_pnl(trades),
    }


def _print_feature_analysis_report(payload: dict[str, object], title: str) -> None:
    print()
    print(title)
    print(f"Samples: {payload.get('sample_size', 0)}")
    avg = payload.get("average_contribution", {})
    if isinstance(avg, dict):
        print("Average contribution:")
        print(", ".join(f"{key}={value}" for key, value in avg.items()))


def _print_score_distribution_report(payload: dict[str, object], title: str) -> None:
    print()
    print(title)
    distribution = payload.get("score_distribution", {})
    rejections = payload.get("rejections", {})
    if isinstance(distribution, dict):
        print(
            "Scores: count={count} mean={mean} median={median} above_threshold={above} acceptance={acc:.1%} scored={coverage:.1%}".format(
                count=distribution.get("count", 0),
                mean=distribution.get("mean", 0.0),
                median=distribution.get("median", 0.0),
                above=distribution.get("above_threshold_count", 0),
                acc=float(distribution.get("acceptance_rate", 0.0)),
                coverage=float(distribution.get("score_coverage_rate", 0.0)),
            )
        )
    if isinstance(rejections, dict):
        categories = rejections.get("by_category", {})
        print(f"Rejections: {categories}")
    dynamic = payload.get("dynamic_threshold", {})
    if isinstance(dynamic, dict) and dynamic:
        print(
            "Dynamic threshold: pctl={p:.1f} window={w} trace={n} mean={m:.2f} median={med:.2f} acceptance_if_applied={acc:.1%}".format(
                p=float(dynamic.get("percentile", 0.0)),
                w=int(dynamic.get("rolling_window", 0)),
                n=int(dynamic.get("trace_count", 0)),
                m=float(dynamic.get("mean_recommended_threshold", 0.0)),
                med=float(dynamic.get("median_recommended_threshold", 0.0)),
                acc=float(dynamic.get("acceptance_rate_if_applied", 0.0)),
            )
        )


def _build_monte_carlo_payload(result, args) -> dict[str, object]:
    return run_monte_carlo(
        [trade.r_multiple for trade in result.trades],
        MonteCarloSettings(
            iterations=args.mc_iterations,
            seed=args.mc_seed,
            ruin_drawdown_r=args.mc_ruin_dd,
        ),
    )


def _print_monte_carlo_report(payload: dict[str, object], title: str) -> None:
    print()
    print(title)
    if payload.get("error"):
        print(f"Monte Carlo unavailable: {payload['error']}")
        return

    terminal = payload.get("terminal_r", {})
    drawdown = payload.get("max_drawdown_r", {})
    streak = payload.get("longest_loss_streak", {})
    if not isinstance(terminal, dict) or not isinstance(drawdown, dict) or not isinstance(streak, dict):
        return

    print(
        "Samples={samples} iterations={iters} positive={positive:.1%} risk_of_ruin={ruin:.1%}".format(
            samples=payload.get("sample_size", 0),
            iters=payload.get("iterations", 0),
            positive=float(payload.get("positive_terminal_probability", 0.0)),
            ruin=float(payload.get("risk_of_ruin_probability", 0.0)),
        )
    )
    print(
        "Terminal R: median={median:.2f} p05={p05:.2f} p95={p95:.2f} | MaxDD R: median={dd_med:.2f} p95={dd95:.2f} p99={dd99:.2f} | Loss streak p95={ls95:.0f}".format(
            median=float(terminal.get("median", 0.0)),
            p05=float(terminal.get("p05", 0.0)),
            p95=float(terminal.get("p95", 0.0)),
            dd_med=float(drawdown.get("median", 0.0)),
            dd95=float(drawdown.get("p95", 0.0)),
            dd99=float(drawdown.get("p99", 0.0)),
            ls95=float(streak.get("p95", 0.0)),
        )
    )


def _print_regime_report(payload: dict[str, object], title: str) -> None:
    print()
    print(title)
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        print(
            "Totals: regimes={regimes} trades={trades} evaluations={evals} accepted={accepted} global_acceptance={acc}".format(
                regimes=summary.get("regime_count", 0),
                trades=summary.get("total_trades", 0),
                evals=summary.get("total_evaluations", 0),
                accepted=summary.get("total_accepted_signals", 0),
                acc=summary.get("global_acceptance_rate", "n/a"),
            )
        )
    regimes = payload.get("regimes", {})
    if isinstance(regimes, dict):
        for regime, row in sorted(regimes.items()):
            if not isinstance(row, dict):
                continue
            print(
                "{regime}: trades={trades} eval={evals} accepted={accepted} acc={acc} win_rate={wr:.1%} avg_r={avg:.2f} exp={exp:.2f} payoff={payoff} pf={pf}".format(
                    regime=regime,
                    trades=row.get("signal_count", 0),
                    evals=row.get("evaluations", 0),
                    accepted=row.get("accepted_signals", 0),
                    acc=row.get("acceptance_rate", "n/a"),
                    wr=float(row.get("win_rate", 0.0)),
                    avg=float(row.get("avg_r", 0.0)),
                    exp=float(row.get("expectancy_r", row.get("avg_r", 0.0))),
                    payoff=format_number(float(row.get("payoff_ratio", 0.0)))
                    if row.get("payoff_ratio") is not None
                    else "inf",
                    pf=format_number(float(row.get("profit_factor", 0.0)))
                    if row.get("profit_factor") is not None
                    else "inf",
                )
            )


def print_walk_forward_report(result: WalkForwardResult) -> None:
    print()
    print("WALK-FORWARD SUMMARY")
    print(f"Period: {result.parameters.get('global_start')} -> {result.parameters.get('global_end')}")
    print(
        "Windows={windows} train_win={train_wr:.1%} test_win={test_wr:.1%} "
        "train_pf={train_pf:.2f} test_pf={test_pf:.2f} "
        "train_avg_r={train_r:.3f} test_avg_r={test_r:.3f} "
        "train_trades={train_trades} test_trades={test_trades}".format(
            windows=result.summary.get("window_count", 0),
            train_wr=float(result.summary.get("train_avg_win_rate", 0.0)),
            test_wr=float(result.summary.get("test_avg_win_rate", 0.0)),
            train_pf=float(result.summary.get("train_avg_profit_factor", 0.0)),
            test_pf=float(result.summary.get("test_avg_profit_factor", 0.0)),
            train_r=float(result.summary.get("train_avg_r", 0.0)),
            test_r=float(result.summary.get("test_avg_r", 0.0)),
            train_trades=result.summary.get("train_total_trades", 0),
            test_trades=result.summary.get("test_total_trades", 0),
        )
    )
    # Extended stability metrics
    stability = result.summary.get("stability_metrics", {})
    if stability:
        print(
            "Stability: gap_pf={gap_pf:.2%} gap_wr={gap_wr:.2%} gap_r={gap_r:.2%} "
            "consistency={consistency:.2%}".format(
                gap_pf=float(stability.get("train_test_gap_pf", 0.0)),
                gap_wr=float(stability.get("train_test_gap_wr", 0.0)),
                gap_r=float(stability.get("train_test_gap_r", 0.0)),
                consistency=float(stability.get("consistency_score", 0.0)),
            )
        )
    # Extended aggregate metrics
    agg = result.summary.get("aggregate_metrics", {})
    if agg:
        print(
            "StdDev: train_pf_std={train_std:.3f} test_pf_std={test_std:.3f} "
            "train_wr_std={train_wr_std:.3f} test_wr_std={test_wr_std:.3f}".format(
                train_std=float(agg.get("train_std_pf", 0.0)),
                test_std=float(agg.get("test_std_pf", 0.0)),
                train_wr_std=float(agg.get("train_std_wr", 0.0)),
                test_wr_std=float(agg.get("test_std_wr", 0.0)),
            )
        )


def print_meta_optimization_report(result: MetaOptimizationResult) -> None:
    """Print meta-optimization results."""
    print()
    print("STABILITY METRICS")
    print(
        "Mean={mean:.2%} Std={std:.2%} BestFold={best} WorstFold={worst}".format(
            mean=result.mean_stability,
            std=result.std_stability,
            best=result.best_fold_index,
            worst=result.worst_fold_index,
        )
    )
    print()
    print("RECOMMENDATIONS (Not Auto-Applied)")
    print(
        "ScoreThreshold={threshold} ATR_Multiplier={atr}".format(
            threshold=result.recommended_threshold,
            atr=result.recommended_atr,
        )
    )
    if result.recommended_weights:
        print("WeightAdjustments: " + str(result.recommended_weights))
    print()
    print("FOLD BREAKDOWN")
    for fold in result.fold_results:
        test_trades = fold.test_metrics.get("trades", 0) if fold.test_metrics else 0
        print(
            "Fold={idx} train_trades={train} test_trades={test} "
            "stability={stab:.2%} complexity_penalty={penalty:.2%}".format(
                idx=fold.fold_index,
                train=fold.train_result.overall_metrics().get("trades", 0),
                test=test_trades,
                stab=fold.stability_score,
                penalty=fold.complexity_penalty,
            )
        )


def _metrics_delta(baseline: dict[str, object], realistic: dict[str, object], key: str) -> float:
    return float(realistic.get(key, 0.0)) - float(baseline.get(key, 0.0))


def print_realistic_comparison(baseline_result, realistic_result) -> dict[str, object]:
    baseline = baseline_result.overall_metrics()
    realistic = realistic_result.overall_metrics()
    delta = {
        "trades": _metrics_delta(baseline, realistic, "trades"),
        "win_rate": _metrics_delta(baseline, realistic, "win_rate"),
        "avg_r": _metrics_delta(baseline, realistic, "avg_r"),
        "profit_factor": _metrics_delta(baseline, realistic, "profit_factor"),
        "max_drawdown_r": _metrics_delta(baseline, realistic, "max_drawdown_r"),
        "fill_rate": _metrics_delta(baseline, realistic, "fill_rate"),
        "avg_slippage_pips": _metrics_delta(baseline, realistic, "avg_slippage_pips"),
        "avg_spread_pips": _metrics_delta(baseline, realistic, "avg_spread_pips"),
        "total_slippage_cost_r": _metrics_delta(baseline, realistic, "total_slippage_cost_r"),
        "total_spread_cost_r": _metrics_delta(baseline, realistic, "total_spread_cost_r"),
        "avg_delay_cost_r": _metrics_delta(baseline, realistic, "avg_delay_cost_r"),
    }

    print()
    print("REALISTIC EXECUTION COMPARISON")
    print(
        "Baseline: trades={trades} win_rate={wr:.1%} avg_r={avg_r:.2f} pf={pf} dd={dd:.2f} fill={fill:.1%}".format(
            trades=baseline.get("trades", 0),
            wr=float(baseline.get("win_rate", 0.0)),
            avg_r=float(baseline.get("avg_r", 0.0)),
            pf=format_number(float(baseline.get("profit_factor", 0.0))),
            dd=float(baseline.get("max_drawdown_r", 0.0)),
            fill=float(baseline.get("fill_rate", 0.0)),
        )
    )
    print(
        "Realistic: trades={trades} win_rate={wr:.1%} avg_r={avg_r:.2f} pf={pf} dd={dd:.2f} fill={fill:.1%} spread_cost={spread:.2f}R slippage_cost={slip:.2f}R".format(
            trades=realistic.get("trades", 0),
            wr=float(realistic.get("win_rate", 0.0)),
            avg_r=float(realistic.get("avg_r", 0.0)),
            pf=format_number(float(realistic.get("profit_factor", 0.0))),
            dd=float(realistic.get("max_drawdown_r", 0.0)),
            fill=float(realistic.get("fill_rate", 0.0)),
            spread=float(realistic.get("total_spread_cost_r", 0.0)),
            slip=float(realistic.get("total_slippage_cost_r", 0.0),
            ),
        )
    )
    print(
        "Delta: trades={trades:+.0f} win_rate={wr:+.2%} avg_r={avg_r:+.2f} pf={pf:+.2f} dd={dd:+.2f} fill={fill:+.2%}".format(
            trades=delta["trades"],
            wr=delta["win_rate"],
            avg_r=delta["avg_r"],
            pf=delta["profit_factor"],
            dd=delta["max_drawdown_r"],
            fill=delta["fill_rate"],
        )
    )
    return {"baseline": baseline, "realistic": realistic, "delta": delta}


def build_signal_engine(
    *,
    market_data: MarketDataClient,
    news_filter: NewsFilter,
    settings: Settings,
    htf: str,
    ltf: str,
    trigger: str,
    enable_shadow_scoring: bool,
    enable_mitigation_entry: bool,
) -> SignalEngine:
    return SignalEngine(
        market_data=market_data,
        news_filter=news_filter,
        htf_timeframe=htf,
        ltf_timeframe=ltf,
        trigger_timeframe=trigger,
        min_score=settings.min_score,
        risk_reward=settings.risk_reward,
        swing_window=settings.swing_window,
        pair_correlation_threshold=settings.pair_correlation_threshold,
        correlation_lookback=settings.correlation_lookback,
        currency_exposure_cap=settings.currency_exposure_cap,
        portfolio_currency_gross_cap=settings.portfolio_currency_gross_cap,
        portfolio_currency_net_cap=settings.portfolio_currency_net_cap,
        portfolio_exposure_window_minutes=settings.portfolio_exposure_window_minutes,
        pair_cooldown_minutes=settings.pair_cooldown_minutes,
        max_entries_per_bias=settings.max_entries_per_bias,
        bias_window_minutes=settings.bias_window_minutes,
        regime_opposition_confidence=settings.regime_opposition_confidence,
        contraction_min_trigger_strength=settings.contraction_min_trigger_strength,
        range_min_trigger_strength=settings.range_min_trigger_strength,
        require_displacement_in_contraction=settings.require_displacement_in_contraction,
        session_min_score=settings.session_min_score,
        enable_smt_confirmation=settings.enable_smt_confirmation,
        smt_hard_gate=settings.smt_hard_gate,
        smt_min_strength=settings.smt_min_strength,
        smt_opposite_block_strength=settings.smt_opposite_block_strength,
        smt_reference_map=settings.smt_reference_map,
        partial_tp_enabled=settings.partial_tp_enabled,
        partial_tp_r=settings.partial_tp_r,
        partial_tp_fraction=settings.partial_tp_fraction,
        break_even_r=settings.break_even_r,
        trailing_enabled=settings.trailing_enabled,
        trailing_start_r=settings.trailing_start_r,
        trailing_lookback_bars=settings.trailing_lookback_bars,
        time_stop_bars=settings.time_stop_bars,
        regime_short_window=settings.regime_short_window,
        regime_long_window=settings.regime_long_window,
        enable_shadow_scoring=enable_shadow_scoring,
        enable_mitigation_entry=enable_mitigation_entry,
        enable_adaptive_weights=settings.enable_adaptive_weights,
        adaptive_weights_preset=settings.adaptive_weights_preset,
        adaptive_regime_weights=settings.adaptive_regime_weights,
        enable_score_normalization=settings.enable_score_normalization,
        score_normalization_method=settings.score_normalization_method,
        score_normalization_window=settings.score_normalization_window,
        score_normalization_scale_factor=settings.score_normalization_scale_factor,
        score_normalization_backtest_only=settings.score_normalization_backtest_only,
        allow_live_score_normalization=settings.allow_live_score_normalization,
        runtime_mode="backtest",
        enable_dynamic_threshold=settings.enable_dynamic_threshold,
        threshold_percentile=settings.threshold_percentile,
        threshold_rolling_window=settings.threshold_rolling_window,
        apply_dynamic_threshold=settings.apply_dynamic_threshold,
        dynamic_threshold_backtest_only=settings.dynamic_threshold_backtest_only,
        allow_live_dynamic_threshold=settings.allow_live_dynamic_threshold,
        enable_structure_quality_scoring=settings.enable_structure_quality_scoring,
        structure_quality_scan_bars=settings.smc_structure_scan_bars,
        structure_quality_min_break_pips=settings.smc_structure_min_break_pips,
        structure_quality_level_bucket_pips=settings.smc_structure_level_bucket_pips,
        structure_quality_min_score_for_bonus=settings.structure_quality_min_score_for_bonus,
        structure_quality_max_bonus=settings.structure_quality_max_bonus,
        structure_quality_backtest_only=settings.structure_quality_backtest_only,
        allow_live_structure_quality_scoring=settings.allow_live_structure_quality_scoring,
        structure_quality_allowed_regimes=settings.structure_quality_allowed_regimes,
        structure_quality_allowed_pairs=settings.structure_quality_allowed_pairs,
        structure_quality_excluded_pairs=settings.structure_quality_excluded_pairs,
        live_exit_profile_enabled=settings.enable_exit_engine,
        live_exit_profile_preset=settings.exit_profile_preset,
        live_exit_use_regime_profiles=settings.exit_use_regime_profiles,
        live_exit_volatility_rr_enabled=settings.exit_volatility_rr_enabled,
        live_exit_volatility_rr_floor=settings.exit_volatility_rr_floor,
        live_exit_volatility_rr_cap=settings.exit_volatility_rr_cap,
        live_exit_liquidity_trailing_enabled=settings.exit_liquidity_trailing_enabled,
        live_exit_liquidity_lookback_bars=settings.exit_liquidity_lookback_bars,
        live_exit_liquidity_buffer_pips=settings.exit_liquidity_buffer_pips,
    )


def build_execution_settings(settings: Settings) -> RealisticExecutionSettings:
    return RealisticExecutionSettings(
        enabled=settings.enable_realistic_execution,
        spread_default_pips=settings.spread_default_pips,
        spread_by_pair=settings.spread_by_pair,
        slippage_mode=settings.slippage_mode,
        max_slippage_pips=settings.max_slippage_pips,
        execution_delay_bars=settings.execution_delay_bars,
        partial_fill_probability=settings.partial_fill_probability,
        partial_fill_min_ratio=settings.partial_fill_min_ratio,
        limit_touch_tolerance_pips=settings.limit_touch_tolerance_pips,
        apply_spread_to_limit=settings.apply_spread_to_limit,
        random_seed=settings.random_seed,
    )


def build_atr_risk_settings(settings: Settings) -> ATRRiskSettings:
    return ATRRiskSettings(
        enabled=settings.enable_atr_risk,
        period=settings.atr_period,
        multiplier=settings.atr_multiplier,
    )


def build_equity_protection_settings(settings: Settings) -> EquityProtectionSettings:
    return EquityProtectionSettings(
        enabled=settings.enable_equity_protection,
        max_drawdown_limit=settings.max_drawdown_limit,
        drawdown_risk_reduction_factor=settings.drawdown_risk_reduction_factor,
        max_consecutive_losses=settings.max_consecutive_losses,
        min_risk_multiplier=settings.min_risk_multiplier,
    )


def build_exit_settings(settings: Settings) -> AdaptiveExitSettings:
    return AdaptiveExitSettings(
        enabled=settings.enable_exit_engine,
        profile_preset=settings.exit_profile_preset,
        use_regime_profiles=settings.exit_use_regime_profiles,
        profile_overrides=settings.exit_profile_overrides,
        atr_trailing_enabled=settings.exit_atr_trailing_enabled,
        atr_trailing_period=settings.exit_atr_trailing_period,
        atr_trailing_multiplier=settings.exit_atr_trailing_multiplier,
        liquidity_trailing_enabled=settings.exit_liquidity_trailing_enabled,
        liquidity_lookback_bars=settings.exit_liquidity_lookback_bars,
        liquidity_buffer_pips=settings.exit_liquidity_buffer_pips,
        volatility_rr_enabled=settings.exit_volatility_rr_enabled,
        volatility_rr_floor=settings.exit_volatility_rr_floor,
        volatility_rr_cap=settings.exit_volatility_rr_cap,
    )


def build_sizing_settings(settings: Settings) -> AdaptiveSizingSettings:
    return AdaptiveSizingSettings(
        enabled=settings.enable_adaptive_sizing,
        min_multiplier=settings.sizing_min_multiplier,
        max_multiplier=settings.sizing_max_multiplier,
        confidence_floor_score=settings.sizing_confidence_floor_score,
        confidence_ceiling_score=settings.sizing_confidence_ceiling_score,
    )


def build_meta_label_settings(settings: Settings) -> MetaLabelSettings:
    return MetaLabelSettings(
        enabled=settings.enable_meta_label,
        mode=settings.meta_label_mode,
        probability_threshold=settings.meta_label_probability_threshold,
        enable_size_adjustment=settings.meta_label_enable_size_adjustment,
        low_probability_multiplier=settings.meta_label_low_probability_multiplier,
        high_probability_multiplier=settings.meta_label_high_probability_multiplier,
        high_probability_threshold=settings.meta_label_high_probability_threshold,
    )


def build_portfolio_layer_settings(settings: Settings) -> PortfolioLayerSettings:
    return PortfolioLayerSettings(
        enabled=settings.enable_portfolio_layer,
        mode=settings.portfolio_layer_mode,
        min_multiplier=settings.portfolio_layer_min_multiplier,
        max_multiplier=settings.portfolio_layer_max_multiplier,
        learning_window=settings.portfolio_layer_learning_window,
        min_trades_per_sleeve=settings.portfolio_layer_min_trades_per_sleeve,
        max_sleeve_concentration=settings.portfolio_layer_max_sleeve_concentration,
    )


def build_snapshot_cache_settings(settings: Settings) -> SnapshotCacheSettings:
    return SnapshotCacheSettings(
        enabled=settings.enable_backtest_snapshot_cache,
        max_entries=settings.backtest_snapshot_cache_max_entries,
    )


def build_trade_cache_settings(settings: Settings, args: argparse.Namespace | None = None) -> TradeCacheSettings:
    enabled = settings.enable_backtest_trade_cache
    cache_dir = settings.backtest_trade_cache_dir
    if args is not None:
        enabled = enabled or bool(getattr(args, "trade_cache", False))
        if getattr(args, "trade_cache_dir", None):
            cache_dir = str(args.trade_cache_dir)
    return TradeCacheSettings(
        enabled=enabled,
        cache_dir=cache_dir,
        version=settings.backtest_trade_cache_version,
    ).sanitized()


def resolve_backtest_end_time(settings: Settings, args: argparse.Namespace | None = None) -> str | None:
    raw = str(getattr(args, "end_time", "") or settings.backtest_end_time or "").strip()
    return raw or None


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


def build_trade_cache_key_payload(
    *,
    settings: Settings,
    args: argparse.Namespace,
    pairs: list[str],
    ltf: str,
    htf: str,
    trigger: str,
    evaluation_step: int,
    cache_mode: str,
    account_settings: BacktestAccountSettings,
    end_time: str | None,
) -> dict[str, object]:
    return {
        "runner": "backtest_runner",
        "pairs": pairs,
        "timeframes": {"ltf": ltf, "htf": htf, "trigger": trigger},
        "history_limit": int(args.history_limit),
        "max_hold_bars": int(args.max_hold_bars),
        "warmup_bars": int(args.warmup_bars),
        "evaluation_step": int(evaluation_step),
        "end_time": end_time,
        "cache_mode": cache_mode,
        "news_csv": args.news_csv or None,
        "account": asdict(account_settings),
        "settings": _safe_settings_payload(settings),
        "execution_settings": asdict(build_execution_settings(settings)),
        "atr_risk_settings": asdict(build_atr_risk_settings(settings)),
        "equity_protection_settings": asdict(build_equity_protection_settings(settings)),
        "exit_settings": asdict(build_exit_settings(settings).sanitized()),
        "sizing_settings": asdict(build_sizing_settings(settings)),
        "meta_label_settings": asdict(build_meta_label_settings(settings)),
        "portfolio_layer_settings": asdict(build_portfolio_layer_settings(settings)),
        "smc_research_feature_settings": asdict(build_smc_research_feature_settings(settings)),
    }


def build_smc_research_feature_settings(settings: Settings) -> SMCResearchFeatureSettings:
    return SMCResearchFeatureSettings(
        enabled=settings.enable_smc_research_features,
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
    )


def build_account_settings(settings: Settings, args: argparse.Namespace | None = None) -> BacktestAccountSettings:
    enabled = settings.backtest_account_enabled
    starting_balance = settings.backtest_starting_balance
    risk_per_trade = settings.backtest_risk_per_trade
    currency = settings.backtest_account_currency
    if args is not None:
        if getattr(args, "disable_account_report", False):
            enabled = False
        if getattr(args, "account_balance", None) is not None:
            starting_balance = float(args.account_balance)
        if getattr(args, "risk_per_trade", None) is not None:
            risk_per_trade = float(args.risk_per_trade)
        if getattr(args, "account_currency", None):
            currency = str(args.account_currency)
    return BacktestAccountSettings(
        enabled=enabled,
        starting_balance=starting_balance,
        risk_per_trade=risk_per_trade,
        currency=currency,
    ).sanitized()


def main() -> None:
    _configure_logging()
    settings = Settings.from_env()
    parser = build_parser()
    args = parser.parse_args()

    pairs = _parse_pairs(args.pairs) if args.pairs else settings.pairs
    ltf = args.ltf or settings.ltf_timeframe
    htf = args.htf or settings.htf_timeframe
    trigger = args.trigger or settings.trigger_timeframe
    evaluation_step = max(1, args.evaluation_step or settings.backtest_evaluation_step)
    account_settings = build_account_settings(settings, args)
    backtest_end_time = resolve_backtest_end_time(settings, args)
    trade_cache = BacktestTradeCache(build_trade_cache_settings(settings, args))
    score_analysis_requested = args.analyze_scores or args.dynamic_threshold_analysis or settings.enable_dynamic_threshold
    dynamic_threshold_analysis_enabled = args.dynamic_threshold_analysis or settings.enable_dynamic_threshold
    cache_mode = settings.market_data_cache_mode
    if args.cache_only:
        cache_mode = "cache_only"
    elif args.refresh_cache:
        cache_mode = "refresh"
    trade_cache_key_payload = build_trade_cache_key_payload(
        settings=settings,
        args=args,
        pairs=pairs,
        ltf=ltf,
        htf=htf,
        trigger=trigger,
        evaluation_step=evaluation_step,
        cache_mode=cache_mode,
        account_settings=account_settings,
        end_time=backtest_end_time,
    )
    trade_cache_key = trade_cache.build_key(trade_cache_key_payload)

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
    live_news = NewsFilter(
        blackout_before_minutes=settings.news_blackout_before_minutes,
        blackout_after_minutes=settings.news_blackout_after_minutes,
        surprise_threshold=settings.news_surprise_threshold,
    )

    if args.news_csv:
        news_feed = HistoricalNewsFeed.from_csv(
            args.news_csv,
            blackout_before_minutes=settings.news_blackout_before_minutes,
            blackout_after_minutes=settings.news_blackout_after_minutes,
            surprise_threshold=settings.news_surprise_threshold,
        )
    else:
        news_feed = NeutralNewsFeed()

    walk_forward_enabled = args.walk_forward or settings.walk_forward_enabled
    meta_optimize_enabled = args.meta_optimize
    complexity_penalty = args.complexity_penalty
    
    if walk_forward_enabled:
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
        engine = BacktestEngine(
            market_data=market_data,
            signal_engine=signal_engine,
            history_limit=args.history_limit,
            max_hold_bars=args.max_hold_bars,
            warmup_bars=args.warmup_bars,
            evaluation_step=evaluation_step,
            news_feed=news_feed,
            execution_settings=build_execution_settings(settings),
            atr_risk_settings=build_atr_risk_settings(settings),
            equity_protection_settings=build_equity_protection_settings(settings),
            exit_settings=build_exit_settings(settings),
            sizing_settings=build_sizing_settings(settings),
            meta_label_settings=build_meta_label_settings(settings),
            portfolio_layer_settings=build_portfolio_layer_settings(settings),
            snapshot_cache_settings=build_snapshot_cache_settings(settings),
            smc_research_feature_settings=build_smc_research_feature_settings(settings),
            account_settings=account_settings,
            end_time=backtest_end_time,
        )
        wf_runner = WalkForwardRunner(
            engine=engine,
            pairs=pairs,
            train_months=args.wf_train_months or settings.wf_train_months,
            test_months=args.wf_test_months or settings.wf_test_months,
            step_months=args.wf_step_months or settings.wf_step_months,
            timeframe_config={"ltf": ltf, "htf": htf, "trigger": trigger},
        )
        wf_result = wf_runner.run()
        print_walk_forward_report(wf_result)
        
        # Meta-optimization on walk-forward result (if enabled)
        if meta_optimize_enabled:
            print("\n" + "=" * 60)
            print("META-OPTIMIZATION (Training Folds Only)")
            print("=" * 60)
            
            opt_config = OptimizationConfig(
                enabled=True,
                optimize_threshold=True,
                optimize_atr=False,
                optimize_weights=False,
                complexity_penalty=complexity_penalty,
                min_train_trades=10,
                grid_resolution="coarse",
            )
            meta_result = run_meta_optimization(engine, wf_result, opt_config)
            print_meta_optimization_report(meta_result)
            
            if (settings.export_meta_report or args.export_meta_report) and not args.no_export:
                report_path = (Path(args.output_dir) if args.output_dir else Path("reports")) / "meta_walkforward_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                exported = meta_result.export(report_path)
                print(f"\nExported meta-optimization report: {exported}")
        
        if settings.export_reports and not args.no_export:
            report_path = (Path(args.output_dir) if args.output_dir else Path("reports")) / "walk_forward.json"
            exported = wf_result.export(report_path)
            print(f"\nExported walk-forward report: {exported}")
        return

    if args.validate_mitigation_entry:
        baseline_engine = BacktestEngine(
            market_data=market_data,
            signal_engine=build_signal_engine(
                market_data=market_data,
                news_filter=live_news,
                settings=settings,
                htf=htf,
                ltf=ltf,
                trigger=trigger,
                enable_shadow_scoring=True,
                enable_mitigation_entry=False,
            ),
            history_limit=args.history_limit,
            max_hold_bars=args.max_hold_bars,
            warmup_bars=args.warmup_bars,
            evaluation_step=evaluation_step,
            news_feed=news_feed,
            execution_settings=build_execution_settings(settings),
            atr_risk_settings=build_atr_risk_settings(settings),
            equity_protection_settings=build_equity_protection_settings(settings),
            exit_settings=build_exit_settings(settings),
            sizing_settings=build_sizing_settings(settings),
            meta_label_settings=build_meta_label_settings(settings),
            portfolio_layer_settings=build_portfolio_layer_settings(settings),
            snapshot_cache_settings=build_snapshot_cache_settings(settings),
            smc_research_feature_settings=build_smc_research_feature_settings(settings),
            account_settings=account_settings,
            end_time=backtest_end_time,
        )
        shadow_engine = BacktestEngine(
            market_data=market_data,
            signal_engine=build_signal_engine(
                market_data=market_data,
                news_filter=live_news,
                settings=settings,
                htf=htf,
                ltf=ltf,
                trigger=trigger,
                enable_shadow_scoring=True,
                enable_mitigation_entry=True,
            ),
            history_limit=args.history_limit,
            max_hold_bars=args.max_hold_bars,
            warmup_bars=args.warmup_bars,
            evaluation_step=evaluation_step,
            news_feed=news_feed,
            execution_settings=build_execution_settings(settings),
            atr_risk_settings=build_atr_risk_settings(settings),
            equity_protection_settings=build_equity_protection_settings(settings),
            exit_settings=build_exit_settings(settings),
            sizing_settings=build_sizing_settings(settings),
            meta_label_settings=build_meta_label_settings(settings),
            portfolio_layer_settings=build_portfolio_layer_settings(settings),
            snapshot_cache_settings=build_snapshot_cache_settings(settings),
            smc_research_feature_settings=build_smc_research_feature_settings(settings),
            account_settings=account_settings,
            end_time=backtest_end_time,
        )

        validator = BacktestValidationRunner(baseline_engine, shadow_engine)
        result = validator.run(pairs)
        print_validation_report(result)

        if score_analysis_requested:
            score_payload = build_validation_score_distribution_report(
                result,
                min_score=settings.min_score,
                dynamic_threshold_enabled=dynamic_threshold_analysis_enabled,
                threshold_percentile=settings.threshold_percentile,
                threshold_window=settings.threshold_rolling_window,
            )
            _print_score_distribution_report(score_payload.get("baseline", {}), "BASELINE SCORE ANALYSIS")
            _print_score_distribution_report(score_payload.get("shadow", {}), "SHADOW SCORE ANALYSIS")

            if settings.enable_feature_analytics:
                feature_payload = {
                    "baseline": _build_feature_analysis_payload(result.baseline),
                    "shadow": _build_feature_analysis_payload(result.shadow),
                }
                _print_feature_analysis_report(feature_payload["baseline"], "BASELINE FEATURE ANALYSIS")
                _print_feature_analysis_report(feature_payload["shadow"], "SHADOW FEATURE ANALYSIS")
            else:
                feature_payload = None

            if settings.export_reports and not args.no_export:
                reports_dir = Path("reports")
                _write_json_report(reports_dir / "score_distribution.json", score_payload)
                if feature_payload is not None:
                    _write_json_report(reports_dir / "feature_analysis.json", feature_payload)
                print(f"\nExported analytics reports to: {reports_dir.resolve()}")

        regime_payload = analyze_regime_performance_from_run(result.shadow)
        _print_regime_report(regime_payload, "SHADOW REGIME PERFORMANCE")
        if settings.export_regime_report and settings.export_reports and not args.no_export:
            reports_dir = Path("reports")
            exported_regime = export_regime_report(regime_payload, reports_dir / "regime_report.json")
            print(f"\nExported regime report to: {exported_regime.resolve()}")

        if args.monte_carlo:
            mc_payload = {
                "baseline": _build_monte_carlo_payload(result.baseline, args),
                "shadow": _build_monte_carlo_payload(result.shadow, args),
            }
            _print_monte_carlo_report(mc_payload["baseline"], "BASELINE MONTE CARLO")
            _print_monte_carlo_report(mc_payload["shadow"], "SHADOW MONTE CARLO")
            if settings.export_reports and not args.no_export:
                reports_dir = Path("reports")
                _write_json_report(reports_dir / "monte_carlo.json", mc_payload)

        if not args.no_export:
            export_dir = (
                Path(args.validation_output_dir)
                if args.validation_output_dir
                else Path("backtests") / "validation" / result.generated_at.strftime("%Y%m%d_%H%M%S")
            )
            exported_to = result.export(export_dir)
            print(f"\nExported validation to: {exported_to}")
    elif args.validate_shadow:
        baseline_engine = BacktestEngine(
            market_data=market_data,
            signal_engine=build_signal_engine(
                market_data=market_data,
                news_filter=live_news,
                settings=settings,
                htf=htf,
                ltf=ltf,
                trigger=trigger,
                enable_shadow_scoring=False,
                enable_mitigation_entry=False,
            ),
            history_limit=args.history_limit,
            max_hold_bars=args.max_hold_bars,
            warmup_bars=args.warmup_bars,
            evaluation_step=evaluation_step,
            news_feed=news_feed,
            execution_settings=build_execution_settings(settings),
            atr_risk_settings=build_atr_risk_settings(settings),
            equity_protection_settings=build_equity_protection_settings(settings),
            exit_settings=build_exit_settings(settings),
            sizing_settings=build_sizing_settings(settings),
            meta_label_settings=build_meta_label_settings(settings),
            portfolio_layer_settings=build_portfolio_layer_settings(settings),
            snapshot_cache_settings=build_snapshot_cache_settings(settings),
            smc_research_feature_settings=build_smc_research_feature_settings(settings),
            account_settings=account_settings,
            end_time=backtest_end_time,
        )
        shadow_engine = BacktestEngine(
            market_data=market_data,
            signal_engine=build_signal_engine(
                market_data=market_data,
                news_filter=live_news,
                settings=settings,
                htf=htf,
                ltf=ltf,
                trigger=trigger,
                enable_shadow_scoring=True,
                enable_mitigation_entry=False,
            ),
            history_limit=args.history_limit,
            max_hold_bars=args.max_hold_bars,
            warmup_bars=args.warmup_bars,
            evaluation_step=evaluation_step,
            news_feed=news_feed,
            execution_settings=build_execution_settings(settings),
            atr_risk_settings=build_atr_risk_settings(settings),
            equity_protection_settings=build_equity_protection_settings(settings),
            exit_settings=build_exit_settings(settings),
            sizing_settings=build_sizing_settings(settings),
            meta_label_settings=build_meta_label_settings(settings),
            portfolio_layer_settings=build_portfolio_layer_settings(settings),
            snapshot_cache_settings=build_snapshot_cache_settings(settings),
            smc_research_feature_settings=build_smc_research_feature_settings(settings),
            account_settings=account_settings,
            end_time=backtest_end_time,
        )

        validator = BacktestValidationRunner(baseline_engine, shadow_engine)
        result = validator.run(pairs)
        print_validation_report(result)

        if score_analysis_requested:
            score_payload = build_validation_score_distribution_report(
                result,
                min_score=settings.min_score,
                dynamic_threshold_enabled=dynamic_threshold_analysis_enabled,
                threshold_percentile=settings.threshold_percentile,
                threshold_window=settings.threshold_rolling_window,
            )
            _print_score_distribution_report(score_payload.get("baseline", {}), "BASELINE SCORE ANALYSIS")
            _print_score_distribution_report(score_payload.get("shadow", {}), "SHADOW SCORE ANALYSIS")

            if settings.enable_feature_analytics:
                feature_payload = {
                    "baseline": _build_feature_analysis_payload(result.baseline),
                    "shadow": _build_feature_analysis_payload(result.shadow),
                }
                _print_feature_analysis_report(feature_payload["baseline"], "BASELINE FEATURE ANALYSIS")
                _print_feature_analysis_report(feature_payload["shadow"], "SHADOW FEATURE ANALYSIS")
            else:
                feature_payload = None

            if settings.export_reports and not args.no_export:
                reports_dir = Path("reports")
                _write_json_report(reports_dir / "score_distribution.json", score_payload)
                if feature_payload is not None:
                    _write_json_report(reports_dir / "feature_analysis.json", feature_payload)
                print(f"\nExported analytics reports to: {reports_dir.resolve()}")

        regime_payload = analyze_regime_performance_from_run(result.shadow)
        _print_regime_report(regime_payload, "SHADOW REGIME PERFORMANCE")
        if settings.export_regime_report and settings.export_reports and not args.no_export:
            reports_dir = Path("reports")
            exported_regime = export_regime_report(regime_payload, reports_dir / "regime_report.json")
            print(f"\nExported regime report to: {exported_regime.resolve()}")

        if args.monte_carlo:
            mc_payload = {
                "baseline": _build_monte_carlo_payload(result.baseline, args),
                "shadow": _build_monte_carlo_payload(result.shadow, args),
            }
            _print_monte_carlo_report(mc_payload["baseline"], "BASELINE MONTE CARLO")
            _print_monte_carlo_report(mc_payload["shadow"], "SHADOW MONTE CARLO")
            if settings.export_reports and not args.no_export:
                reports_dir = Path("reports")
                _write_json_report(reports_dir / "monte_carlo.json", mc_payload)

        if not args.no_export:
            export_dir = (
                Path(args.validation_output_dir)
                if args.validation_output_dir
                else Path("backtests") / "validation" / result.generated_at.strftime("%Y%m%d_%H%M%S")
            )
            exported_to = result.export(export_dir)
            print(f"\nExported validation to: {exported_to}")
    else:
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

        engine = BacktestEngine(
            market_data=market_data,
            signal_engine=signal_engine,
            history_limit=args.history_limit,
            max_hold_bars=args.max_hold_bars,
            warmup_bars=args.warmup_bars,
            evaluation_step=evaluation_step,
            news_feed=news_feed,
            execution_settings=build_execution_settings(settings),
            atr_risk_settings=build_atr_risk_settings(settings),
            equity_protection_settings=build_equity_protection_settings(settings),
            exit_settings=build_exit_settings(settings),
            sizing_settings=build_sizing_settings(settings),
            meta_label_settings=build_meta_label_settings(settings),
            portfolio_layer_settings=build_portfolio_layer_settings(settings),
            snapshot_cache_settings=build_snapshot_cache_settings(settings),
            smc_research_feature_settings=build_smc_research_feature_settings(settings),
            account_settings=account_settings,
            end_time=backtest_end_time,
        )

        result = None
        if trade_cache.enabled and not args.refresh_trade_cache:
            result = trade_cache.load(trade_cache_key)
            if result is not None:
                print(f"Trade cache: HIT {result.parameters.get('trade_cache_path')}")

        if result is None:
            result = engine.run(pairs)
            if trade_cache.enabled:
                path = trade_cache.store(
                    trade_cache_key,
                    trade_cache_key_payload,
                    result,
                    metadata={"runner": "backtest_runner"},
                )
                if path is not None:
                    print(f"Trade cache: STORED {path}")

        print_report(result)

        realism_active = (
            settings.enable_realistic_execution
            or settings.enable_atr_risk
            or settings.enable_equity_protection
        )
        comparison_payload: dict[str, object] | None = None
        if realism_active:
            baseline_engine = BacktestEngine(
                market_data=market_data,
                signal_engine=build_signal_engine(
                    market_data=market_data,
                    news_filter=live_news,
                    settings=settings,
                    htf=htf,
                    ltf=ltf,
                    trigger=trigger,
                    enable_shadow_scoring=True,
                    enable_mitigation_entry=settings.enable_mitigation_entry,
                ),
                history_limit=args.history_limit,
                max_hold_bars=args.max_hold_bars,
                warmup_bars=args.warmup_bars,
                evaluation_step=evaluation_step,
                news_feed=news_feed,
                execution_settings=RealisticExecutionSettings(enabled=False),
                atr_risk_settings=ATRRiskSettings(enabled=False),
                equity_protection_settings=EquityProtectionSettings(enabled=False),
                exit_settings=build_exit_settings(settings),
                sizing_settings=build_sizing_settings(settings),
                meta_label_settings=build_meta_label_settings(settings),
                portfolio_layer_settings=build_portfolio_layer_settings(settings),
                snapshot_cache_settings=build_snapshot_cache_settings(settings),
                smc_research_feature_settings=build_smc_research_feature_settings(settings),
                account_settings=account_settings,
                end_time=backtest_end_time,
            )
            baseline_result = baseline_engine.run(pairs)
            comparison_payload = print_realistic_comparison(baseline_result, result)

        if score_analysis_requested:
            score_payload = build_score_distribution_report(
                result,
                min_score=settings.min_score,
                dynamic_threshold_enabled=dynamic_threshold_analysis_enabled,
                threshold_percentile=settings.threshold_percentile,
                threshold_window=settings.threshold_rolling_window,
            )
            _print_score_distribution_report(score_payload, "SCORE DISTRIBUTION ANALYSIS")
            if settings.enable_feature_analytics:
                feature_payload = _build_feature_analysis_payload(result)
                _print_feature_analysis_report(feature_payload, "FEATURE CONTRIBUTION ANALYSIS")
            else:
                feature_payload = None

            if settings.export_reports and not args.no_export:
                reports_dir = Path("reports")
                _write_json_report(reports_dir / "score_distribution.json", score_payload)
                if feature_payload is not None:
                    _write_json_report(reports_dir / "feature_analysis.json", feature_payload)
                print(f"\nExported analytics reports to: {reports_dir.resolve()}")

        regime_payload = analyze_regime_performance_from_run(result)
        _print_regime_report(regime_payload, "REGIME PERFORMANCE")
        if settings.export_regime_report and settings.export_reports and not args.no_export:
            reports_dir = Path("reports")
            exported_regime = export_regime_report(regime_payload, reports_dir / "regime_report.json")
            print(f"\nExported regime report to: {exported_regime.resolve()}")

        if args.monte_carlo:
            mc_payload = _build_monte_carlo_payload(result, args)
            _print_monte_carlo_report(mc_payload, "MONTE CARLO")
            if settings.export_reports and not args.no_export:
                reports_dir = Path("reports")
                _write_json_report(reports_dir / "monte_carlo.json", mc_payload)

        if comparison_payload is not None and settings.export_reports and not args.no_export:
            reports_dir = Path("reports")
            _write_json_report(reports_dir / "realistic_comparison.json", comparison_payload)
            print(f"\nExported realistic comparison report to: {(reports_dir / 'realistic_comparison.json').resolve()}")

        if not args.no_export:
            export_dir = Path(args.output_dir) if args.output_dir else Path("backtests") / result.started_at.strftime("%Y%m%d_%H%M%S")
            exported_to = result.export(export_dir)
            print(f"\nExported to: {exported_to}")


if __name__ == "__main__":
    main()
