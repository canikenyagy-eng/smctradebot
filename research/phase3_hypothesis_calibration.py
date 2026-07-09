from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from backtest.engine import BacktestPairReport, BacktestRunResult, BacktestTrade
from backtest.snapshot_cache import SnapshotCache
from backtest.trade_cache import BacktestTradeCache
from config import Settings
from research.phase3_stress_validation import (
    STRICT_LTF_STEP3_PAIR_PROFILES,
    account_settings,
    build_engine,
    cache_key_payload,
    candidate_score,
    json_safe,
    parse_names,
    parse_pairs,
    result_payload,
    snapshot_cache_settings,
    stress_presets,
    trade_cache_settings,
    write_report,
)


@dataclass(frozen=True)
class HypothesisScenario:
    name: str
    description: str
    pair_profiles: dict[str, dict[str, object]]
    exit_profile_overrides: dict[str, dict[str, object]] | None = None
    enable_targeted_pre_trade_filter: bool = False
    pre_trade_block_expansion_continuation: bool = False
    post_trade_filter: str | None = None


def base_profiles() -> dict[str, dict[str, object]]:
    return {pair: dict(profile) for pair, profile in STRICT_LTF_STEP3_PAIR_PROFILES.items()}


def expansion_block_profiles() -> dict[str, dict[str, object]]:
    profiles = base_profiles()
    for profile in profiles.values():
        profile["regime_blocklist"] = "TREND,EXPANSION"
    return profiles


def fallback_strong_profiles() -> dict[str, dict[str, object]]:
    profiles = base_profiles()
    for profile in profiles.values():
        profile["market_fallback_min_trigger_strength"] = 16
        profile["market_fallback_require_displacement"] = True
    return profiles


def fallback_min_strength_profiles(min_strength: int) -> dict[str, dict[str, object]]:
    profiles = base_profiles()
    for profile in profiles.values():
        profile["market_fallback_min_trigger_strength"] = min_strength
        profile["market_fallback_require_displacement"] = False
    return profiles


def fallback_disabled_profiles() -> dict[str, dict[str, object]]:
    profiles = base_profiles()
    for profile in profiles.values():
        profile["allow_market_fallback"] = False
    return profiles


def timeout_fast_overrides() -> dict[str, dict[str, object]]:
    quick_reaction = {
        "target_rr": 1.20,
        "partials": [{"r": 0.60, "fraction": 0.65}],
        "break_even_r": 0.55,
        "trailing_enabled": True,
        "trailing_start_r": 0.85,
        "trailing_lookback_bars": 4,
        "time_stop_bars": 16,
    }
    return {
        "range": dict(quick_reaction),
        "contraction": dict(quick_reaction),
        "neutral": dict(quick_reaction),
    }


def scenarios() -> dict[str, HypothesisScenario]:
    return {
        "baseline": HypothesisScenario(
            name="baseline",
            description="strict_ltf_only + strict_ltf_step3_v1 unchanged.",
            pair_profiles=base_profiles(),
        ),
        "block_expansion": HypothesisScenario(
            name="block_expansion",
            description="Block EXPANSION in addition to TREND to test continuation/regime drag.",
            pair_profiles=expansion_block_profiles(),
        ),
        "fallback_strong": HypothesisScenario(
            name="fallback_strong",
            description="Allow MARKET fallback only with trigger_strength >= 16 and aligned displacement.",
            pair_profiles=fallback_strong_profiles(),
        ),
        "fallback_disabled": HypothesisScenario(
            name="fallback_disabled",
            description="Disable MARKET fallback entirely.",
            pair_profiles=fallback_disabled_profiles(),
        ),
        "timeout_fast": HypothesisScenario(
            name="timeout_fast",
            description="Earlier range/contraction partial, BE, trailing, and shorter time stop.",
            pair_profiles=base_profiles(),
            exit_profile_overrides=timeout_fast_overrides(),
        ),
        "timeout_fast_fallback_strong": HypothesisScenario(
            name="timeout_fast_fallback_strong",
            description="Timeout-fast exits plus stricter MARKET fallback confirmation.",
            pair_profiles=fallback_strong_profiles(),
            exit_profile_overrides=timeout_fast_overrides(),
        ),
        "timeout_fast_block_expansion": HypothesisScenario(
            name="timeout_fast_block_expansion",
            description="Timeout-fast exits plus EXPANSION regime block.",
            pair_profiles=expansion_block_profiles(),
            exit_profile_overrides=timeout_fast_overrides(),
        ),
        "timeout_fast_fallback_8": HypothesisScenario(
            name="timeout_fast_fallback_8",
            description="Timeout-fast exits plus soft MARKET fallback trigger-strength floor at 8.",
            pair_profiles=fallback_min_strength_profiles(8),
            exit_profile_overrides=timeout_fast_overrides(),
        ),
        "timeout_fast_fallback_10": HypothesisScenario(
            name="timeout_fast_fallback_10",
            description="Timeout-fast exits plus soft MARKET fallback trigger-strength floor at 10.",
            pair_profiles=fallback_min_strength_profiles(10),
            exit_profile_overrides=timeout_fast_overrides(),
        ),
        "timeout_fast_fallback_12": HypothesisScenario(
            name="timeout_fast_fallback_12",
            description="Timeout-fast exits plus soft MARKET fallback trigger-strength floor at 12.",
            pair_profiles=fallback_min_strength_profiles(12),
            exit_profile_overrides=timeout_fast_overrides(),
        ),
        "timeout_fast_fallback_14": HypothesisScenario(
            name="timeout_fast_fallback_14",
            description="Timeout-fast exits plus soft MARKET fallback trigger-strength floor at 14.",
            pair_profiles=fallback_min_strength_profiles(14),
            exit_profile_overrides=timeout_fast_overrides(),
        ),
        "timeout_fast_soft_fallback_no_expansion_continuation_fallback": HypothesisScenario(
            name="timeout_fast_soft_fallback_no_expansion_continuation_fallback",
            description="Timeout-fast soft fallback with pre-trade EXPANSION continuation fallback veto.",
            pair_profiles=fallback_min_strength_profiles(8),
            exit_profile_overrides=timeout_fast_overrides(),
            enable_targeted_pre_trade_filter=True,
        ),
        "timeout_fast_soft_fallback_no_expansion_continuation": HypothesisScenario(
            name="timeout_fast_soft_fallback_no_expansion_continuation",
            description="Timeout-fast soft fallback with pre-trade EXPANSION continuation veto for every entry source.",
            pair_profiles=fallback_min_strength_profiles(8),
            exit_profile_overrides=timeout_fast_overrides(),
            pre_trade_block_expansion_continuation=True,
        ),
    }


def scenario_settings(base: Settings, scenario: HypothesisScenario) -> Settings:
    return replace(
        base,
        enable_strict_ltf_direction_gate=True,
        enable_market_fallback_entry=True,
        market_fallback_min_trigger_strength=0,
        market_fallback_require_displacement=False,
        enable_pip_aware_liquidity=False,
        enable_structure_quality_scoring=False,
        structure_quality_replaces_raw_structure_score=False,
        enable_pair_profiles=True,
        pair_profiles=scenario.pair_profiles,
        pair_profiles_backtest_only=True,
        allow_live_pair_profiles=False,
        exit_profile_overrides=scenario.exit_profile_overrides,
        enable_pre_trade_filter=(
            scenario.enable_targeted_pre_trade_filter or scenario.pre_trade_block_expansion_continuation
        ),
        pre_trade_block_expansion_continuation=scenario.pre_trade_block_expansion_continuation,
        pre_trade_block_expansion_continuation_fallback=scenario.enable_targeted_pre_trade_filter,
    )


def should_filter_trade(trade: BacktestTrade, filter_name: str | None) -> bool:
    if filter_name != "no_expansion_continuation_fallback":
        return False
    return (
        (trade.regime_label or "").upper() == "EXPANSION"
        and (trade.portfolio_sleeve or "").lower() == "continuation"
        and (trade.entry_source or "").lower() == "fallback"
    )


def apply_post_trade_filter(result: BacktestRunResult, scenario: HypothesisScenario) -> BacktestRunResult:
    if not scenario.post_trade_filter:
        return result

    filtered_reports: list[BacktestPairReport] = []
    removed_total = 0
    removed_r_total = 0.0
    for report in result.pair_reports:
        kept: list[BacktestTrade] = []
        removed: list[BacktestTrade] = []
        for trade in report.trades:
            if should_filter_trade(trade, scenario.post_trade_filter):
                removed.append(trade)
            else:
                kept.append(trade)
        removed_total += len(removed)
        removed_r_total += sum(trade.r_multiple for trade in removed)
        rejection_counts = dict(report.rejection_counts or {})
        if removed:
            rejection_counts[f"post_trade_filter:{scenario.post_trade_filter}"] = (
                rejection_counts.get(f"post_trade_filter:{scenario.post_trade_filter}", 0) + len(removed)
            )
        filtered_reports.append(
            BacktestPairReport(
                pair=report.pair,
                trades=kept,
                rejection_counts=rejection_counts,
                evaluations=report.evaluations,
                bars_processed=report.bars_processed,
                account_settings=report.account_settings,
                error=report.error,
                regime_evaluations=report.regime_evaluations,
                regime_acceptances=report.regime_acceptances,
                score_observations=report.score_observations,
            )
        )

    parameters = dict(result.parameters or {})
    parameters["post_trade_filter"] = scenario.post_trade_filter
    parameters["post_trade_filter_removed_trades"] = removed_total
    parameters["post_trade_filter_removed_r"] = round(removed_r_total, 6)
    return BacktestRunResult(
        pair_reports=filtered_reports,
        parameters=parameters,
        started_at=result.started_at,
        finished_at=result.finished_at,
        news_mode=result.news_mode,
        account_settings=result.account_settings,
    )


def compact_run(run: dict[str, object]) -> dict[str, object]:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    monte_carlo = run.get("monte_carlo", {}) if isinstance(run.get("monte_carlo"), dict) else {}
    terminal = monte_carlo.get("terminal_r", {}) if isinstance(monte_carlo.get("terminal_r"), dict) else {}
    terminal_usd = monte_carlo.get("terminal_usd", {}) if isinstance(monte_carlo.get("terminal_usd"), dict) else {}
    return {
        "stress": run.get("scenario"),
        "candidate_score": run.get("candidate_score"),
        "trades": metrics.get("trades"),
        "profit_factor": metrics.get("profit_factor"),
        "avg_r": metrics.get("avg_r"),
        "net_pnl_usd": metrics.get("net_pnl_usd"),
        "max_drawdown_usd": metrics.get("max_drawdown_usd"),
        "fill_rate": metrics.get("fill_rate"),
        "mc_p05_r": terminal.get("p05"),
        "mc_p05_usd": terminal_usd.get("p05"),
        "positive_terminal_probability": monte_carlo.get("positive_terminal_probability"),
        "risk_of_ruin_probability": monte_carlo.get("risk_of_ruin_probability"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3.3 hypothesis calibration runner.")
    parser.add_argument("--pairs", default="EURUSD,EURJPY,CADJPY")
    parser.add_argument("--history-limit", type=int, default=3000)
    parser.add_argument("--evaluation-step", type=int, default=3)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    parser.add_argument("--warmup-bars", type=int, default=120)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--ltf", default=None)
    parser.add_argument("--htf", default=None)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--scenarios", default="baseline,block_expansion,fallback_strong,timeout_fast")
    parser.add_argument("--stress-presets", default="moderate,harsh")
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
    parser.add_argument("--output", default="reports/phase3_hypothesis_calibration.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Reuse Phase 3 stress helpers while keeping this runner's CLI focused on hypotheses.
    args.signal_profile = "strict_ltf_only"
    args.pair_profile_preset = "phase3_hypothesis"
    base_settings = Settings.from_env()
    available_scenarios = scenarios()
    selected_names = parse_names(args.scenarios, available_scenarios)
    missing_scenarios = [name for name in selected_names if name not in available_scenarios]
    if missing_scenarios:
        raise SystemExit(f"Unknown scenarios: {', '.join(missing_scenarios)}")
    selected_scenarios = [available_scenarios[name] for name in selected_names]

    available_stress = stress_presets(args.random_seed)
    selected_stress_names = parse_names(args.stress_presets, available_stress)
    missing_stress = [name for name in selected_stress_names if name not in available_stress]
    if missing_stress:
        raise SystemExit(f"Unknown stress presets: {', '.join(missing_stress)}")
    selected_stress = [available_stress[name] for name in selected_stress_names]

    pairs = parse_pairs(args.pairs)
    account = account_settings(args)
    output = Path(args.output)
    snapshot_settings = snapshot_cache_settings(base_settings, args)
    snapshot_cache = SnapshotCache(snapshot_settings)
    report: dict[str, object] = {
        "runner": "research.phase3_hypothesis_calibration",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "settings": {
            "pairs": pairs,
            "history_limit": max(150, int(args.history_limit)),
            "evaluation_step": max(1, int(args.evaluation_step)),
            "scenarios": [scenario.name for scenario in selected_scenarios],
            "stress_presets": [preset.name for preset in selected_stress],
            "account": asdict(account),
            "monte_carlo": {
                "iterations": max(100, int(args.mc_iterations)),
                "seed": int(args.mc_seed),
                "ruin_drawdown_r": max(0.1, float(args.mc_ruin_dd)),
            },
            "snapshot_cache": asdict(snapshot_settings),
        },
        "runs": [],
        "ranking": [],
        "snapshot_cache": snapshot_cache.stats(),
    }

    print(
        "Phase 3.3 hypothesis calibration | pairs={pairs} bars={bars} step={step} scenarios={scenarios} stress={stress}".format(
            pairs=",".join(pairs),
            bars=max(150, int(args.history_limit)),
            step=max(1, int(args.evaluation_step)),
            scenarios=",".join(scenario.name for scenario in selected_scenarios),
            stress=",".join(preset.name for preset in selected_stress),
        ),
        flush=True,
    )

    for scenario in selected_scenarios:
        settings = scenario_settings(base_settings, scenario)
        trade_cache = BacktestTradeCache(trade_cache_settings(settings, args))
        for preset in selected_stress:
            print(f"\n=== {scenario.name} | {preset.name} ===", flush=True)
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
            if scenario.name == "baseline":
                key_payload["pair_profile_preset"] = "strict_ltf_step3_v1"
            else:
                key_payload["hypothesis_scenario"] = scenario.name
            if scenario.post_trade_filter:
                key_payload["post_trade_filter"] = scenario.post_trade_filter
            cache_key = trade_cache.build_key(key_payload)
            result = None
            if trade_cache.enabled and not args.refresh_trade_cache:
                result = trade_cache.load(cache_key)
                if result is not None:
                    print(f"Trade cache: HIT {result.parameters.get('trade_cache_path')}", flush=True)
            if result is None:
                result = apply_post_trade_filter(engine.run(pairs), scenario)
                if trade_cache.enabled:
                    path = trade_cache.store(
                        cache_key,
                        key_payload,
                        result,
                        metadata={
                            "runner": "research.phase3_hypothesis_calibration",
                            "scenario": scenario.name,
                            "stress_preset": preset.name,
                        },
                    )
                    if path is not None:
                        print(f"Trade cache: STORED {path}", flush=True)

            payload = result_payload(preset=preset, result=result, account=account, args=args)
            payload["hypothesis"] = {
                "name": scenario.name,
                "description": scenario.description,
                "pair_profiles": scenario.pair_profiles,
                "exit_profile_overrides": scenario.exit_profile_overrides,
                "enable_targeted_pre_trade_filter": scenario.enable_targeted_pre_trade_filter,
                "pre_trade_block_expansion_continuation": scenario.pre_trade_block_expansion_continuation,
                "post_trade_filter": scenario.post_trade_filter,
                "post_trade_filter_removed_trades": result.parameters.get("post_trade_filter_removed_trades", 0),
                "post_trade_filter_removed_r": result.parameters.get("post_trade_filter_removed_r", 0.0),
            }
            payload["candidate_score"] = candidate_score(payload.get("metrics", {}), payload.get("monte_carlo", {}))
            run_row = {
                "scenario": scenario.name,
                "stress": preset.name,
                "description": scenario.description,
                **compact_run(payload),
                "payload": payload,
            }
            report["runs"].append(run_row)  # type: ignore[union-attr]
            report["snapshot_cache"] = snapshot_cache.stats()
            report["ranking"] = sorted(
                [
                    {
                        key: row.get(key)
                        for key in (
                            "scenario",
                            "stress",
                            "candidate_score",
                            "trades",
                            "profit_factor",
                            "avg_r",
                            "net_pnl_usd",
                            "max_drawdown_usd",
                            "mc_p05_usd",
                            "positive_terminal_probability",
                        )
                    }
                    for row in report.get("runs", [])
                    if isinstance(row, dict)
                ],
                key=lambda row: float(row.get("candidate_score") or -999.0),
                reverse=True,
            )
            write_report(output, report)
            print(
                "{scenario:18s} {stress:8s} trades={trades:>3} pf={pf:>5.2f} avg_r={avg_r:>7.3f} "
                "net=${net:>8.2f} dd=${dd:>7.2f} mc_p05=${p05:>8.2f} score={score:.3f}".format(
                    scenario=scenario.name,
                    stress=preset.name,
                    trades=int(run_row.get("trades") or 0),
                    pf=float(run_row.get("profit_factor") or 0.0),
                    avg_r=float(run_row.get("avg_r") or 0.0),
                    net=float(run_row.get("net_pnl_usd") or 0.0),
                    dd=float(run_row.get("max_drawdown_usd") or 0.0),
                    p05=float(run_row.get("mc_p05_usd") or 0.0),
                    score=float(run_row.get("candidate_score") or 0.0),
                ),
                flush=True,
            )

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["snapshot_cache"] = snapshot_cache.stats()
    write_report(output, report)
    print(f"\nSaved Phase 3.3 calibration report: {output}", flush=True)
    print("\nRANKING", flush=True)
    for row in report.get("ranking", [])[:20]:
        if not isinstance(row, dict):
            continue
        print(
            "{scenario:18s} {stress:8s} score={score:.3f} pf={pf:.2f} avg_r={avg_r:.3f} net=${net:+.2f} dd=${dd:.2f} mc_p05=${p05:+.2f}".format(
                scenario=str(row.get("scenario")),
                stress=str(row.get("stress")),
                score=float(row.get("candidate_score") or 0.0),
                pf=float(row.get("profit_factor") or 0.0),
                avg_r=float(row.get("avg_r") or 0.0),
                net=float(row.get("net_pnl_usd") or 0.0),
                dd=float(row.get("max_drawdown_usd") or 0.0),
                p05=float(row.get("mc_p05_usd") or 0.0),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
