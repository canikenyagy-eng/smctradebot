from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analytics.score_distribution import analyze_dynamic_threshold, analyze_rejections, analyze_scores
from backtest.engine import BacktestEngine, BacktestPairReport, BacktestRunResult
from backtest.portfolio_layer import PortfolioLayerState
from backtest.risk import EquityProtectionState


METRIC_KEYS = (
    "trades",
    "win_rate",
    "avg_r",
    "median_r",
    "avg_win_r",
    "avg_loss_r",
    "payoff_ratio",
    "expectancy_r",
    "sharpe_r",
    "profit_factor",
    "max_drawdown_r",
    "avg_score",
    "avg_shadow_bonus",
    "avg_bars_held",
    "limit_entries",
    "market_entries",
    "avg_fill_delay_bars",
    "partial_exits",
    "partial_fills",
    "break_even_activations",
    "trailing_activations",
    "tp_hits",
    "sl_hits",
    "timeout_exits",
    "avg_slippage_pips",
    "avg_spread_pips",
    "total_slippage_cost_r",
    "total_spread_cost_r",
    "avg_slippage_cost_r",
    "avg_spread_cost_r",
    "avg_delay_cost_r",
    "avg_risk_multiplier",
    "avg_sizing_multiplier",
    "avg_meta_probability",
    "meta_accepted_count",
    "avg_meta_size_multiplier",
    "avg_portfolio_multiplier",
    "realistic_execution_trades",
    "fill_rate",
    "partial_fill_rate",
    "acceptance_rate",
)


@dataclass(frozen=True)
class BacktestValidationResult:
    baseline: BacktestRunResult
    shadow: BacktestRunResult
    comparison: dict[str, object]
    pair_rows: list[dict[str, object]]
    generated_at: datetime

    def export(self, output_dir: str | Path) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        baseline_dir = target / "baseline"
        shadow_dir = target / "shadow"
        self.baseline.export(baseline_dir)
        self.shadow.export(shadow_dir)

        payload = {
            "generated_at": self.generated_at.isoformat(),
            "baseline": {
                "parameters": self.baseline.parameters,
                "overall": self.baseline.overall_metrics(),
            },
            "shadow": {
                "parameters": self.shadow.parameters,
                "overall": self.shadow.overall_metrics(),
            },
            "comparison": self.comparison,
            "pairs": self.pair_rows,
        }

        (target / "validation.json").write_text(
            json.dumps(_sanitize(payload), indent=2, default=str),
            encoding="utf-8",
        )

        if self.pair_rows:
            pd.DataFrame(self.pair_rows).to_csv(target / "validation_pairs.csv", index=False)

        return target


def _sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return BacktestRunResult._jsonable(value)


def _numeric_delta(baseline: object, shadow: object) -> dict[str, object]:
    base_value = baseline if isinstance(baseline, (int, float)) else None
    shadow_value = shadow if isinstance(shadow, (int, float)) else None

    if isinstance(base_value, float) and not math.isfinite(base_value):
        base_value = None
    if isinstance(shadow_value, float) and not math.isfinite(shadow_value):
        shadow_value = None

    delta: float | None = None
    pct_delta: float | None = None

    if base_value is not None and shadow_value is not None:
        delta = float(shadow_value) - float(base_value)
        if float(base_value) != 0.0:
            pct_delta = delta / abs(float(base_value))

    return {
        "baseline": baseline,
        "shadow": shadow,
        "delta": delta,
        "pct_delta": pct_delta,
    }


def _compare_metric_dict(baseline: dict[str, object], shadow: dict[str, object]) -> dict[str, dict[str, object]]:
    return {metric: _numeric_delta(baseline.get(metric), shadow.get(metric)) for metric in METRIC_KEYS}


def _build_rejection_delta(baseline: dict[str, int], shadow: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(baseline) | set(shadow))
    return {key: int(shadow.get(key, 0)) - int(baseline.get(key, 0)) for key in keys}


def _build_pair_row(baseline_row: dict[str, object], shadow_row: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "pair": baseline_row.get("pair") or shadow_row.get("pair"),
        "baseline_error": baseline_row.get("error"),
        "shadow_error": shadow_row.get("error"),
    }

    for metric in METRIC_KEYS:
        delta = _numeric_delta(baseline_row.get(metric), shadow_row.get(metric))
        row[f"baseline_{metric}"] = delta["baseline"]
        row[f"shadow_{metric}"] = delta["shadow"]
        row[f"delta_{metric}"] = delta["delta"]
        row[f"pct_delta_{metric}"] = delta["pct_delta"]

    baseline_rejections = baseline_row.get("rejections") if isinstance(baseline_row.get("rejections"), dict) else {}
    shadow_rejections = shadow_row.get("rejections") if isinstance(shadow_row.get("rejections"), dict) else {}
    row["baseline_rejections"] = baseline_rejections
    row["shadow_rejections"] = shadow_rejections
    row["rejection_delta"] = _build_rejection_delta(baseline_rejections, shadow_rejections)
    return row


def build_validation_result(
    baseline: BacktestRunResult,
    shadow: BacktestRunResult,
    *,
    generated_at: datetime | None = None,
) -> BacktestValidationResult:
    baseline_overall = baseline.overall_metrics()
    shadow_overall = shadow.overall_metrics()

    baseline_pairs = {row["pair"]: row for row in baseline.pair_rows()}
    shadow_pairs = {row["pair"]: row for row in shadow.pair_rows()}
    pair_names = sorted(set(baseline_pairs) | set(shadow_pairs))

    pair_rows = [
        _build_pair_row(baseline_pairs.get(pair, {"pair": pair}), shadow_pairs.get(pair, {"pair": pair}))
        for pair in pair_names
    ]

    comparison = {
        "baseline": baseline_overall,
        "shadow": shadow_overall,
        "deltas": _compare_metric_dict(baseline_overall, shadow_overall),
        "rejections_delta": _build_rejection_delta(
            baseline_overall.get("rejections", {}) if isinstance(baseline_overall.get("rejections"), dict) else {},
            shadow_overall.get("rejections", {}) if isinstance(shadow_overall.get("rejections"), dict) else {},
        ),
    }

    return BacktestValidationResult(
        baseline=baseline,
        shadow=shadow,
        comparison=comparison,
        pair_rows=pair_rows,
        generated_at=generated_at or datetime.now(timezone.utc),
    )


def build_score_distribution_report(
    run_result: BacktestRunResult,
    *,
    min_score: int = 70,
    bucket_size: int = 5,
    dynamic_threshold_enabled: bool = False,
    threshold_percentile: float = 80.0,
    threshold_window: int = 200,
) -> dict[str, Any]:
    overall = run_result.overall_metrics()
    evaluations = int(sum(report.evaluations for report in run_result.pair_reports))
    rejection_counts = overall.get("rejections", {})
    if not isinstance(rejection_counts, dict):
        rejection_counts = {}

    payload: dict[str, Any] = {
        "score_distribution": analyze_scores(
            run_result.score_observations,
            threshold=min_score,
            total_evaluations=evaluations,
            accepted_count=len(run_result.trades),
            bucket_size=bucket_size,
        ),
        "rejections": analyze_rejections(rejection_counts),
    }
    if dynamic_threshold_enabled:
        payload["dynamic_threshold"] = analyze_dynamic_threshold(
            run_result.score_observations,
            percentile=threshold_percentile,
            rolling_window=threshold_window,
        )
    return payload


def build_validation_score_distribution_report(
    result: BacktestValidationResult,
    *,
    min_score: int = 70,
    bucket_size: int = 5,
    dynamic_threshold_enabled: bool = False,
    threshold_percentile: float = 80.0,
    threshold_window: int = 200,
) -> dict[str, Any]:
    return {
        "baseline": build_score_distribution_report(
            result.baseline,
            min_score=min_score,
            bucket_size=bucket_size,
            dynamic_threshold_enabled=dynamic_threshold_enabled,
            threshold_percentile=threshold_percentile,
            threshold_window=threshold_window,
        ),
        "shadow": build_score_distribution_report(
            result.shadow,
            min_score=min_score,
            bucket_size=bucket_size,
            dynamic_threshold_enabled=dynamic_threshold_enabled,
            threshold_percentile=threshold_percentile,
            threshold_window=threshold_window,
        ),
    }


class BacktestValidationRunner:
    def __init__(self, baseline_engine: BacktestEngine, shadow_engine: BacktestEngine) -> None:
        self.baseline_engine = baseline_engine
        self.shadow_engine = shadow_engine

    @staticmethod
    def _error_report(pair: str, error: str) -> BacktestPairReport:
        return BacktestPairReport(
            pair=pair,
            trades=[],
            rejection_counts={},
            evaluations=0,
            bars_processed=0,
            error=error,
        )

    def run(self, pairs: Iterable[str]) -> BacktestValidationResult:
        pairs = list(pairs)
        pair_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        errors: dict[str, str] = {}

        for pair in pairs:
            try:
                pair_frames[pair] = self.baseline_engine.load_pair_frames(pair)
            except Exception as exc:
                errors[pair] = str(exc)
        universe = set(pair_frames.keys())

        baseline_started = datetime.now(timezone.utc)
        baseline_reports: list[BacktestPairReport] = []
        baseline_equity = (
            EquityProtectionState(self.baseline_engine.equity_protection_settings)
            if self.baseline_engine.equity_protection_settings.enabled
            else None
        )
        baseline_portfolio = PortfolioLayerState(self.baseline_engine.portfolio_layer_settings)
        for pair in pairs:
            error = errors.get(pair)
            if error is not None:
                baseline_reports.append(self._error_report(pair, error))
                continue

            ltf, htf, trigger = pair_frames[pair]
            try:
                reference_pair = self.baseline_engine.signal_engine._resolve_smt_reference_pair(pair, universe)
                reference_trigger = pair_frames.get(reference_pair, (None, None, None))[2] if reference_pair in pair_frames else None
                baseline_reports.append(
                    self.baseline_engine.run_pair_from_frames(
                        pair,
                        ltf,
                        htf,
                        trigger,
                        reference_pair=reference_pair,
                        reference_trigger=reference_trigger,
                        equity_state=baseline_equity,
                        portfolio_state=baseline_portfolio,
                    )
                )
            except Exception as exc:
                baseline_reports.append(self._error_report(pair, str(exc)))
        baseline_finished = datetime.now(timezone.utc)

        shadow_started = datetime.now(timezone.utc)
        shadow_reports: list[BacktestPairReport] = []
        shadow_equity = (
            EquityProtectionState(self.shadow_engine.equity_protection_settings)
            if self.shadow_engine.equity_protection_settings.enabled
            else None
        )
        shadow_portfolio = PortfolioLayerState(self.shadow_engine.portfolio_layer_settings)
        for pair in pairs:
            error = errors.get(pair)
            if error is not None:
                shadow_reports.append(self._error_report(pair, error))
                continue

            ltf, htf, trigger = pair_frames[pair]
            try:
                reference_pair = self.shadow_engine.signal_engine._resolve_smt_reference_pair(pair, universe)
                reference_trigger = pair_frames.get(reference_pair, (None, None, None))[2] if reference_pair in pair_frames else None
                shadow_reports.append(
                    self.shadow_engine.run_pair_from_frames(
                        pair,
                        ltf,
                        htf,
                        trigger,
                        reference_pair=reference_pair,
                        reference_trigger=reference_trigger,
                        equity_state=shadow_equity,
                        portfolio_state=shadow_portfolio,
                    )
                )
            except Exception as exc:
                shadow_reports.append(self._error_report(pair, str(exc)))
        shadow_finished = datetime.now(timezone.utc)

        baseline_result = BacktestRunResult(
            pair_reports=baseline_reports,
            parameters={
                "mode": "baseline",
                "shadow_scoring": self.baseline_engine.signal_engine.enable_shadow_scoring,
                "enable_shadow_scoring": self.baseline_engine.signal_engine.enable_shadow_scoring,
                "enable_mitigation_entry": self.baseline_engine.signal_engine.enable_mitigation_entry,
                "enable_adaptive_weights": self.baseline_engine.signal_engine.enable_adaptive_weights,
                "adaptive_regime_weights": self.baseline_engine.signal_engine._adaptive_weight_settings.regime_weights,
                "enable_score_normalization": self.baseline_engine.signal_engine._score_normalizer.settings.enabled,
                "score_normalization_method": self.baseline_engine.signal_engine._score_normalizer.settings.method,
                "score_normalization_window": self.baseline_engine.signal_engine._score_normalizer.settings.window,
                "score_normalization_scale_factor": self.baseline_engine.signal_engine._score_normalizer.settings.scale_factor,
                "score_normalization_backtest_only": self.baseline_engine.signal_engine._score_normalizer.settings.backtest_only,
                "allow_live_score_normalization": self.baseline_engine.signal_engine._score_normalizer.settings.allow_live,
                "enable_dynamic_threshold": self.baseline_engine.signal_engine._dynamic_threshold_tracker.settings.enabled,
                "threshold_percentile": self.baseline_engine.signal_engine._dynamic_threshold_tracker.settings.percentile,
                "threshold_rolling_window": self.baseline_engine.signal_engine._dynamic_threshold_tracker.settings.rolling_window,
                "apply_dynamic_threshold": self.baseline_engine.signal_engine._dynamic_threshold_tracker.settings.apply_threshold,
                "dynamic_threshold_backtest_only": self.baseline_engine.signal_engine._dynamic_threshold_tracker.settings.backtest_only,
                "allow_live_dynamic_threshold": self.baseline_engine.signal_engine._dynamic_threshold_tracker.settings.allow_live,
                "ltf_timeframe": self.baseline_engine.signal_engine.ltf_timeframe,
                "htf_timeframe": self.baseline_engine.signal_engine.htf_timeframe,
                "trigger_timeframe": self.baseline_engine.signal_engine.trigger_timeframe,
                "history_limit": self.baseline_engine.history_limit,
                "max_hold_bars": self.baseline_engine.max_hold_bars,
                "warmup_bars": self.baseline_engine.warmup_bars,
                "evaluation_step": self.baseline_engine.evaluation_step,
                "min_score": self.baseline_engine.signal_engine.min_score,
                "risk_reward": self.baseline_engine.signal_engine.risk_reward,
                "swing_window": self.baseline_engine.signal_engine.swing_window,
                "regime_short_window": self.baseline_engine.signal_engine.regime_short_window,
                "regime_long_window": self.baseline_engine.signal_engine.regime_long_window,
                "regime_opposition_confidence": self.baseline_engine.signal_engine.regime_opposition_confidence,
                "contraction_min_trigger_strength": self.baseline_engine.signal_engine.contraction_min_trigger_strength,
                "range_min_trigger_strength": self.baseline_engine.signal_engine.range_min_trigger_strength,
                "require_displacement_in_contraction": self.baseline_engine.signal_engine.require_displacement_in_contraction,
                "session_min_score": self.baseline_engine.signal_engine.session_min_score,
                "enable_smt_confirmation": self.baseline_engine.signal_engine.enable_smt_confirmation,
                "smt_hard_gate": self.baseline_engine.signal_engine.smt_hard_gate,
                "smt_min_strength": self.baseline_engine.signal_engine.smt_min_strength,
                "smt_opposite_block_strength": self.baseline_engine.signal_engine.smt_opposite_block_strength,
                "partial_tp_enabled": self.baseline_engine.signal_engine.partial_tp_enabled,
                "partial_tp_r": self.baseline_engine.signal_engine.partial_tp_r,
                "partial_tp_fraction": self.baseline_engine.signal_engine.partial_tp_fraction,
                "break_even_r": self.baseline_engine.signal_engine.break_even_r,
                "trailing_enabled": self.baseline_engine.signal_engine.trailing_enabled,
                "trailing_start_r": self.baseline_engine.signal_engine.trailing_start_r,
                "trailing_lookback_bars": self.baseline_engine.signal_engine.trailing_lookback_bars,
                "time_stop_bars": self.baseline_engine.signal_engine.time_stop_bars,
                "pair_correlation_threshold": self.baseline_engine.signal_engine.correlation_cap.threshold,
                "correlation_lookback": self.baseline_engine.signal_engine.correlation_cap.lookback,
                "currency_exposure_cap": self.baseline_engine.signal_engine.currency_exposure_cap,
                "portfolio_currency_gross_cap": self.baseline_engine.signal_engine.portfolio_currency_gross_cap,
                "portfolio_currency_net_cap": self.baseline_engine.signal_engine.portfolio_currency_net_cap,
                "portfolio_exposure_window_minutes": self.baseline_engine.signal_engine.portfolio_exposure_window_minutes,
                "pair_cooldown_minutes": self.baseline_engine.signal_engine.pair_cooldown_minutes,
                "max_entries_per_bias": self.baseline_engine.signal_engine.max_entries_per_bias,
                "bias_window_minutes": self.baseline_engine.signal_engine.bias_window_minutes,
                "enable_realistic_execution": self.baseline_engine.execution_settings.enabled,
                "spread_default_pips": self.baseline_engine.execution_settings.spread_default_pips,
                "spread_by_pair": self.baseline_engine.execution_settings.spread_by_pair,
                "slippage_mode": self.baseline_engine.execution_settings.slippage_mode,
                "max_slippage_pips": self.baseline_engine.execution_settings.max_slippage_pips,
                "execution_delay_bars": self.baseline_engine.execution_settings.execution_delay_bars,
                "partial_fill_probability": self.baseline_engine.execution_settings.partial_fill_probability,
                "partial_fill_min_ratio": self.baseline_engine.execution_settings.partial_fill_min_ratio,
                "limit_touch_tolerance_pips": self.baseline_engine.execution_settings.limit_touch_tolerance_pips,
                "apply_spread_to_limit": self.baseline_engine.execution_settings.apply_spread_to_limit,
                "random_seed": self.baseline_engine.execution_settings.random_seed,
                "enable_atr_risk": self.baseline_engine.atr_risk_settings.enabled,
                "atr_period": self.baseline_engine.atr_risk_settings.period,
                "atr_multiplier": self.baseline_engine.atr_risk_settings.multiplier,
                "enable_equity_protection": self.baseline_engine.equity_protection_settings.enabled,
                "max_drawdown_limit": self.baseline_engine.equity_protection_settings.max_drawdown_limit,
                "drawdown_risk_reduction_factor": self.baseline_engine.equity_protection_settings.drawdown_risk_reduction_factor,
                "max_consecutive_losses": self.baseline_engine.equity_protection_settings.max_consecutive_losses,
                "min_risk_multiplier": self.baseline_engine.equity_protection_settings.min_risk_multiplier,
                "enable_exit_engine": self.baseline_engine.exit_settings.enabled,
                "exit_use_regime_profiles": self.baseline_engine.exit_settings.use_regime_profiles,
                "exit_profile_overrides": self.baseline_engine.exit_settings.profile_overrides,
                "exit_atr_trailing_enabled": self.baseline_engine.exit_settings.atr_trailing_enabled,
                "exit_liquidity_trailing_enabled": self.baseline_engine.exit_settings.liquidity_trailing_enabled,
                "exit_volatility_rr_enabled": self.baseline_engine.exit_settings.volatility_rr_enabled,
                "enable_adaptive_sizing": self.baseline_engine.sizing_settings.enabled,
                "sizing_min_multiplier": self.baseline_engine.sizing_settings.min_multiplier,
                "sizing_max_multiplier": self.baseline_engine.sizing_settings.max_multiplier,
                "enable_meta_label": self.baseline_engine.meta_label_settings.enabled,
                "meta_label_mode": self.baseline_engine.meta_label_settings.mode,
                "meta_label_probability_threshold": self.baseline_engine.meta_label_settings.probability_threshold,
                "meta_label_enable_size_adjustment": self.baseline_engine.meta_label_settings.enable_size_adjustment,
                "enable_portfolio_layer": self.baseline_engine.portfolio_layer_settings.enabled,
                "portfolio_layer_mode": self.baseline_engine.portfolio_layer_settings.mode,
            },
            started_at=baseline_started,
            finished_at=baseline_finished,
            news_mode=self.baseline_engine.news_feed.__class__.__name__,
        )
        shadow_result = BacktestRunResult(
            pair_reports=shadow_reports,
            parameters={
                "mode": "shadow",
                "shadow_scoring": self.shadow_engine.signal_engine.enable_shadow_scoring,
                "enable_shadow_scoring": self.shadow_engine.signal_engine.enable_shadow_scoring,
                "enable_mitigation_entry": self.shadow_engine.signal_engine.enable_mitigation_entry,
                "enable_adaptive_weights": self.shadow_engine.signal_engine.enable_adaptive_weights,
                "adaptive_regime_weights": self.shadow_engine.signal_engine._adaptive_weight_settings.regime_weights,
                "enable_score_normalization": self.shadow_engine.signal_engine._score_normalizer.settings.enabled,
                "score_normalization_method": self.shadow_engine.signal_engine._score_normalizer.settings.method,
                "score_normalization_window": self.shadow_engine.signal_engine._score_normalizer.settings.window,
                "score_normalization_scale_factor": self.shadow_engine.signal_engine._score_normalizer.settings.scale_factor,
                "score_normalization_backtest_only": self.shadow_engine.signal_engine._score_normalizer.settings.backtest_only,
                "allow_live_score_normalization": self.shadow_engine.signal_engine._score_normalizer.settings.allow_live,
                "enable_dynamic_threshold": self.shadow_engine.signal_engine._dynamic_threshold_tracker.settings.enabled,
                "threshold_percentile": self.shadow_engine.signal_engine._dynamic_threshold_tracker.settings.percentile,
                "threshold_rolling_window": self.shadow_engine.signal_engine._dynamic_threshold_tracker.settings.rolling_window,
                "apply_dynamic_threshold": self.shadow_engine.signal_engine._dynamic_threshold_tracker.settings.apply_threshold,
                "dynamic_threshold_backtest_only": self.shadow_engine.signal_engine._dynamic_threshold_tracker.settings.backtest_only,
                "allow_live_dynamic_threshold": self.shadow_engine.signal_engine._dynamic_threshold_tracker.settings.allow_live,
                "ltf_timeframe": self.shadow_engine.signal_engine.ltf_timeframe,
                "htf_timeframe": self.shadow_engine.signal_engine.htf_timeframe,
                "trigger_timeframe": self.shadow_engine.signal_engine.trigger_timeframe,
                "history_limit": self.shadow_engine.history_limit,
                "max_hold_bars": self.shadow_engine.max_hold_bars,
                "warmup_bars": self.shadow_engine.warmup_bars,
                "evaluation_step": self.shadow_engine.evaluation_step,
                "min_score": self.shadow_engine.signal_engine.min_score,
                "risk_reward": self.shadow_engine.signal_engine.risk_reward,
                "swing_window": self.shadow_engine.signal_engine.swing_window,
                "regime_short_window": self.shadow_engine.signal_engine.regime_short_window,
                "regime_long_window": self.shadow_engine.signal_engine.regime_long_window,
                "regime_opposition_confidence": self.shadow_engine.signal_engine.regime_opposition_confidence,
                "contraction_min_trigger_strength": self.shadow_engine.signal_engine.contraction_min_trigger_strength,
                "range_min_trigger_strength": self.shadow_engine.signal_engine.range_min_trigger_strength,
                "require_displacement_in_contraction": self.shadow_engine.signal_engine.require_displacement_in_contraction,
                "session_min_score": self.shadow_engine.signal_engine.session_min_score,
                "enable_smt_confirmation": self.shadow_engine.signal_engine.enable_smt_confirmation,
                "smt_hard_gate": self.shadow_engine.signal_engine.smt_hard_gate,
                "smt_min_strength": self.shadow_engine.signal_engine.smt_min_strength,
                "smt_opposite_block_strength": self.shadow_engine.signal_engine.smt_opposite_block_strength,
                "partial_tp_enabled": self.shadow_engine.signal_engine.partial_tp_enabled,
                "partial_tp_r": self.shadow_engine.signal_engine.partial_tp_r,
                "partial_tp_fraction": self.shadow_engine.signal_engine.partial_tp_fraction,
                "break_even_r": self.shadow_engine.signal_engine.break_even_r,
                "trailing_enabled": self.shadow_engine.signal_engine.trailing_enabled,
                "trailing_start_r": self.shadow_engine.signal_engine.trailing_start_r,
                "trailing_lookback_bars": self.shadow_engine.signal_engine.trailing_lookback_bars,
                "time_stop_bars": self.shadow_engine.signal_engine.time_stop_bars,
                "pair_correlation_threshold": self.shadow_engine.signal_engine.correlation_cap.threshold,
                "correlation_lookback": self.shadow_engine.signal_engine.correlation_cap.lookback,
                "currency_exposure_cap": self.shadow_engine.signal_engine.currency_exposure_cap,
                "portfolio_currency_gross_cap": self.shadow_engine.signal_engine.portfolio_currency_gross_cap,
                "portfolio_currency_net_cap": self.shadow_engine.signal_engine.portfolio_currency_net_cap,
                "portfolio_exposure_window_minutes": self.shadow_engine.signal_engine.portfolio_exposure_window_minutes,
                "pair_cooldown_minutes": self.shadow_engine.signal_engine.pair_cooldown_minutes,
                "max_entries_per_bias": self.shadow_engine.signal_engine.max_entries_per_bias,
                "bias_window_minutes": self.shadow_engine.signal_engine.bias_window_minutes,
                "enable_realistic_execution": self.shadow_engine.execution_settings.enabled,
                "spread_default_pips": self.shadow_engine.execution_settings.spread_default_pips,
                "spread_by_pair": self.shadow_engine.execution_settings.spread_by_pair,
                "slippage_mode": self.shadow_engine.execution_settings.slippage_mode,
                "max_slippage_pips": self.shadow_engine.execution_settings.max_slippage_pips,
                "execution_delay_bars": self.shadow_engine.execution_settings.execution_delay_bars,
                "partial_fill_probability": self.shadow_engine.execution_settings.partial_fill_probability,
                "partial_fill_min_ratio": self.shadow_engine.execution_settings.partial_fill_min_ratio,
                "limit_touch_tolerance_pips": self.shadow_engine.execution_settings.limit_touch_tolerance_pips,
                "apply_spread_to_limit": self.shadow_engine.execution_settings.apply_spread_to_limit,
                "random_seed": self.shadow_engine.execution_settings.random_seed,
                "enable_atr_risk": self.shadow_engine.atr_risk_settings.enabled,
                "atr_period": self.shadow_engine.atr_risk_settings.period,
                "atr_multiplier": self.shadow_engine.atr_risk_settings.multiplier,
                "enable_equity_protection": self.shadow_engine.equity_protection_settings.enabled,
                "max_drawdown_limit": self.shadow_engine.equity_protection_settings.max_drawdown_limit,
                "drawdown_risk_reduction_factor": self.shadow_engine.equity_protection_settings.drawdown_risk_reduction_factor,
                "max_consecutive_losses": self.shadow_engine.equity_protection_settings.max_consecutive_losses,
                "min_risk_multiplier": self.shadow_engine.equity_protection_settings.min_risk_multiplier,
                "enable_exit_engine": self.shadow_engine.exit_settings.enabled,
                "exit_use_regime_profiles": self.shadow_engine.exit_settings.use_regime_profiles,
                "exit_profile_overrides": self.shadow_engine.exit_settings.profile_overrides,
                "exit_atr_trailing_enabled": self.shadow_engine.exit_settings.atr_trailing_enabled,
                "exit_liquidity_trailing_enabled": self.shadow_engine.exit_settings.liquidity_trailing_enabled,
                "exit_volatility_rr_enabled": self.shadow_engine.exit_settings.volatility_rr_enabled,
                "enable_adaptive_sizing": self.shadow_engine.sizing_settings.enabled,
                "sizing_min_multiplier": self.shadow_engine.sizing_settings.min_multiplier,
                "sizing_max_multiplier": self.shadow_engine.sizing_settings.max_multiplier,
                "enable_meta_label": self.shadow_engine.meta_label_settings.enabled,
                "meta_label_mode": self.shadow_engine.meta_label_settings.mode,
                "meta_label_probability_threshold": self.shadow_engine.meta_label_settings.probability_threshold,
                "meta_label_enable_size_adjustment": self.shadow_engine.meta_label_settings.enable_size_adjustment,
                "enable_portfolio_layer": self.shadow_engine.portfolio_layer_settings.enabled,
                "portfolio_layer_mode": self.shadow_engine.portfolio_layer_settings.mode,
            },
            started_at=shadow_started,
            finished_at=shadow_finished,
            news_mode=self.shadow_engine.news_feed.__class__.__name__,
        )

        return build_validation_result(
            baseline_result,
            shadow_result,
            generated_at=datetime.now(timezone.utc),
        )
