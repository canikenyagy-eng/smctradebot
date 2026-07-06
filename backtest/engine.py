from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable
import json
import math

import pandas as pd

from backtest.exit_engine import AdaptiveExitEngine, AdaptiveExitSettings
from backtest.meta_label import MetaLabelDecision, MetaLabelEngine, MetaLabelSettings
from backtest.portfolio_layer import PortfolioDecision, PortfolioLayerSettings, PortfolioLayerState
from backtest.execution import RealisticExecutionSettings, build_rng, volatility_slippage_factor
from backtest.risk import ATRRiskSettings, EquityProtectionSettings, EquityProtectionState, atr_value_at
from backtest.sizing import AdaptiveSizingEngine, AdaptiveSizingSettings, SizingDecision
from backtest.smc_research_features import SMCResearchFeatureSettings, extract_smc_research_features
from backtest.snapshot_cache import SnapshotCache, SnapshotCacheKey, SnapshotCacheSettings
from core.signal_engine import SignalEngine, SignalEvaluation, TradeSignal
from data.market_data import MarketDataClient
from execution.news import NewsAssessment
from backtest.news import NeutralNewsFeed


def expectancy_stats(r_values: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in r_values]
    if not values:
        return {
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_r": 0.0,
            "sharpe_r": 0.0,
        }

    wins = [value for value in values if value > 0]
    losses = [abs(value) for value in values if value < 0]
    win_rate = len(wins) / len(values)
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)
    expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
    volatility = pstdev(values) if len(values) > 1 else 0.0
    sharpe = (mean(values) / volatility) if volatility > 0 else 0.0
    return {
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy_r": expectancy,
        "sharpe_r": sharpe,
    }


@dataclass(frozen=True)
class BacktestAccountSettings:
    enabled: bool = True
    starting_balance: float = 1000.0
    risk_per_trade: float = 50.0
    currency: str = "USD"

    def sanitized(self) -> "BacktestAccountSettings":
        return BacktestAccountSettings(
            enabled=bool(self.enabled),
            starting_balance=max(0.0, float(self.starting_balance)),
            risk_per_trade=max(0.0, float(self.risk_per_trade)),
            currency=(self.currency or "USD").upper().strip(),
        )


def account_money_stats(r_values: Iterable[float], settings: BacktestAccountSettings | None) -> dict[str, object]:
    cfg = (settings or BacktestAccountSettings(enabled=False)).sanitized()
    if not cfg.enabled:
        return {}

    values = [float(value) for value in r_values]
    pnl_values = [value * cfg.risk_per_trade for value in values]
    gross_profit = sum(value for value in pnl_values if value > 0)
    gross_loss = abs(sum(value for value in pnl_values if value < 0))
    net_pnl = sum(pnl_values)
    equity = cfg.starting_balance
    peak = equity
    max_drawdown = 0.0
    min_balance = equity
    for pnl in pnl_values:
        equity += pnl
        peak = max(peak, equity)
        min_balance = min(min_balance, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return {
        "accounting_enabled": True,
        "account_currency": cfg.currency,
        "starting_balance": round(cfg.starting_balance, 2),
        "risk_per_trade": round(cfg.risk_per_trade, 2),
        "risk_per_trade_pct_start": round((cfg.risk_per_trade / cfg.starting_balance * 100.0), 4)
        if cfg.starting_balance > 0
        else 0.0,
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "final_balance_usd": round(equity, 2),
        "min_balance_usd": round(min_balance, 2),
        "max_drawdown_usd": round(max_drawdown, 2),
        "max_drawdown_pct": round((max_drawdown / peak * 100.0), 4) if peak > 0 else 0.0,
        "roi_pct": round((net_pnl / cfg.starting_balance * 100.0), 4) if cfg.starting_balance > 0 else 0.0,
        "avg_trade_pnl_usd": round(mean(pnl_values), 2) if pnl_values else 0.0,
        "median_trade_pnl_usd": round(median(pnl_values), 2) if pnl_values else 0.0,
        "expectancy_usd": round(mean(pnl_values), 2) if pnl_values else 0.0,
    }


@dataclass(frozen=True)
class BacktestTrade:
    pair: str
    side: str
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    entry_index: int
    exit_index: int
    entry: float
    stop_loss: float
    take_profit: float
    exit_price: float
    exit_reason: str
    r_multiple: float
    bars_held: int
    score: int
    htf_bias: str
    regime_label: str
    regime_direction: str
    zone: str
    trigger_direction: str
    trigger_event: str
    trigger_strength: int
    structure_event: str
    structure_trend: str
    score_htf: int
    score_regime: int
    score_trigger: int
    score_liquidity: int
    score_zone: int
    score_news: int
    score_session: int
    score_fvg: int
    score_order_block: int
    score_mitigation: int
    score_smt: int
    shadow_bonus: int
    entry_mode: str
    entry_source: str
    fill_delay_bars: int
    partial_taken: bool
    break_even_activated: bool
    trailing_activated: bool
    feature_breakdown: dict[str, int]
    raw_r_multiple: float
    risk_multiplier: float
    sizing_multiplier: float
    meta_probability: float
    meta_accepted: bool
    meta_mode: str
    meta_size_multiplier: float
    meta_blocked: bool
    portfolio_sleeve: str
    portfolio_multiplier: float
    portfolio_applied: bool
    atr_stop_applied: bool
    atr_value: float | None
    realistic_execution: bool
    spread_pips: float
    spread_cost_r: float
    slippage_pips: float
    slippage_cost_r: float
    execution_delay_bars: int
    execution_delay_cost_r: float
    fill_ratio: float
    partial_fill: bool
    exit_engine_mode: str
    exit_profile: str
    exit_target_rr: float
    exit_partial_plan: str
    atr_trailing_activated: bool
    liquidity_trailing_activated: bool
    smc_features: dict[str, object]


@dataclass
class BacktestPairReport:
    pair: str
    trades: list[BacktestTrade]
    rejection_counts: dict[str, int]
    evaluations: int
    bars_processed: int
    account_settings: BacktestAccountSettings | None = None
    error: str | None = None
    regime_evaluations: dict[str, int] | None = None
    regime_acceptances: dict[str, int] | None = None
    score_observations: list[int] | None = None

    def metrics(self) -> dict[str, object]:
        r_values = [trade.r_multiple for trade in self.trades]
        expectancy = expectancy_stats(r_values)
        trade_count = len(r_values)
        wins = sum(1 for value in r_values if value > 0)
        losses = sum(1 for value in r_values if value < 0)
        breakeven = sum(1 for value in r_values if value == 0)
        gross_profit = sum(value for value in r_values if value > 0)
        gross_loss = abs(sum(value for value in r_values if value < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        win_rate = wins / trade_count if trade_count else 0.0
        avg_r = mean(r_values) if r_values else 0.0
        median_r = median(r_values) if r_values else 0.0
        avg_score = mean(trade.score for trade in self.trades) if self.trades else 0.0
        avg_shadow_bonus = mean(trade.shadow_bonus for trade in self.trades) if self.trades else 0.0
        avg_bars_held = mean(trade.bars_held for trade in self.trades) if self.trades else 0.0
        limit_entries = sum(1 for trade in self.trades if trade.entry_mode == "MITIGATION_LIMIT")
        market_entries = sum(1 for trade in self.trades if trade.entry_mode == "MARKET")
        avg_fill_delay_bars = mean(trade.fill_delay_bars for trade in self.trades) if self.trades else 0.0
        partial_exits = sum(1 for trade in self.trades if trade.partial_taken)
        break_even_activations = sum(1 for trade in self.trades if trade.break_even_activated)
        trailing_activations = sum(1 for trade in self.trades if trade.trailing_activated)
        atr_trailing_activations = sum(1 for trade in self.trades if trade.atr_trailing_activated)
        liquidity_trailing_activations = sum(1 for trade in self.trades if trade.liquidity_trailing_activated)
        adaptive_exit_trades = sum(1 for trade in self.trades if trade.exit_engine_mode != "legacy")
        partial_fills = sum(1 for trade in self.trades if trade.partial_fill)
        tp_hits = sum(1 for trade in self.trades if trade.exit_reason == "take_profit")
        sl_hits = sum(1 for trade in self.trades if trade.exit_reason in {"stop_loss", "trailing_stop", "break_even_stop"})
        timeout_exits = sum(1 for trade in self.trades if trade.exit_reason == "timeout")
        avg_slippage_pips = mean(trade.slippage_pips for trade in self.trades) if self.trades else 0.0
        avg_spread_pips = mean(trade.spread_pips for trade in self.trades) if self.trades else 0.0
        total_slippage_cost_r = sum(trade.slippage_cost_r for trade in self.trades)
        total_spread_cost_r = sum(trade.spread_cost_r for trade in self.trades)
        avg_slippage_cost_r = mean(trade.slippage_cost_r for trade in self.trades) if self.trades else 0.0
        avg_spread_cost_r = mean(trade.spread_cost_r for trade in self.trades) if self.trades else 0.0
        avg_delay_cost_r = mean(trade.execution_delay_cost_r for trade in self.trades) if self.trades else 0.0
        avg_risk_multiplier = mean(trade.risk_multiplier for trade in self.trades) if self.trades else 1.0
        avg_sizing_multiplier = mean(trade.sizing_multiplier for trade in self.trades) if self.trades else 1.0
        avg_meta_probability = mean(trade.meta_probability for trade in self.trades) if self.trades else 1.0
        meta_accepted_count = sum(1 for trade in self.trades if trade.meta_accepted)
        avg_meta_size_multiplier = mean(trade.meta_size_multiplier for trade in self.trades) if self.trades else 1.0
        avg_portfolio_multiplier = mean(trade.portfolio_multiplier for trade in self.trades) if self.trades else 1.0
        portfolio_sleeves: Counter[str] = Counter(trade.portfolio_sleeve for trade in self.trades)
        avg_exit_target_rr = mean(trade.exit_target_rr for trade in self.trades) if self.trades else 0.0
        realistic_trades = sum(1 for trade in self.trades if trade.realistic_execution)
        attempted_fills = trade_count + int(self.rejection_counts.get("entry_not_filled", 0))
        fill_rate = trade_count / attempted_fills if attempted_fills > 0 else 0.0
        partial_fill_rate = partial_fills / trade_count if trade_count > 0 else 0.0
        acceptance_rate = trade_count / self.evaluations if self.evaluations else 0.0
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for value in r_values:
            equity += value
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)

        metrics = {
            "pair": self.pair,
            "trades": trade_count,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": win_rate,
            "avg_r": avg_r,
            "median_r": median_r,
            "avg_win_r": expectancy["avg_win_r"],
            "avg_loss_r": expectancy["avg_loss_r"],
            "payoff_ratio": expectancy["payoff_ratio"],
            "expectancy_r": expectancy["expectancy_r"],
            "sharpe_r": expectancy["sharpe_r"],
            "profit_factor": profit_factor,
            "max_drawdown_r": max_drawdown,
            "avg_score": avg_score,
            "avg_shadow_bonus": avg_shadow_bonus,
            "avg_bars_held": avg_bars_held,
            "limit_entries": limit_entries,
            "market_entries": market_entries,
            "avg_fill_delay_bars": avg_fill_delay_bars,
            "partial_exits": partial_exits,
            "break_even_activations": break_even_activations,
            "trailing_activations": trailing_activations,
            "atr_trailing_activations": atr_trailing_activations,
            "liquidity_trailing_activations": liquidity_trailing_activations,
            "adaptive_exit_trades": adaptive_exit_trades,
            "partial_fills": partial_fills,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "timeout_exits": timeout_exits,
            "avg_slippage_pips": avg_slippage_pips,
            "avg_spread_pips": avg_spread_pips,
            "total_slippage_cost_r": total_slippage_cost_r,
            "total_spread_cost_r": total_spread_cost_r,
            "avg_slippage_cost_r": avg_slippage_cost_r,
            "avg_spread_cost_r": avg_spread_cost_r,
            "avg_delay_cost_r": avg_delay_cost_r,
            "avg_risk_multiplier": avg_risk_multiplier,
            "avg_sizing_multiplier": avg_sizing_multiplier,
            "avg_meta_probability": avg_meta_probability,
            "meta_accepted_count": meta_accepted_count,
            "avg_meta_size_multiplier": avg_meta_size_multiplier,
            "avg_portfolio_multiplier": avg_portfolio_multiplier,
            "portfolio_sleeves": dict(portfolio_sleeves),
            "avg_exit_target_rr": avg_exit_target_rr,
            "realistic_execution_trades": realistic_trades,
            "fill_rate": fill_rate,
            "partial_fill_rate": partial_fill_rate,
            "evaluations": self.evaluations,
            "acceptance_rate": acceptance_rate,
            "rejections": dict(self.rejection_counts),
            "regime_evaluations": dict(self.regime_evaluations or {}),
            "regime_acceptances": dict(self.regime_acceptances or {}),
            "score_observations": list(self.score_observations or []),
            "error": self.error,
        }
        metrics.update(account_money_stats(r_values, self.account_settings))
        return metrics


@dataclass
class BacktestRunResult:
    pair_reports: list[BacktestPairReport]
    parameters: dict[str, object]
    started_at: datetime
    finished_at: datetime
    news_mode: str
    account_settings: BacktestAccountSettings | None = None

    @property
    def trades(self) -> list[BacktestTrade]:
        items: list[BacktestTrade] = []
        for report in self.pair_reports:
            items.extend(report.trades)
        return items

    @property
    def score_observations(self) -> list[int]:
        scores: list[int] = []
        for report in self.pair_reports:
            scores.extend(report.score_observations or [])
        if scores:
            return scores
        return [int(trade.score) for trade in self.trades]

    def overall_metrics(self) -> dict[str, object]:
        trades = self.trades
        r_values = [trade.r_multiple for trade in trades]
        expectancy = expectancy_stats(r_values)
        trade_count = len(r_values)
        wins = sum(1 for value in r_values if value > 0)
        losses = sum(1 for value in r_values if value < 0)
        breakeven = sum(1 for value in r_values if value == 0)
        gross_profit = sum(value for value in r_values if value > 0)
        gross_loss = abs(sum(value for value in r_values if value < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        win_rate = wins / trade_count if trade_count else 0.0
        avg_r = mean(r_values) if r_values else 0.0
        median_r = median(r_values) if r_values else 0.0
        avg_score = mean(trade.score for trade in trades) if trades else 0.0
        avg_shadow_bonus = mean(trade.shadow_bonus for trade in trades) if trades else 0.0
        avg_bars_held = mean(trade.bars_held for trade in trades) if trades else 0.0
        limit_entries = sum(1 for trade in trades if trade.entry_mode == "MITIGATION_LIMIT")
        market_entries = sum(1 for trade in trades if trade.entry_mode == "MARKET")
        avg_fill_delay_bars = mean(trade.fill_delay_bars for trade in trades) if trades else 0.0
        partial_exits = sum(1 for trade in trades if trade.partial_taken)
        break_even_activations = sum(1 for trade in trades if trade.break_even_activated)
        trailing_activations = sum(1 for trade in trades if trade.trailing_activated)
        atr_trailing_activations = sum(1 for trade in trades if trade.atr_trailing_activated)
        liquidity_trailing_activations = sum(1 for trade in trades if trade.liquidity_trailing_activated)
        adaptive_exit_trades = sum(1 for trade in trades if trade.exit_engine_mode != "legacy")
        partial_fills = sum(1 for trade in trades if trade.partial_fill)
        tp_hits = sum(1 for trade in trades if trade.exit_reason == "take_profit")
        sl_hits = sum(1 for trade in trades if trade.exit_reason in {"stop_loss", "trailing_stop", "break_even_stop"})
        timeout_exits = sum(1 for trade in trades if trade.exit_reason == "timeout")
        avg_slippage_pips = mean(trade.slippage_pips for trade in trades) if trades else 0.0
        avg_spread_pips = mean(trade.spread_pips for trade in trades) if trades else 0.0
        total_slippage_cost_r = sum(trade.slippage_cost_r for trade in trades)
        total_spread_cost_r = sum(trade.spread_cost_r for trade in trades)
        avg_slippage_cost_r = mean(trade.slippage_cost_r for trade in trades) if trades else 0.0
        avg_spread_cost_r = mean(trade.spread_cost_r for trade in trades) if trades else 0.0
        avg_delay_cost_r = mean(trade.execution_delay_cost_r for trade in trades) if trades else 0.0
        avg_risk_multiplier = mean(trade.risk_multiplier for trade in trades) if trades else 1.0
        avg_sizing_multiplier = mean(trade.sizing_multiplier for trade in trades) if trades else 1.0
        avg_meta_probability = mean(trade.meta_probability for trade in trades) if trades else 1.0
        meta_accepted_count = sum(1 for trade in trades if trade.meta_accepted)
        avg_meta_size_multiplier = mean(trade.meta_size_multiplier for trade in trades) if trades else 1.0
        avg_portfolio_multiplier = mean(trade.portfolio_multiplier for trade in trades) if trades else 1.0
        portfolio_sleeves: Counter[str] = Counter(trade.portfolio_sleeve for trade in trades)
        avg_exit_target_rr = mean(trade.exit_target_rr for trade in trades) if trades else 0.0
        realistic_trades = sum(1 for trade in trades if trade.realistic_execution)

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for value in r_values:
            equity += value
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)

        rejection_totals: Counter[str] = Counter()
        regime_eval_totals: Counter[str] = Counter()
        regime_accept_totals: Counter[str] = Counter()
        for report in self.pair_reports:
            rejection_totals.update(report.rejection_counts)
            regime_eval_totals.update(report.regime_evaluations or {})
            regime_accept_totals.update(report.regime_acceptances or {})
        attempted_fills = trade_count + int(rejection_totals.get("entry_not_filled", 0))
        fill_rate = trade_count / attempted_fills if attempted_fills > 0 else 0.0
        partial_fill_rate = partial_fills / trade_count if trade_count > 0 else 0.0

        metrics = {
            "trades": trade_count,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": win_rate,
            "avg_r": avg_r,
            "median_r": median_r,
            "avg_win_r": expectancy["avg_win_r"],
            "avg_loss_r": expectancy["avg_loss_r"],
            "payoff_ratio": expectancy["payoff_ratio"],
            "expectancy_r": expectancy["expectancy_r"],
            "sharpe_r": expectancy["sharpe_r"],
            "profit_factor": profit_factor,
            "max_drawdown_r": max_drawdown,
            "avg_score": avg_score,
            "avg_shadow_bonus": avg_shadow_bonus,
            "avg_bars_held": avg_bars_held,
            "limit_entries": limit_entries,
            "market_entries": market_entries,
            "avg_fill_delay_bars": avg_fill_delay_bars,
            "partial_exits": partial_exits,
            "break_even_activations": break_even_activations,
            "trailing_activations": trailing_activations,
            "atr_trailing_activations": atr_trailing_activations,
            "liquidity_trailing_activations": liquidity_trailing_activations,
            "adaptive_exit_trades": adaptive_exit_trades,
            "partial_fills": partial_fills,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "timeout_exits": timeout_exits,
            "avg_slippage_pips": avg_slippage_pips,
            "avg_spread_pips": avg_spread_pips,
            "total_slippage_cost_r": total_slippage_cost_r,
            "total_spread_cost_r": total_spread_cost_r,
            "avg_slippage_cost_r": avg_slippage_cost_r,
            "avg_spread_cost_r": avg_spread_cost_r,
            "avg_delay_cost_r": avg_delay_cost_r,
            "avg_risk_multiplier": avg_risk_multiplier,
            "avg_sizing_multiplier": avg_sizing_multiplier,
            "avg_meta_probability": avg_meta_probability,
            "meta_accepted_count": meta_accepted_count,
            "avg_meta_size_multiplier": avg_meta_size_multiplier,
            "avg_portfolio_multiplier": avg_portfolio_multiplier,
            "portfolio_sleeves": dict(portfolio_sleeves),
            "avg_exit_target_rr": avg_exit_target_rr,
            "realistic_execution_trades": realistic_trades,
            "fill_rate": fill_rate,
            "partial_fill_rate": partial_fill_rate,
            "rejections": dict(rejection_totals),
            "regime_evaluations": dict(regime_eval_totals),
            "regime_acceptances": dict(regime_accept_totals),
            "news_mode": self.news_mode,
        }
        metrics.update(account_money_stats(r_values, self.account_settings))
        return metrics

    def pair_rows(self) -> list[dict[str, object]]:
        return [report.metrics() for report in self.pair_reports]

    def trade_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        account = (self.account_settings or BacktestAccountSettings(enabled=False)).sanitized()
        equity = account.starting_balance
        for trade in self.trades:
            row = asdict(trade)
            if account.enabled:
                pnl = float(trade.r_multiple) * account.risk_per_trade
                equity += pnl
                row["pnl_usd"] = round(pnl, 2)
                row["equity_usd"] = round(equity, 2)
                row["risk_per_trade_usd"] = round(account.risk_per_trade, 2)
                row["starting_balance_usd"] = round(account.starting_balance, 2)
                row["account_currency"] = account.currency
            rows.append(row)
        return rows

    @staticmethod
    def _jsonable(value: object) -> object:
        if isinstance(value, float) and math.isinf(value):
            return None
        if isinstance(value, Counter):
            return dict(value)
        return value

    def export(self, output_dir: str | Path) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        summary = {
            "parameters": self.parameters,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "news_mode": self.news_mode,
            "overall": self.overall_metrics(),
            "pairs": self.pair_rows(),
        }

        for key, value in list(summary["overall"].items()):
            summary["overall"][key] = self._jsonable(value)
        for row in summary["pairs"]:
            for key, value in list(row.items()):
                row[key] = self._jsonable(value)

        (target / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        trades_df = pd.DataFrame(self.trade_rows())
        if not trades_df.empty:
            trades_df.to_csv(target / "trades.csv", index=False)

        pairs_df = pd.DataFrame(self.pair_rows())
        if not pairs_df.empty:
            pairs_df.to_csv(target / "pair_summary.csv", index=False)

        return target


class BacktestEngine:
    def __init__(
        self,
        market_data: MarketDataClient,
        signal_engine: SignalEngine,
        *,
        history_limit: int = 3000,
        max_hold_bars: int = 48,
        warmup_bars: int = 120,
        evaluation_step: int = 1,
        news_feed: object | None = None,
        execution_settings: RealisticExecutionSettings | None = None,
        atr_risk_settings: ATRRiskSettings | None = None,
        equity_protection_settings: EquityProtectionSettings | None = None,
        exit_settings: AdaptiveExitSettings | None = None,
        sizing_settings: AdaptiveSizingSettings | None = None,
        meta_label_settings: MetaLabelSettings | None = None,
        portfolio_layer_settings: PortfolioLayerSettings | None = None,
        snapshot_cache_settings: SnapshotCacheSettings | None = None,
        snapshot_cache: SnapshotCache | None = None,
        smc_research_feature_settings: SMCResearchFeatureSettings | None = None,
        account_settings: BacktestAccountSettings | None = None,
        end_time: object | None = None,
    ) -> None:
        self.market_data = market_data
        self.signal_engine = signal_engine
        self.history_limit = history_limit
        self.max_hold_bars = max(1, max_hold_bars)
        self.warmup_bars = max(80, warmup_bars)
        self.evaluation_step = max(1, int(evaluation_step))
        self.news_feed = news_feed or NeutralNewsFeed()
        self.execution_settings = (execution_settings or RealisticExecutionSettings()).sanitized()
        self.atr_risk_settings = (atr_risk_settings or ATRRiskSettings()).sanitized()
        self.equity_protection_settings = (equity_protection_settings or EquityProtectionSettings()).sanitized()
        self.exit_settings = (exit_settings or AdaptiveExitSettings()).sanitized()
        self.sizing_settings = (sizing_settings or AdaptiveSizingSettings()).sanitized()
        self.meta_label_settings = (meta_label_settings or MetaLabelSettings()).sanitized()
        self.portfolio_layer_settings = (portfolio_layer_settings or PortfolioLayerSettings()).sanitized()
        self.snapshot_cache_settings = (snapshot_cache_settings or SnapshotCacheSettings()).sanitized()
        self.smc_research_feature_settings = (smc_research_feature_settings or SMCResearchFeatureSettings()).sanitized()
        self.account_settings = (account_settings or BacktestAccountSettings()).sanitized()
        self.end_time = self._coerce_end_time(end_time)
        self.snapshot_cache = snapshot_cache or SnapshotCache(self.snapshot_cache_settings)
        self.exit_engine = AdaptiveExitEngine(self.exit_settings)
        self.sizing_engine = AdaptiveSizingEngine(self.sizing_settings)
        self.meta_label_engine = MetaLabelEngine(self.meta_label_settings)
        self._rng = build_rng(self.execution_settings.random_seed)
        self._snapshot_config_key = self._build_snapshot_config_key()

    def _evaluation_step_for_pair(self, pair: str) -> int:
        profile_getter = getattr(self.signal_engine, "pair_runtime_profile", None)
        profile = profile_getter(pair) if callable(profile_getter) else None
        step = getattr(profile, "evaluation_step", None)
        if step is None:
            return self.evaluation_step
        return max(1, int(step))

    @staticmethod
    def _coerce_end_time(end_time: object | None) -> pd.Timestamp | None:
        if end_time is None:
            return None
        text = str(end_time).strip()
        if not text:
            return None
        timestamp = pd.Timestamp(text)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def load_pair_frames(self, pair: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ltf = self.market_data.fetch_ohlcv(
            pair,
            self.signal_engine.ltf_timeframe,
            limit=self.history_limit,
            end_time=self.end_time,
        )
        htf = self.market_data.fetch_ohlcv(
            pair,
            self.signal_engine.htf_timeframe,
            limit=self.history_limit,
            end_time=self.end_time,
        )
        trigger = self.market_data.fetch_ohlcv(
            pair,
            self.signal_engine.trigger_timeframe,
            limit=self.history_limit,
            end_time=self.end_time,
        )
        return ltf, htf, trigger

    @staticmethod
    def _evaluate_news(feed: object, pair: str, as_of: datetime) -> NewsAssessment:
        if hasattr(feed, "evaluate"):
            return feed.evaluate(pair, as_of)  # type: ignore[no-any-return]
        return NewsAssessment(
            allow_trading=True,
            score=15,
            uncertainty="neutral",
            summary="Backtest neutral news assumption",
            high_impact_events=0,
        )

    @staticmethod
    def _stable_json(value: object) -> str:
        return json.dumps(value, sort_keys=True, default=str)

    @staticmethod
    def _news_cache_key(news: NewsAssessment) -> tuple[object, ...]:
        return (
            bool(news.allow_trading),
            int(news.score),
            str(news.uncertainty),
            str(news.summary),
            int(news.high_impact_events),
        )

    def _build_snapshot_config_key(self) -> str:
        engine = self.signal_engine
        payload = {
            "min_score": engine.min_score,
            "risk_reward": engine.risk_reward,
            "swing_window": engine.swing_window,
            "regime_short_window": engine.regime_short_window,
            "regime_long_window": engine.regime_long_window,
            "enable_shadow_scoring": engine.enable_shadow_scoring,
            "enable_mitigation_entry": engine.enable_mitigation_entry,
            "regime_opposition_confidence": engine.regime_opposition_confidence,
            "contraction_min_trigger_strength": engine.contraction_min_trigger_strength,
            "range_min_trigger_strength": engine.range_min_trigger_strength,
            "require_displacement_in_contraction": engine.require_displacement_in_contraction,
            "enable_strict_ltf_direction_gate": engine.enable_strict_ltf_direction_gate,
            "enable_market_fallback_entry": engine.enable_market_fallback_entry,
            "market_fallback_min_trigger_strength": engine.market_fallback_min_trigger_strength,
            "market_fallback_require_displacement": engine.market_fallback_require_displacement,
            "enable_pip_aware_liquidity": engine.enable_pip_aware_liquidity,
            "liquidity_equal_level_tolerance_pips": engine.liquidity_equal_level_tolerance_pips,
            "liquidity_atr_tolerance_factor": engine.liquidity_atr_tolerance_factor,
            "session_min_score": engine.session_min_score,
            "enable_smt_confirmation": engine.enable_smt_confirmation,
            "enable_order_block_shadow": engine.enable_order_block_shadow,
            "smt_hard_gate": engine.smt_hard_gate,
            "smt_min_strength": engine.smt_min_strength,
            "smt_opposite_block_strength": engine.smt_opposite_block_strength,
            "smt_reference_map": engine.smt_reference_map,
            "partial_tp_enabled": engine.partial_tp_enabled,
            "partial_tp_r": engine.partial_tp_r,
            "partial_tp_fraction": engine.partial_tp_fraction,
            "break_even_r": engine.break_even_r,
            "trailing_enabled": engine.trailing_enabled,
            "trailing_start_r": engine.trailing_start_r,
            "trailing_lookback_bars": engine.trailing_lookback_bars,
            "time_stop_bars": engine.time_stop_bars,
            "enable_adaptive_weights": engine.enable_adaptive_weights,
            "adaptive_regime_weights": engine._adaptive_weight_settings.regime_weights,
            "structure_quality_replaces_raw_structure_score": engine.structure_quality_replaces_raw_structure_score,
            "structure_quality": asdict(engine._structure_quality_settings),
        }
        return self._stable_json(payload)

    def _snapshot_cache_disabled_reason(self) -> str | None:
        if not self.snapshot_cache.enabled:
            return "disabled"
        if self.signal_engine._score_normalizer.settings.enabled:
            return "score normalization is stateful"
        if self.signal_engine._dynamic_threshold_tracker.settings.enabled:
            return "dynamic threshold is stateful"
        return None

    def _snapshot_cache_key(
        self,
        *,
        pair: str,
        trigger_time: pd.Timestamp,
        trigger_end: int,
        ltf_end: int,
        htf_end: int,
        reference_pair: str | None,
        reference_end: int,
        news_assessment: NewsAssessment,
    ) -> SnapshotCacheKey:
        return (
            "signal_snapshot_v1",
            self._snapshot_config_key,
            pair.upper().replace("/", ""),
            int(trigger_time.value),
            int(trigger_end),
            int(ltf_end),
            int(htf_end),
            reference_pair or "",
            int(reference_end),
            *self._news_cache_key(news_assessment),
        )

    @staticmethod
    def _volatility_ratio(frame: pd.DataFrame, index: int, short_window: int = 20, long_window: int = 80) -> float:
        if frame.empty or index <= 0:
            return 1.0
        scoped = frame.iloc[: index + 1]
        closes = scoped["close"].astype(float)
        returns = closes.pct_change().dropna()
        if len(returns) < max(short_window, long_window):
            return 1.0
        short_vol = float(returns.tail(short_window).std(ddof=0) or 0.0)
        long_vol = float(returns.tail(long_window).std(ddof=0) or 0.0)
        if long_vol <= 1e-9:
            return 1.0
        return max(0.5, min(1.5, short_vol / long_vol))

    def _slippage_pips(self, frame: pd.DataFrame, index: int) -> float:
        settings = self.execution_settings
        if not settings.enabled or settings.max_slippage_pips <= 0:
            return 0.0

        mode = settings.slippage_mode
        if mode == "none":
            return 0.0
        if mode == "random":
            return float(self._rng.uniform(0.0, settings.max_slippage_pips))
        if mode == "volatility":
            factor = volatility_slippage_factor(frame, index=index)
            jitter = self._rng.uniform(0.6, 1.0)
            return float(settings.max_slippage_pips * factor * jitter)
        return 0.0

    def _market_execution_price(
        self,
        *,
        pair: str,
        side: str,
        raw_entry: float,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float, float]:
        settings = self.execution_settings
        spread_pips = settings.spread_pips(pair)
        slippage_pips = self._slippage_pips(frame, index)
        pip_size = settings.pip_size(pair)

        spread_adjust = (spread_pips * pip_size) * 0.5
        slippage_adjust = slippage_pips * pip_size
        total_adjust = spread_adjust + slippage_adjust

        if side == "BUY":
            return raw_entry + total_adjust, spread_pips, slippage_pips
        return raw_entry - total_adjust, spread_pips, slippage_pips

    def _simulate_trade(
        self,
        pair: str,
        signal: TradeSignal,
        frame: pd.DataFrame,
        signal_index: int,
        *,
        risk_multiplier: float = 1.0,
        sizing_decision: SizingDecision | None = None,
        meta_decision: MetaLabelDecision | None = None,
        portfolio_decision: PortfolioDecision | None = None,
        feature_frame: pd.DataFrame | None = None,
    ) -> tuple[BacktestTrade | None, str | None]:
        execution = self.execution_settings
        realistic_enabled = execution.enabled
        entry_mode = signal.entry_mode.upper()
        fill_index = signal_index
        entry_time = signal.generated_at
        fill_delay_bars = 0
        execution_delay_bars = execution.execution_delay_bars if realistic_enabled else 0
        fill_ratio = 1.0
        partial_fill = False
        spread_pips = 0.0
        slippage_pips = 0.0
        atr_stop_applied = False
        atr_value = None
        raw_entry = signal.entry
        entry = signal.entry
        sizing = sizing_decision or SizingDecision(
            multiplier=1.0,
            confidence_component=1.0,
            regime_component=1.0,
            volatility_component=1.0,
            drawdown_component=1.0,
        )
        meta = meta_decision or MetaLabelDecision(
            enabled=False,
            probability=1.0,
            accepted=True,
            mode="analysis_only",
            size_multiplier=1.0,
            reason="disabled",
            features={},
        )
        portfolio = portfolio_decision or PortfolioDecision(
            sleeve="unassigned",
            multiplier=1.0,
            applied=False,
            reason="disabled",
        )

        if entry_mode == "MITIGATION_LIMIT":
            limit_fill_index: int | None = None
            tolerance = execution.tolerance_price(pair) if realistic_enabled else 0.0
            start_index = signal_index + 1 + execution_delay_bars
            max_fill_index = len(frame) - 1
            for idx in range(start_index, max_fill_index + 1):
                candle = frame.iloc[idx]
                high = float(candle["high"])
                low = float(candle["low"])

                if signal.side == "BUY" and low <= (signal.entry + tolerance) and high >= (signal.entry - tolerance):
                    limit_fill_index = idx
                    break
                if signal.side == "SELL" and high >= (signal.entry - tolerance) and low <= (signal.entry + tolerance):
                    limit_fill_index = idx
                    break

            if limit_fill_index is None:
                return None, "entry_not_filled"

            fill_index = limit_fill_index
            entry_time = frame.index[fill_index].to_pydatetime()
            fill_delay_bars = fill_index - signal_index
            raw_entry = signal.entry
            entry = raw_entry

            if realistic_enabled:
                if execution.apply_spread_to_limit:
                    adjusted, spread_pips, slippage_pips = self._market_execution_price(
                        pair=pair,
                        side=signal.side,
                        raw_entry=raw_entry,
                        frame=frame,
                        index=fill_index,
                    )
                    entry = adjusted
                elif execution.max_slippage_pips > 0 and execution.slippage_mode != "none":
                    slippage_pips = self._slippage_pips(frame, fill_index)
                    slippage_price = slippage_pips * execution.pip_size(pair)
                    entry = raw_entry + slippage_price if signal.side == "BUY" else raw_entry - slippage_price

                if self._rng.random() > execution.partial_fill_probability:
                    fill_ratio = float(self._rng.uniform(execution.partial_fill_min_ratio, 0.99))
                    partial_fill = True
        else:
            if realistic_enabled:
                fill_index = min(len(frame) - 1, signal_index + execution_delay_bars)
                if fill_index < 0 or fill_index >= len(frame):
                    return None, "entry_not_filled"
                fill_delay_bars = fill_index - signal_index
                entry_time = frame.index[fill_index].to_pydatetime()
                raw_entry = float(frame.iloc[fill_index]["close"])
                entry, spread_pips, slippage_pips = self._market_execution_price(
                    pair=pair,
                    side=signal.side,
                    raw_entry=raw_entry,
                    frame=frame,
                    index=fill_index,
                )

        initial_stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        atr_settings = self.atr_risk_settings
        if atr_settings.enabled:
            atr_value = atr_value_at(frame, fill_index, period=atr_settings.period)
            if atr_value is not None:
                atr_distance = atr_value * atr_settings.multiplier
                if atr_distance > 0:
                    base_risk = max(abs(signal.entry - signal.stop_loss), 1e-9)
                    rr = abs(signal.take_profit - signal.entry) / base_risk
                    if signal.side == "BUY":
                        initial_stop_loss = entry - atr_distance
                        take_profit = entry + atr_distance * rr
                    else:
                        initial_stop_loss = entry + atr_distance
                        take_profit = entry - atr_distance * rr
                    atr_stop_applied = True

        risk = abs(entry - initial_stop_loss)
        if risk <= 0:
            risk = max(entry * 0.0001, 1e-9)

        volatility_ratio = self._volatility_ratio(frame, fill_index)
        exit_plan = self.exit_engine.build_plan(
            pair=pair,
            signal=signal,
            entry=entry,
            stop_loss=initial_stop_loss,
            take_profit=take_profit,
            risk=risk,
            volatility_ratio=volatility_ratio,
        )
        take_profit = float(exit_plan.take_profit)
        managed_hold_bars = exit_plan.time_stop_bars if exit_plan.time_stop_bars > 0 else (
            signal.time_stop_bars if signal.time_stop_bars > 0 else self.max_hold_bars
        )
        max_exit_index = min(len(frame) - 1, fill_index + max(1, managed_hold_bars))
        if max_exit_index <= fill_index:
            return None, "entry_not_filled"

        stop_loss = initial_stop_loss
        partial_levels = [
            {"price": float(level.price), "fraction": float(level.fraction), "taken": False}
            for level in exit_plan.partial_targets
        ]
        partial_taken = False
        break_even_activated = False
        trailing_activated = False
        atr_trailing_activated = False
        liquidity_trailing_activated = False
        remaining = fill_ratio
        realized_r = 0.0

        exit_index = max_exit_index
        exit_price = float(frame.iloc[exit_index]["close"])
        exit_reason = "timeout"

        for idx in range(fill_index + 1, max_exit_index + 1):
            candle = frame.iloc[idx]
            high = float(candle["high"])
            low = float(candle["low"])
            close = float(candle["close"])

            if signal.side == "BUY":
                stop_hit = low <= stop_loss
                if stop_hit:
                    exit_index = idx
                    exit_price = stop_loss
                    hit_r = (stop_loss - entry) / risk
                    realized_r += remaining * hit_r
                    remaining = 0.0
                    if trailing_activated and stop_loss > initial_stop_loss:
                        exit_reason = "trailing_stop"
                    elif break_even_activated and abs(stop_loss - entry) <= max(1e-9, risk * 0.03):
                        exit_reason = "break_even_stop"
                    else:
                        exit_reason = "stop_loss"
                    break

                for level in partial_levels:
                    if level["taken"] or remaining <= 0:
                        continue
                    level_price = float(level["price"])
                    level_fraction = float(level["fraction"])
                    if level_fraction <= 0 or high < level_price:
                        continue
                    closed = min(level_fraction, remaining)
                    if closed <= 0:
                        continue
                    realized_r += closed * ((level_price - entry) / risk)
                    remaining -= closed
                    level["taken"] = True
                    partial_taken = True

                if high >= take_profit:
                    exit_index = idx
                    exit_price = take_profit
                    realized_r += remaining * ((take_profit - entry) / risk)
                    remaining = 0.0
                    exit_reason = "take_profit"
                    break

                best_r = (high - entry) / risk
                if (
                    exit_plan.break_even_r > 0
                    and not break_even_activated
                    and best_r >= exit_plan.break_even_r
                    and stop_loss < entry
                ):
                    stop_loss = entry
                    break_even_activated = True

                if exit_plan.trailing_enabled and best_r >= exit_plan.trailing_start_r:
                    trailing_activated = True
                    trail_candidate, atr_used, liq_used = self.exit_engine.trailing_stop_candidate(
                        pair=pair,
                        side=signal.side,
                        frame=frame,
                        fill_index=fill_index,
                        index=idx,
                        current_stop=stop_loss,
                        lookback_bars=exit_plan.trailing_lookback_bars,
                        atr_enabled=exit_plan.atr_trailing_enabled,
                        atr_period=exit_plan.atr_trailing_period,
                        atr_multiplier=exit_plan.atr_trailing_multiplier,
                        liquidity_enabled=exit_plan.liquidity_trailing_enabled,
                        liquidity_lookback_bars=exit_plan.liquidity_lookback_bars,
                        liquidity_buffer_pips=exit_plan.liquidity_buffer_pips,
                    )
                    if trail_candidate != stop_loss:
                        stop_loss = trail_candidate
                        atr_trailing_activated = atr_trailing_activated or atr_used
                        liquidity_trailing_activated = liquidity_trailing_activated or liq_used

            else:
                stop_hit = high >= stop_loss
                if stop_hit:
                    exit_index = idx
                    exit_price = stop_loss
                    hit_r = (entry - stop_loss) / risk
                    realized_r += remaining * hit_r
                    remaining = 0.0
                    if trailing_activated and stop_loss < initial_stop_loss:
                        exit_reason = "trailing_stop"
                    elif break_even_activated and abs(stop_loss - entry) <= max(1e-9, risk * 0.03):
                        exit_reason = "break_even_stop"
                    else:
                        exit_reason = "stop_loss"
                    break

                for level in partial_levels:
                    if level["taken"] or remaining <= 0:
                        continue
                    level_price = float(level["price"])
                    level_fraction = float(level["fraction"])
                    if level_fraction <= 0 or low > level_price:
                        continue
                    closed = min(level_fraction, remaining)
                    if closed <= 0:
                        continue
                    realized_r += closed * ((entry - level_price) / risk)
                    remaining -= closed
                    level["taken"] = True
                    partial_taken = True

                if low <= take_profit:
                    exit_index = idx
                    exit_price = take_profit
                    realized_r += remaining * ((entry - take_profit) / risk)
                    remaining = 0.0
                    exit_reason = "take_profit"
                    break

                best_r = (entry - low) / risk
                if (
                    exit_plan.break_even_r > 0
                    and not break_even_activated
                    and best_r >= exit_plan.break_even_r
                    and stop_loss > entry
                ):
                    stop_loss = entry
                    break_even_activated = True

                if exit_plan.trailing_enabled and best_r >= exit_plan.trailing_start_r:
                    trailing_activated = True
                    trail_candidate, atr_used, liq_used = self.exit_engine.trailing_stop_candidate(
                        pair=pair,
                        side=signal.side,
                        frame=frame,
                        fill_index=fill_index,
                        index=idx,
                        current_stop=stop_loss,
                        lookback_bars=exit_plan.trailing_lookback_bars,
                        atr_enabled=exit_plan.atr_trailing_enabled,
                        atr_period=exit_plan.atr_trailing_period,
                        atr_multiplier=exit_plan.atr_trailing_multiplier,
                        liquidity_enabled=exit_plan.liquidity_trailing_enabled,
                        liquidity_lookback_bars=exit_plan.liquidity_lookback_bars,
                        liquidity_buffer_pips=exit_plan.liquidity_buffer_pips,
                    )
                    if trail_candidate != stop_loss:
                        stop_loss = trail_candidate
                        atr_trailing_activated = atr_trailing_activated or atr_used
                        liquidity_trailing_activated = liquidity_trailing_activated or liq_used

            if exit_reason == "timeout":
                exit_price = close

        if exit_reason == "timeout":
            exit_price = float(frame.iloc[exit_index]["close"])
            if remaining > 0:
                if signal.side == "BUY":
                    realized_r += remaining * ((exit_price - entry) / risk)
                else:
                    realized_r += remaining * ((entry - exit_price) / risk)
                remaining = 0.0

        raw_r_multiple = realized_r
        adjusted_r_multiple = raw_r_multiple * max(0.0, risk_multiplier)

        spread_cost_r = 0.0
        slippage_cost_r = 0.0
        execution_delay_cost_r = 0.0
        if realistic_enabled:
            pip_size = execution.pip_size(pair)
            spread_price_cost = (spread_pips * pip_size * 0.5) if (entry_mode == "MARKET" or execution.apply_spread_to_limit) else 0.0
            slippage_price_cost = slippage_pips * pip_size
            spread_cost_r = -spread_price_cost / risk
            slippage_cost_r = -slippage_price_cost / risk
            if fill_delay_bars > 0:
                if signal.side == "BUY":
                    execution_delay_cost_r = -((raw_entry - signal.entry) / risk)
                else:
                    execution_delay_cost_r = -((signal.entry - raw_entry) / risk)

        raw_breakdown = signal.meta.get("score_breakdown") if isinstance(signal.meta, dict) else None
        if isinstance(raw_breakdown, dict):
            feature_breakdown = {str(key): int(value) for key, value in raw_breakdown.items()}
        else:
            feature_breakdown = {
                "htf": int(signal.score_breakdown.htf_alignment),
                "regime": int(signal.score_breakdown.regime_alignment),
                "trigger": int(signal.score_breakdown.trigger_confirmation),
                "liquidity": int(signal.score_breakdown.liquidity_displacement),
                "pd": int(signal.score_breakdown.premium_discount),
                "session": int(signal.score_breakdown.session_timing),
                "news": int(signal.score_breakdown.news_filter),
                "shadow_fvg": int(signal.score_breakdown.fvg_alignment),
                "shadow_ob": int(signal.score_breakdown.order_block_alignment),
                "shadow_mitigation": int(signal.score_breakdown.mitigation_alignment),
                "shadow_smt": int(signal.score_breakdown.smt_alignment),
            }

        research_feature_frame = feature_frame if feature_frame is not None and not feature_frame.empty else frame.iloc[: signal_index + 1]
        smc_features = extract_smc_research_features(
            pair=pair,
            side=signal.side,
            entry=entry,
            frame=research_feature_frame,
            structure_event=signal.structure_event,
            swing_window=self.signal_engine.swing_window,
            settings=self.smc_research_feature_settings,
        )

        exit_time = frame.index[exit_index].to_pydatetime()
        trade = BacktestTrade(
            pair=pair,
            side=signal.side,
            signal_time=signal.generated_at,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_index=fill_index,
            exit_index=exit_index,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            exit_price=round(exit_price, 5),
            exit_reason=exit_reason,
            r_multiple=round(adjusted_r_multiple, 4),
            bars_held=exit_index - fill_index,
            score=signal.score,
            htf_bias=signal.htf_bias,
            regime_label=signal.regime_label,
            regime_direction=signal.regime_direction,
            zone=signal.zone,
            trigger_direction=signal.trigger_direction,
            trigger_event=signal.trigger_event,
            trigger_strength=signal.trigger_strength,
            structure_event=signal.structure_event,
            structure_trend=signal.structure_trend,
            score_htf=signal.score_breakdown.htf_alignment,
            score_regime=signal.score_breakdown.regime_alignment,
            score_trigger=signal.score_breakdown.trigger_confirmation,
            score_liquidity=signal.score_breakdown.liquidity_displacement,
            score_zone=signal.score_breakdown.premium_discount,
            score_news=signal.score_breakdown.news_filter,
            score_session=signal.score_breakdown.session_timing,
            score_fvg=signal.score_breakdown.fvg_alignment,
            score_order_block=signal.score_breakdown.order_block_alignment,
            score_mitigation=signal.score_breakdown.mitigation_alignment,
            score_smt=signal.score_breakdown.smt_alignment,
            shadow_bonus=signal.score_breakdown.shadow_bonus,
            entry_mode=entry_mode,
            entry_source=signal.entry_source,
            fill_delay_bars=fill_delay_bars,
            partial_taken=partial_taken,
            break_even_activated=break_even_activated,
            trailing_activated=trailing_activated,
            feature_breakdown=feature_breakdown,
            raw_r_multiple=round(raw_r_multiple, 4),
            risk_multiplier=round(max(0.0, risk_multiplier), 4),
            sizing_multiplier=round(float(sizing.multiplier), 4),
            meta_probability=round(float(meta.probability), 6),
            meta_accepted=bool(meta.accepted),
            meta_mode=meta.mode,
            meta_size_multiplier=round(float(meta.size_multiplier), 4),
            meta_blocked=bool(meta.enabled and not meta.accepted and meta.mode == "hard_gate"),
            portfolio_sleeve=portfolio.sleeve,
            portfolio_multiplier=round(float(portfolio.multiplier), 4),
            portfolio_applied=bool(portfolio.applied),
            atr_stop_applied=atr_stop_applied,
            atr_value=round(float(atr_value), 6) if atr_value is not None else None,
            realistic_execution=realistic_enabled,
            spread_pips=round(spread_pips, 4),
            spread_cost_r=round(spread_cost_r, 6),
            slippage_pips=round(slippage_pips, 4),
            slippage_cost_r=round(slippage_cost_r, 6),
            execution_delay_bars=fill_delay_bars,
            execution_delay_cost_r=round(execution_delay_cost_r, 6),
            fill_ratio=round(fill_ratio, 4),
            partial_fill=partial_fill,
            exit_engine_mode=exit_plan.mode,
            exit_profile=exit_plan.profile,
            exit_target_rr=round(float(exit_plan.target_rr), 4),
            exit_partial_plan="|".join(
                f"{round(level['price'], 5)}:{round(level['fraction'], 4)}"
                for level in partial_levels
            ),
            atr_trailing_activated=atr_trailing_activated,
            liquidity_trailing_activated=liquidity_trailing_activated,
            smc_features=smc_features,
        )
        return trade, None

    def run_pair_from_frames(
        self,
        pair: str,
        ltf: pd.DataFrame,
        htf: pd.DataFrame,
        trigger: pd.DataFrame,
        *,
        reference_pair: str | None = None,
        reference_trigger: pd.DataFrame | None = None,
        equity_state: EquityProtectionState | None = None,
        portfolio_state: PortfolioLayerState | None = None,
        evaluation_start_time: pd.Timestamp | datetime | None = None,
    ) -> BacktestPairReport:
        if ltf.empty or htf.empty or trigger.empty:
            return BacktestPairReport(
                pair=pair,
                trades=[],
                rejection_counts={},
                evaluations=0,
                bars_processed=0,
                account_settings=self.account_settings,
                error="empty frame",
            )

        ltf = ltf.sort_index()
        htf = htf.sort_index()
        trigger = trigger.sort_index()
        ltf_index = ltf.index
        htf_index = htf.index
        trigger_index = trigger.index
        reference_index = reference_trigger.sort_index().index if reference_trigger is not None and not reference_trigger.empty else None
        if reference_trigger is not None and not reference_trigger.empty:
            reference_trigger = reference_trigger.sort_index()

        trades: list[BacktestTrade] = []
        rejection_counts: defaultdict[str, int] = defaultdict(int)
        regime_evaluations: defaultdict[str, int] = defaultdict(int)
        regime_acceptances: defaultdict[str, int] = defaultdict(int)
        score_observations: list[int] = []
        evaluations = 0
        cursor = self.warmup_bars
        pair_evaluation_step = self._evaluation_step_for_pair(pair)
        if evaluation_start_time is not None:
            start_ts = pd.Timestamp(evaluation_start_time)
            if start_ts.tzinfo is None and trigger_index.tz is not None:
                start_ts = start_ts.tz_localize(trigger_index.tz)
            elif start_ts.tzinfo is not None and trigger_index.tz is not None:
                start_ts = start_ts.tz_convert(trigger_index.tz)
            cursor = max(cursor, int(trigger_index.searchsorted(start_ts, side="left")))
        min_bars = max(120, self.signal_engine.regime_long_window)
        cache_disabled_reason = self._snapshot_cache_disabled_reason()

        while cursor < len(trigger) - self.max_hold_bars - 1:
            trigger_end = cursor + 1
            trigger_time = trigger_index[cursor]
            current_time = trigger_time.to_pydatetime()
            ltf_end = int(ltf_index.searchsorted(trigger_time, side="right"))
            htf_end = int(htf_index.searchsorted(trigger_time, side="right"))
            reference_end = (
                int(reference_index.searchsorted(trigger_time, side="right"))
                if reference_pair is not None and reference_trigger is not None and reference_index is not None
                else 0
            )

            if trigger_end < min_bars or ltf_end < min_bars or htf_end < min_bars:
                cursor += 1
                continue

            news_assessment = self._evaluate_news(self.news_feed, pair, current_time)
            cache_key = self._snapshot_cache_key(
                pair=pair,
                trigger_time=trigger_time,
                trigger_end=trigger_end,
                ltf_end=ltf_end,
                htf_end=htf_end,
                reference_pair=reference_pair,
                reference_end=reference_end,
                news_assessment=news_assessment,
            )
            decision = None if cache_disabled_reason is not None else self.snapshot_cache.get(cache_key)
            if decision is None:
                if cache_disabled_reason is not None:
                    self.snapshot_cache.skip()
                trigger_slice = trigger.iloc[:trigger_end]
                ltf_slice = ltf.iloc[:ltf_end]
                htf_slice = htf.iloc[:htf_end]
                reference_trigger_slice = (
                    reference_trigger.iloc[:reference_end]
                    if reference_pair is not None and reference_trigger is not None
                    else None
                )
                decision = self.signal_engine.evaluate_snapshot(
                    pair,
                    htf_slice,
                    ltf_slice,
                    trigger_frame=trigger_slice,
                    reference_pair=reference_pair,
                    reference_trigger_frame=reference_trigger_slice,
                    news_assessment=news_assessment,
                    emit_logs=False,
                )
                if cache_disabled_reason is None:
                    self.snapshot_cache.put(cache_key, decision)
            evaluations += 1
            if decision.regime_label:
                regime_evaluations[decision.regime_label.upper()] += 1
            if decision.score_value is not None:
                score_observations.append(int(decision.score_value))
            elif decision.score_breakdown is not None:
                score_observations.append(int(decision.score_breakdown.total))

            if not decision.accepted or decision.signal is None:
                rejection_counts[decision.rejection_stage or "unknown"] += 1
                cursor += pair_evaluation_step
                continue
            if decision.regime_label:
                regime_acceptances[decision.regime_label.upper()] += 1

            if equity_state is not None and not equity_state.allow_new_trade():
                rejection_counts["equity_halt"] += 1
                cursor += pair_evaluation_step
                continue

            release_allowed, release_drop = self.signal_engine.gate_signal_release(decision.signal, commit=True)
            if not release_allowed:
                rejection_counts[(release_drop.stage if release_drop is not None else "release_gate")] += 1
                cursor += pair_evaluation_step
                continue

            base_risk_multiplier = equity_state.current_risk_multiplier() if equity_state is not None else 1.0
            trigger_slice = trigger.iloc[:trigger_end]
            volatility_ratio = self._volatility_ratio(trigger_slice, len(trigger_slice) - 1)
            sizing_decision = self.sizing_engine.decide(
                signal=decision.signal,
                volatility_ratio=volatility_ratio,
                equity_state=equity_state,
            )
            spread_preview = self.execution_settings.spread_pips(pair) if self.execution_settings.enabled else 0.0
            meta_decision = self.meta_label_engine.evaluate(
                signal=decision.signal,
                spread_pips=spread_preview,
            )
            if meta_decision.enabled and meta_decision.mode == "hard_gate" and not meta_decision.accepted:
                rejection_counts["meta_label"] += 1
                cursor += pair_evaluation_step
                continue

            portfolio_decision = (
                portfolio_state.decide(decision.signal)
                if portfolio_state is not None
                else PortfolioDecision(sleeve="unassigned", multiplier=1.0, applied=False, reason="disabled")
            )
            portfolio_multiplier = portfolio_decision.multiplier if portfolio_decision.applied else 1.0
            risk_multiplier = (
                max(0.0, base_risk_multiplier)
                * max(0.0, sizing_decision.multiplier)
                * max(0.0, meta_decision.size_multiplier)
                * max(0.0, portfolio_multiplier)
            )
            trade, entry_rejection = self._simulate_trade(
                pair,
                decision.signal,
                trigger,
                cursor,
                risk_multiplier=risk_multiplier,
                sizing_decision=sizing_decision,
                meta_decision=meta_decision,
                portfolio_decision=portfolio_decision,
                feature_frame=ltf.iloc[:ltf_end],
            )
            if trade is None:
                rejection_counts[entry_rejection or "entry_not_filled"] += 1
                cursor += pair_evaluation_step
                continue

            trades.append(trade)
            if equity_state is not None:
                equity_state.register_trade(trade.r_multiple)
            if portfolio_state is not None:
                portfolio_state.register(trade.portfolio_sleeve, trade.r_multiple)
            cursor = trade.exit_index + 1

        return BacktestPairReport(
            pair=pair,
            trades=trades,
            rejection_counts=dict(rejection_counts),
            evaluations=evaluations,
            bars_processed=len(trigger),
            account_settings=self.account_settings,
            regime_evaluations=dict(regime_evaluations),
            regime_acceptances=dict(regime_acceptances),
            score_observations=score_observations,
        )

    def run_pair(
        self,
        pair: str,
        *,
        equity_state: EquityProtectionState | None = None,
        portfolio_state: PortfolioLayerState | None = None,
    ) -> BacktestPairReport:
        try:
            ltf, htf, trigger = self.load_pair_frames(pair)
        except Exception as exc:
            return BacktestPairReport(
                pair=pair,
                trades=[],
                rejection_counts={},
                evaluations=0,
                bars_processed=0,
                account_settings=self.account_settings,
                error=str(exc),
            )

        reference_pair = self.signal_engine._resolve_smt_reference_pair(pair, None)
        reference_trigger: pd.DataFrame | None = None
        if self.signal_engine.enable_smt_confirmation and reference_pair is not None:
            try:
                reference_trigger = self.market_data.fetch_ohlcv(
                    reference_pair,
                    self.signal_engine.trigger_timeframe,
                    limit=self.history_limit,
                    end_time=self.end_time,
                )
            except Exception:
                reference_trigger = None

        return self.run_pair_from_frames(
            pair,
            ltf,
            htf,
            trigger,
            reference_pair=reference_pair,
            reference_trigger=reference_trigger,
            equity_state=equity_state,
            portfolio_state=portfolio_state,
        )

    def run(self, pairs: Iterable[str]) -> BacktestRunResult:
        started_at = datetime.now(timezone.utc)
        if hasattr(self.signal_engine, "reset_release_state"):
            self.signal_engine.reset_release_state()
        self._rng = build_rng(self.execution_settings.random_seed)
        equity_state = EquityProtectionState(self.equity_protection_settings) if self.equity_protection_settings.enabled else None
        portfolio_state = PortfolioLayerState(self.portfolio_layer_settings)
        pair_list = list(pairs)
        reports = [self.run_pair(pair, equity_state=equity_state, portfolio_state=portfolio_state) for pair in pair_list]
        finished_at = datetime.now(timezone.utc)
        parameters = {
            "ltf_timeframe": self.signal_engine.ltf_timeframe,
            "htf_timeframe": self.signal_engine.htf_timeframe,
            "trigger_timeframe": self.signal_engine.trigger_timeframe,
            "history_limit": self.history_limit,
            "backtest_end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "market_data_cache_enabled": self.market_data.cache_config.enabled,
            "market_data_cache_dir": str(self.market_data.cache_config.cache_dir),
            "market_data_cache_ttl_hours": self.market_data.cache_config.ttl_hours,
            "market_data_cache_mode": self.market_data.cache_config.mode,
            "snapshot_cache_enabled": self.snapshot_cache.enabled,
            "snapshot_cache_disabled_reason": self._snapshot_cache_disabled_reason(),
            "snapshot_cache_stats": self.snapshot_cache.stats(),
            "max_hold_bars": self.max_hold_bars,
            "warmup_bars": self.warmup_bars,
            "evaluation_step": self.evaluation_step,
            "pair_evaluation_steps": {
                str(pair): self._evaluation_step_for_pair(str(pair))
                for pair in pair_list
            },
            "min_score": self.signal_engine.min_score,
            "risk_reward": self.signal_engine.risk_reward,
            "swing_window": self.signal_engine.swing_window,
            "regime_short_window": self.signal_engine.regime_short_window,
            "regime_long_window": self.signal_engine.regime_long_window,
            "regime_opposition_confidence": self.signal_engine.regime_opposition_confidence,
            "contraction_min_trigger_strength": self.signal_engine.contraction_min_trigger_strength,
            "range_min_trigger_strength": self.signal_engine.range_min_trigger_strength,
            "require_displacement_in_contraction": self.signal_engine.require_displacement_in_contraction,
            "session_min_score": self.signal_engine.session_min_score,
            "enable_smt_confirmation": self.signal_engine.enable_smt_confirmation,
            "smt_hard_gate": self.signal_engine.smt_hard_gate,
            "smt_min_strength": self.signal_engine.smt_min_strength,
            "smt_opposite_block_strength": self.signal_engine.smt_opposite_block_strength,
            "partial_tp_enabled": self.signal_engine.partial_tp_enabled,
            "partial_tp_r": self.signal_engine.partial_tp_r,
            "partial_tp_fraction": self.signal_engine.partial_tp_fraction,
            "break_even_r": self.signal_engine.break_even_r,
            "trailing_enabled": self.signal_engine.trailing_enabled,
            "trailing_start_r": self.signal_engine.trailing_start_r,
            "trailing_lookback_bars": self.signal_engine.trailing_lookback_bars,
            "time_stop_bars": self.signal_engine.time_stop_bars,
            "pair_correlation_threshold": self.signal_engine.correlation_cap.threshold,
            "correlation_lookback": self.signal_engine.correlation_cap.lookback,
            "currency_exposure_cap": self.signal_engine.currency_exposure_cap,
            "portfolio_currency_gross_cap": self.signal_engine.portfolio_currency_gross_cap,
            "portfolio_currency_net_cap": self.signal_engine.portfolio_currency_net_cap,
            "portfolio_exposure_window_minutes": self.signal_engine.portfolio_exposure_window_minutes,
            "pair_cooldown_minutes": self.signal_engine.pair_cooldown_minutes,
            "max_entries_per_bias": self.signal_engine.max_entries_per_bias,
            "bias_window_minutes": self.signal_engine.bias_window_minutes,
            "enable_shadow_scoring": self.signal_engine.enable_shadow_scoring,
            "enable_mitigation_entry": self.signal_engine.enable_mitigation_entry,
            "enable_adaptive_weights": self.signal_engine.enable_adaptive_weights,
            "adaptive_regime_weights": self.signal_engine._adaptive_weight_settings.regime_weights,
            "enable_score_normalization": self.signal_engine._score_normalizer.settings.enabled,
            "score_normalization_method": self.signal_engine._score_normalizer.settings.method,
            "score_normalization_window": self.signal_engine._score_normalizer.settings.window,
            "score_normalization_scale_factor": self.signal_engine._score_normalizer.settings.scale_factor,
            "score_normalization_backtest_only": self.signal_engine._score_normalizer.settings.backtest_only,
            "allow_live_score_normalization": self.signal_engine._score_normalizer.settings.allow_live,
            "enable_dynamic_threshold": self.signal_engine._dynamic_threshold_tracker.settings.enabled,
            "threshold_percentile": self.signal_engine._dynamic_threshold_tracker.settings.percentile,
            "threshold_rolling_window": self.signal_engine._dynamic_threshold_tracker.settings.rolling_window,
            "apply_dynamic_threshold": self.signal_engine._dynamic_threshold_tracker.settings.apply_threshold,
            "dynamic_threshold_backtest_only": self.signal_engine._dynamic_threshold_tracker.settings.backtest_only,
            "allow_live_dynamic_threshold": self.signal_engine._dynamic_threshold_tracker.settings.allow_live,
            "enable_realistic_execution": self.execution_settings.enabled,
            "spread_default_pips": self.execution_settings.spread_default_pips,
            "spread_by_pair": self.execution_settings.spread_by_pair,
            "slippage_mode": self.execution_settings.slippage_mode,
            "max_slippage_pips": self.execution_settings.max_slippage_pips,
            "execution_delay_bars": self.execution_settings.execution_delay_bars,
            "partial_fill_probability": self.execution_settings.partial_fill_probability,
            "partial_fill_min_ratio": self.execution_settings.partial_fill_min_ratio,
            "limit_touch_tolerance_pips": self.execution_settings.limit_touch_tolerance_pips,
            "apply_spread_to_limit": self.execution_settings.apply_spread_to_limit,
            "random_seed": self.execution_settings.random_seed,
            "enable_atr_risk": self.atr_risk_settings.enabled,
            "atr_period": self.atr_risk_settings.period,
            "atr_multiplier": self.atr_risk_settings.multiplier,
            "enable_equity_protection": self.equity_protection_settings.enabled,
            "max_drawdown_limit": self.equity_protection_settings.max_drawdown_limit,
            "drawdown_risk_reduction_factor": self.equity_protection_settings.drawdown_risk_reduction_factor,
            "max_consecutive_losses": self.equity_protection_settings.max_consecutive_losses,
            "min_risk_multiplier": self.equity_protection_settings.min_risk_multiplier,
            "enable_exit_engine": self.exit_settings.enabled,
            "exit_profile_preset": self.exit_settings.profile_preset,
            "exit_use_regime_profiles": self.exit_settings.use_regime_profiles,
            "exit_profile_overrides": self.exit_settings.profile_overrides,
            "exit_atr_trailing_enabled": self.exit_settings.atr_trailing_enabled,
            "exit_atr_trailing_period": self.exit_settings.atr_trailing_period,
            "exit_atr_trailing_multiplier": self.exit_settings.atr_trailing_multiplier,
            "exit_liquidity_trailing_enabled": self.exit_settings.liquidity_trailing_enabled,
            "exit_liquidity_lookback_bars": self.exit_settings.liquidity_lookback_bars,
            "exit_liquidity_buffer_pips": self.exit_settings.liquidity_buffer_pips,
            "exit_volatility_rr_enabled": self.exit_settings.volatility_rr_enabled,
            "exit_volatility_rr_floor": self.exit_settings.volatility_rr_floor,
            "exit_volatility_rr_cap": self.exit_settings.volatility_rr_cap,
            "enable_adaptive_sizing": self.sizing_settings.enabled,
            "sizing_min_multiplier": self.sizing_settings.min_multiplier,
            "sizing_max_multiplier": self.sizing_settings.max_multiplier,
            "sizing_confidence_floor_score": self.sizing_settings.confidence_floor_score,
            "sizing_confidence_ceiling_score": self.sizing_settings.confidence_ceiling_score,
            "sizing_regime_multipliers": self.sizing_settings.regime_multipliers,
            "enable_meta_label": self.meta_label_settings.enabled,
            "meta_label_mode": self.meta_label_settings.mode,
            "meta_label_probability_threshold": self.meta_label_settings.probability_threshold,
            "meta_label_enable_size_adjustment": self.meta_label_settings.enable_size_adjustment,
            "meta_label_low_probability_multiplier": self.meta_label_settings.low_probability_multiplier,
            "meta_label_high_probability_multiplier": self.meta_label_settings.high_probability_multiplier,
            "meta_label_high_probability_threshold": self.meta_label_settings.high_probability_threshold,
            "enable_portfolio_layer": self.portfolio_layer_settings.enabled,
            "portfolio_layer_mode": self.portfolio_layer_settings.mode,
            "portfolio_layer_min_multiplier": self.portfolio_layer_settings.min_multiplier,
            "portfolio_layer_max_multiplier": self.portfolio_layer_settings.max_multiplier,
            "portfolio_layer_learning_window": self.portfolio_layer_settings.learning_window,
            "portfolio_layer_min_trades_per_sleeve": self.portfolio_layer_settings.min_trades_per_sleeve,
            "portfolio_layer_max_sleeve_concentration": self.portfolio_layer_settings.max_sleeve_concentration,
            "enable_smc_research_features": self.smc_research_feature_settings.enabled,
            "smc_research_feature_settings": asdict(self.smc_research_feature_settings),
            "backtest_account_enabled": self.account_settings.enabled,
            "backtest_starting_balance": self.account_settings.starting_balance,
            "backtest_risk_per_trade": self.account_settings.risk_per_trade,
            "backtest_account_currency": self.account_settings.currency,
        }
        return BacktestRunResult(
            pair_reports=reports,
            parameters=parameters,
            started_at=started_at,
            finished_at=finished_at,
            news_mode=self.news_feed.__class__.__name__,
            account_settings=self.account_settings,
        )
