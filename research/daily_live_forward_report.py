from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import Settings
from data.market_data import MarketDataCacheConfig, MarketDataClient
from main import _itick_config_from_settings, _live_bar_config_from_settings, _redundant_config_from_settings
from services.daily_forward_report import (
    DailyForwardReportBuilder,
    DailyForwardReportSettings,
    print_daily_forward_report,
)
from services.forward_outcomes import ForwardOutcomeSettings, ForwardOutcomeTracker


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the daily live forward validation + feed-quality report.")
    parser.add_argument("--recent-minutes", type=int, default=None, help="Report window, default from env or 1440")
    parser.add_argument("--output", default=None, help="Daily report JSON path")
    parser.add_argument("--skip-outcome-update", action="store_true", help="Do not refresh theoretical outcomes first")
    parser.add_argument("--no-write-outcomes", action="store_true", help="Evaluate outcomes but do not append JSONL rows")
    parser.add_argument("--data-source", default=None, help="Outcome market data source override")
    parser.add_argument("--timeframe", default=None, help="Outcome timeframe, default from env")
    parser.add_argument("--history-limit", type=int, default=None, help="Outcome bars to load per symbol")
    parser.add_argument("--sent-only", action="store_true", help="Include only Telegram-delivered candidates")
    parser.add_argument("--include-unsent", action="store_true", help="Include all candidates even if env sent-only is enabled")
    parser.add_argument("--min-closed-trades", type=int, default=None, help="Minimum closed outcomes before acting on stats")
    parser.add_argument("--include-rows", action="store_true", help="Keep per-candidate rows in exported JSON")
    return parser


def build_market_data(settings: Settings, *, data_source: str | None, history_limit: int) -> MarketDataClient:
    source = (data_source or settings.data_source).strip().lower()
    cache_enabled = settings.market_data_cache_enabled
    cache_mode = settings.market_data_cache_mode
    if source in {"live_bars", "redundant"}:
        cache_enabled = False
        cache_mode = "disabled"

    return MarketDataClient(
        history_limit=max(settings.history_limit, history_limit),
        data_source=source,
        mt5_login=settings.mt5_login,
        mt5_password=settings.mt5_password,
        mt5_server=settings.mt5_server,
        mt5_path=settings.mt5_path,
        itick_config=_itick_config_from_settings(settings),
        live_bar_config=_live_bar_config_from_settings(settings),
        redundant_config=_redundant_config_from_settings(settings),
        cache_config=MarketDataCacheConfig(
            enabled=cache_enabled,
            cache_dir=settings.market_data_cache_dir,
            ttl_hours=settings.market_data_cache_ttl_hours,
            mode=cache_mode,
        ),
    )


def outcome_sent_only(settings: Settings, args: argparse.Namespace) -> bool:
    sent_only = settings.forward_outcome_sent_only
    if settings.daily_forward_report_sent_only:
        sent_only = True
    if args.sent_only:
        sent_only = True
    if args.include_unsent:
        sent_only = False
    return sent_only


def update_outcomes(settings: Settings, args: argparse.Namespace) -> dict[str, object]:
    tracker_settings = ForwardOutcomeSettings(
        journal_path=settings.forward_journal_log_path,
        output_path=settings.forward_outcome_log_path,
        timeframe=args.timeframe or settings.forward_outcome_timeframe,
        history_limit=args.history_limit or settings.forward_outcome_history_limit,
        sent_only=outcome_sent_only(settings, args),
        max_hold_bars=settings.forward_outcome_max_hold_bars,
        entry_expiry_bars=settings.forward_outcome_entry_expiry_bars,
        ambiguous_policy=settings.forward_outcome_ambiguous_policy,
        skip_terminal_existing=True,
    )
    tracker = ForwardOutcomeTracker(tracker_settings)
    market_data = build_market_data(
        settings,
        data_source=args.data_source,
        history_limit=tracker.settings.history_limit,
    )
    try:
        outcomes = tracker.run(market_data)
        written = 0 if args.no_write_outcomes else tracker.append_outcomes(outcomes)
        summary = tracker.summarize(outcomes)
    finally:
        market_data.close()

    summary_path = Path(settings.forward_outcome_summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {
        "enabled": True,
        "written": written,
        "summary_path": str(summary_path),
        "outcome_path": str(tracker.settings.output_path),
        "summary": summary,
    }


def build_report_settings(settings: Settings, args: argparse.Namespace) -> DailyForwardReportSettings:
    sent_only = settings.daily_forward_report_sent_only
    if args.sent_only:
        sent_only = True
    if args.include_unsent:
        sent_only = False
    return DailyForwardReportSettings(
        report_path=args.output or settings.daily_forward_report_path,
        recent_minutes=args.recent_minutes or settings.daily_forward_report_recent_minutes,
        sent_only=sent_only,
        min_closed_trades=(
            args.min_closed_trades
            if args.min_closed_trades is not None
            else settings.daily_forward_report_min_closed_trades
        ),
        include_rows=args.include_rows,
    )


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    outcome_update = {"enabled": False, "reason": "skipped"}
    if not args.skip_outcome_update:
        outcome_update = update_outcomes(settings, args)

    builder = DailyForwardReportBuilder(settings, build_report_settings(settings, args))
    report = builder.build_report(outcome_update=outcome_update)
    builder.write_report(report)
    print_daily_forward_report(report)
    print(f"Report saved: {builder.settings.report_path}")


if __name__ == "__main__":
    main()
