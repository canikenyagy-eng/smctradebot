from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import Settings
from data.market_data import MarketDataCacheConfig, MarketDataClient
from services.forward_outcomes import ForwardOutcomeSettings, ForwardOutcomeTracker
from main import _itick_config_from_settings, _live_bar_config_from_settings, _redundant_config_from_settings


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track outcomes for forward-test live signal candidates.")
    parser.add_argument("--journal", default=None, help="Forward journal JSONL path")
    parser.add_argument("--output", default=None, help="Outcome JSONL path")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON path")
    parser.add_argument("--timeframe", default=None, help="Outcome OHLCV timeframe, default from env or M15")
    parser.add_argument("--history-limit", type=int, default=None, help="Bars to fetch per symbol")
    parser.add_argument("--data-source", default=None, help="Market data source override, e.g. yahoo, itick, mt5")
    parser.add_argument("--cache-dir", default=None, help="OHLCV cache directory override")
    parser.add_argument("--cache-only", action="store_true", help="Use cache only; do not fetch provider data")
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh OHLCV cache before tracking")
    parser.add_argument("--sent-only", action="store_true", help="Track only candidates with successful Telegram delivery")
    parser.add_argument("--include-unsent", action="store_true", help="Track all candidates even if env sent-only is enabled")
    parser.add_argument("--max-hold-bars", type=int, default=None, help="Fallback max hold bars when candidate has no time stop")
    parser.add_argument("--entry-expiry-bars", type=int, default=None, help="Bars before unfilled limit entries expire; 0 uses candidate time stop")
    parser.add_argument(
        "--ambiguous-policy",
        choices=("ambiguous", "stop_first", "target_first"),
        default=None,
        help="How to mark candles that hit TP and SL together",
    )
    parser.add_argument("--write-all", action="store_true", help="Do not skip candidates that already have terminal outcomes")
    parser.add_argument("--no-write", action="store_true", help="Analyze only; do not append outcome JSONL rows")
    return parser


def cache_mode(args: argparse.Namespace) -> str:
    if args.cache_only:
        return "cache_only"
    if args.refresh_cache:
        return "refresh"
    return "read_through"


def build_market_data(settings: Settings, args: argparse.Namespace, history_limit: int) -> MarketDataClient:
    source = (args.data_source or settings.data_source).strip().lower()
    cache_enabled = settings.market_data_cache_enabled
    selected_cache_mode = cache_mode(args)
    if source in {"live_bars", "redundant"}:
        cache_enabled = False
        selected_cache_mode = "disabled"
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
            cache_dir=args.cache_dir or settings.market_data_cache_dir,
            ttl_hours=settings.market_data_cache_ttl_hours,
            mode=selected_cache_mode,
        ),
    )


def build_tracker_settings(settings: Settings, args: argparse.Namespace) -> ForwardOutcomeSettings:
    sent_only = settings.forward_outcome_sent_only
    if args.sent_only:
        sent_only = True
    if args.include_unsent:
        sent_only = False

    return ForwardOutcomeSettings(
        journal_path=args.journal or settings.forward_journal_log_path,
        output_path=args.output or settings.forward_outcome_log_path,
        timeframe=args.timeframe or settings.forward_outcome_timeframe,
        history_limit=args.history_limit or settings.forward_outcome_history_limit,
        sent_only=sent_only,
        max_hold_bars=args.max_hold_bars or settings.forward_outcome_max_hold_bars,
        entry_expiry_bars=(
            args.entry_expiry_bars
            if args.entry_expiry_bars is not None
            else settings.forward_outcome_entry_expiry_bars
        ),
        ambiguous_policy=args.ambiguous_policy or settings.forward_outcome_ambiguous_policy,
        skip_terminal_existing=not args.write_all,
    )


def print_summary(summary: dict[str, object], *, written: int) -> None:
    print()
    print("FORWARD OUTCOME SUMMARY")
    print(f"Candidates: {summary['candidates']} | Closed: {summary['closed']} | Written: {written}")
    print(
        "Win rate: {wr:.1%} | AvgR: {avg:.3f} | PF: {pf}".format(
            wr=float(summary.get("win_rate", 0.0)),
            avg=float(summary.get("avg_r", 0.0)),
            pf=summary.get("profit_factor", 0.0),
        )
    )
    print(f"Status: {summary.get('status_counts', {})}")
    print(f"Reasons: {summary.get('exit_reason_counts', {})}")
    by_symbol = summary.get("by_symbol", {})
    if isinstance(by_symbol, dict) and by_symbol:
        print()
        print("BY SYMBOL")
        for symbol, row in sorted(by_symbol.items()):
            if not isinstance(row, dict):
                continue
            print(
                "{symbol:<8} candidates={candidates:<4} closed={closed:<4} wr={wr:.1%} avg_r={avg:.3f} pf={pf}".format(
                    symbol=symbol,
                    candidates=int(row.get("candidates", 0)),
                    closed=int(row.get("closed", 0)),
                    wr=float(row.get("win_rate", 0.0)),
                    avg=float(row.get("avg_r", 0.0)),
                    pf=row.get("profit_factor", 0.0),
                )
            )


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    tracker_settings = build_tracker_settings(settings, args)
    tracker = ForwardOutcomeTracker(tracker_settings)
    market_data = build_market_data(settings, args, tracker.settings.history_limit)

    try:
        outcomes = tracker.run(market_data)
        written = 0 if args.no_write else tracker.append_outcomes(outcomes)
        summary = tracker.summarize(outcomes)
        summary_output = args.summary_output or settings.forward_outcome_summary_path
        if summary_output:
            summary_path = Path(summary_output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print_summary(summary, written=written)
        if summary_output:
            print(f"Summary saved: {summary_output}")
        if not args.no_write:
            print(f"Outcomes appended: {tracker.settings.output_path}")
    finally:
        market_data.close()


if __name__ == "__main__":
    main()
