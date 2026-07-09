from __future__ import annotations

import argparse
import logging
from typing import Mapping

from config import Settings
from services.market_data_diagnostics import MarketDataDiagnosticsReporter, MarketDataDiagnosticsReportSettings


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize market data freshness and provider latency diagnostics.")
    parser.add_argument("--log", default=None, help="Market data diagnostics JSONL path")
    parser.add_argument("--output", default=None, help="Summary JSON path")
    parser.add_argument("--recent-minutes", type=int, default=1440, help="Recent window to analyze")
    parser.add_argument("--max-latency-seconds", type=float, default=None, help="Slow fetch threshold")
    parser.add_argument("--max-candle-age-seconds", type=int, default=None, help="Stale candle threshold")
    parser.add_argument("--fail-on-alert", action="store_true", help="Exit 2 if errors/stale/slow rows are present")
    return parser


def build_settings(settings: Settings, args: argparse.Namespace) -> MarketDataDiagnosticsReportSettings:
    return MarketDataDiagnosticsReportSettings(
        log_path=args.log or settings.market_data_diagnostics_log_path,
        summary_path=args.output or settings.market_data_diagnostics_summary_path,
        recent_minutes=args.recent_minutes,
        max_latency_seconds=args.max_latency_seconds or settings.market_data_diagnostics_max_latency_seconds,
        max_candle_age_seconds=args.max_candle_age_seconds or settings.market_data_diagnostics_max_candle_age_seconds,
    )


def _fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_group(title: str, data: Mapping[str, object], *, limit: int = 12) -> None:
    if not data:
        return
    print()
    print(title)
    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            rows.append((key, value))
    rows.sort(key=lambda item: (int(item[1].get("errors", 0)) + int(item[1].get("stale", 0)) + int(item[1].get("slow", 0)), int(item[1].get("fetches", 0))), reverse=True)
    for key, row in rows[:limit]:
        print(
            "{key:<18} fetches={fetches:<4} err={errors:<3} stale={stale:<3} slow={slow:<3} avg_lat={avg}s p95={p95}s max_age={age}s".format(
                key=key[:18],
                fetches=int(row.get("fetches", 0)),
                errors=int(row.get("errors", 0)),
                stale=int(row.get("stale", 0)),
                slow=int(row.get("slow", 0)),
                avg=_fmt(row.get("avg_latency_seconds"), 3),
                p95=_fmt(row.get("p95_latency_seconds"), 3),
                age=_fmt(row.get("max_candle_age_seconds"), 0),
            )
        )


def print_report(report: Mapping[str, object]) -> None:
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    print()
    print("MARKET DATA DIAGNOSTICS")
    print(
        "Fetches: {fetches} | OK: {ok} | Errors: {errors} | Stale: {stale} | Slow: {slow} | Alert: {alert}".format(
            fetches=int(overall.get("fetches", 0)),
            ok=int(overall.get("ok", 0)),
            errors=int(overall.get("errors", 0)),
            stale=int(overall.get("stale", 0)),
            slow=int(overall.get("slow", 0)),
            alert=overall.get("alert", False),
        )
    )
    print(
        "Latency avg={avg}s p95={p95}s max={max_lat}s | Candle age avg={avg_age}s max={max_age}s".format(
            avg=_fmt(overall.get("avg_latency_seconds"), 3),
            p95=_fmt(overall.get("p95_latency_seconds"), 3),
            max_lat=_fmt(overall.get("max_latency_seconds"), 3),
            avg_age=_fmt(overall.get("avg_candle_age_seconds"), 0),
            max_age=_fmt(overall.get("max_candle_age_seconds"), 0),
        )
    )
    print(f"Served from: {overall.get('served_from', {})}")
    for title, key in (
        ("BY SOURCE", "by_source"),
        ("BY TIMEFRAME", "by_timeframe"),
        ("BY PAIR", "by_pair"),
        ("BY PAIR/TIMEFRAME", "by_pair_timeframe"),
    ):
        data = report.get(key, {})
        if isinstance(data, dict):
            print_group(title, data)


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    reporter = MarketDataDiagnosticsReporter(build_settings(settings, args))
    report = reporter.build_report()
    reporter.write_report(report)
    print_report(report)
    print(f"Summary saved: {reporter.settings.summary_path}")
    overall = report.get("overall", {})
    if args.fail_on_alert and isinstance(overall, dict) and overall.get("alert"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
