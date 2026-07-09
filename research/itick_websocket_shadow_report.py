from __future__ import annotations

import argparse
import logging
from typing import Mapping

from config import Settings
from services.itick_websocket_shadow import ItickWebSocketShadowReporter


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize iTick WebSocket shadow quote freshness and latency.")
    parser.add_argument("--log", default=None, help="iTick WebSocket shadow JSONL path")
    parser.add_argument("--output", default=None, help="Summary JSON path")
    parser.add_argument("--recent-minutes", type=int, default=1440, help="Recent window to analyze")
    parser.add_argument("--fail-on-alert", action="store_true", help="Exit 2 if stale/slow/no quotes are present")
    return parser


def _fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_group(title: str, rows: Mapping[str, object]) -> None:
    if not rows:
        return
    print()
    print(title)
    for key, value in rows.items():
        if not isinstance(value, dict):
            continue
        print(
            "{key:<12} quotes={quotes:<5} stale={stale:<4} slow={slow:<4} avg_lat={avg}s p95={p95}s max={max_lat}s alert={alert}".format(
                key=key[:12],
                quotes=int(value.get("quotes", 0)),
                stale=int(value.get("stale", 0)),
                slow=int(value.get("slow", 0)),
                avg=_fmt(value.get("avg_latency_seconds"), 3),
                p95=_fmt(value.get("p95_latency_seconds"), 3),
                max_lat=_fmt(value.get("max_latency_seconds"), 3),
                alert=value.get("alert", False),
            )
        )


def print_report(report: Mapping[str, object]) -> None:
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    print()
    print("ITICK WEBSOCKET SHADOW")
    print(
        "Quotes: {quotes} | Stale: {stale} ({stale_rate:.3%}) | Slow: {slow} ({slow_rate:.3%}) | Errors: {errors} | LatestAge: {latest_age}s | Alert: {alert}".format(
            quotes=int(overall.get("quotes", 0)),
            stale=int(overall.get("stale", 0)),
            stale_rate=float(overall.get("stale_rate", 0.0)),
            slow=int(overall.get("slow", 0)),
            slow_rate=float(overall.get("slow_rate", 0.0)),
            errors=int(overall.get("connection_errors", 0)),
            latest_age=_fmt(overall.get("latest_quote_age_seconds"), 3),
            alert=overall.get("alert", False),
        )
    )
    print(
        "Latency avg={avg}s p95={p95}s max={max_lat}s | Events: {events}".format(
            avg=_fmt(overall.get("avg_latency_seconds"), 3),
            p95=_fmt(overall.get("p95_latency_seconds"), 3),
            max_lat=_fmt(overall.get("max_latency_seconds"), 3),
            events=report.get("events", {}),
        )
    )
    by_pair = report.get("by_pair", {})
    if isinstance(by_pair, dict):
        print_group("BY PAIR", by_pair)
    latest = report.get("latest_by_pair", {})
    if isinstance(latest, dict) and latest:
        print()
        print("LATEST")
        for pair, row in latest.items():
            if isinstance(row, dict):
                print(f"{pair:<12} price={row.get('last_price')} latency={row.get('latency_seconds')} provider_time={row.get('provider_time')}")


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    reporter = ItickWebSocketShadowReporter(
        log_path=args.log or settings.itick_websocket_log_path,
        summary_path=args.output or settings.itick_websocket_summary_path,
        recent_minutes=args.recent_minutes,
        stale_seconds=settings.itick_websocket_stale_seconds,
        max_latency_seconds=settings.itick_websocket_max_latency_seconds,
        max_stale_rate=settings.itick_websocket_max_stale_rate,
        max_slow_rate=settings.itick_websocket_max_slow_rate,
        max_connection_errors=settings.itick_websocket_max_connection_errors,
        max_latest_quote_age_seconds=settings.itick_websocket_max_latest_quote_age_seconds,
    )
    report = reporter.build_report()
    reporter.write_report(report)
    print_report(report)
    print(f"Summary saved: {reporter.summary_path}")
    overall = report.get("overall", {})
    if args.fail_on_alert and isinstance(overall, dict) and overall.get("alert"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
