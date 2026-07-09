from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping

from config import Settings
from services.live_bar_builder import LiveBarBuilderReporter


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize iTick live-bar builder freshness and generated bars.")
    parser.add_argument("--log", default=None, help="Live-bar builder JSONL path")
    parser.add_argument("--output", default=None, help="Summary JSON path")
    parser.add_argument("--recent-minutes", type=int, default=1440, help="Recent window to analyze")
    parser.add_argument("--max-bar-age-seconds", type=float, default=30.0, help="Bar update age alert threshold")
    parser.add_argument("--fail-on-alert", action="store_true", help="Exit 2 if stale/no bar updates are present")
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
            "{key:<15} updates={updates:<5} closed={closed:<4} stale={stale:<4} avg_age={avg}s max_age={max_age}s avg_quotes={quotes} alert={alert}".format(
                key=key[:15],
                updates=int(value.get("updates", 0)),
                closed=int(value.get("closed", 0)),
                stale=int(value.get("stale_updates", 0)),
                avg=_fmt(value.get("avg_bar_age_seconds"), 3),
                max_age=_fmt(value.get("max_bar_age_seconds"), 3),
                quotes=_fmt(value.get("avg_quote_count"), 1),
                alert=value.get("alert", False),
            )
        )


def print_report(report: Mapping[str, object]) -> None:
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    print()
    print("LIVE BAR BUILDER")
    print(
        "Updates: {updates} | Closed: {closed} | Stale: {stale} | Alert: {alert}".format(
            updates=int(overall.get("updates", 0)),
            closed=int(overall.get("closed", 0)),
            stale=int(overall.get("stale_updates", 0)),
            alert=overall.get("alert", False),
        )
    )
    print(
        "Bar age avg={avg}s max={max_age}s | Avg quotes/bar={quotes}".format(
            avg=_fmt(overall.get("avg_bar_age_seconds"), 3),
            max_age=_fmt(overall.get("max_bar_age_seconds"), 3),
            quotes=_fmt(overall.get("avg_quote_count"), 1),
        )
    )
    by_pair_timeframe = report.get("by_pair_timeframe", {})
    if isinstance(by_pair_timeframe, dict):
        print_group("BY PAIR/TIMEFRAME", by_pair_timeframe)
    latest = report.get("latest", {})
    if isinstance(latest, dict) and latest:
        print()
        print("LATEST")
        for key, row in latest.items():
            if isinstance(row, dict):
                print(
                    f"{key:<15} close={row.get('close')} quotes={row.get('quote_count')} complete={row.get('complete')} start={row.get('timestamp')}"
                )


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    reporter = LiveBarBuilderReporter(
        log_path=args.log or settings.live_bar_builder_log_path,
        recent_minutes=args.recent_minutes,
        max_bar_age_seconds=args.max_bar_age_seconds,
    )
    report = reporter.build_report()
    output = Path(args.output or settings.live_bar_builder_summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print_report(report)
    print(f"Summary saved: {output}")
    overall = report.get("overall", {})
    if args.fail_on_alert and isinstance(overall, dict) and overall.get("alert"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
