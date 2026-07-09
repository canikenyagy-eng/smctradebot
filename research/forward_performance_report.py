from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping

from config import Settings
from services.forward_performance import ForwardPerformanceReporter, ForwardPerformanceSettings


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build forward performance analytics from journal + outcome JSONL files.")
    parser.add_argument("--journal", default=None, help="Forward journal JSONL path")
    parser.add_argument("--outcomes", default=None, help="Forward outcomes JSONL path")
    parser.add_argument("--output", default=None, help="Report JSON path")
    parser.add_argument("--sent-only", action="store_true", help="Include only Telegram-delivered candidates")
    parser.add_argument("--include-unsent", action="store_true", help="Include all candidates even if env sent-only is enabled")
    parser.add_argument("--score-bucket-size", type=int, default=None, help="Score bucket size, e.g. 5 or 10")
    parser.add_argument("--min-closed-trades", type=int, default=None, help="Minimum closed-with-R sample marker per group")
    parser.add_argument("--recent-minutes", type=int, default=None, help="Only include candidates generated in this window")
    parser.add_argument("--no-rows", action="store_true", help="Omit per-candidate rows from exported JSON")
    return parser


def build_settings(settings: Settings, args: argparse.Namespace) -> ForwardPerformanceSettings:
    sent_only = settings.forward_performance_sent_only
    if args.sent_only:
        sent_only = True
    if args.include_unsent:
        sent_only = False

    return ForwardPerformanceSettings(
        journal_path=args.journal or settings.forward_journal_log_path,
        outcome_path=args.outcomes or settings.forward_outcome_log_path,
        report_path=args.output or settings.forward_performance_report_path,
        sent_only=sent_only,
        score_bucket_size=args.score_bucket_size or settings.forward_performance_score_bucket_size,
        min_closed_trades=(
            args.min_closed_trades
            if args.min_closed_trades is not None
            else settings.forward_performance_min_closed_trades
        ),
        recent_minutes=args.recent_minutes,
    )


def _fmt_pf(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def print_group(title: str, data: Mapping[str, object], *, limit: int = 12) -> None:
    if not data:
        return
    print()
    print(title)
    rows = []
    for key, raw in data.items():
        if not isinstance(raw, dict):
            continue
        rows.append((key, raw))
    rows.sort(key=lambda item: (int(item[1].get("closed_with_r", 0)), float(item[1].get("total_r", 0.0))), reverse=True)
    for key, row in rows[:limit]:
        print(
            "{key:<36} cand={cand:<4} closed={closed:<4} wr={wr:>6.1%} avgR={avg:>7.3f} pf={pf:>6} dd={dd:>7.3f} sample_ok={ok}".format(
                key=key[:36],
                cand=int(row.get("candidates", 0)),
                closed=int(row.get("closed_with_r", 0)),
                wr=float(row.get("win_rate", 0.0)),
                avg=float(row.get("avg_r", 0.0)),
                pf=_fmt_pf(row.get("profit_factor", 0.0)),
                dd=float(row.get("max_drawdown_r", 0.0)),
                ok=row.get("sample_ok", False),
            )
        )


def print_report(report: Mapping[str, object]) -> None:
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    print()
    print("FORWARD PERFORMANCE REPORT")
    print(
        "Candidates: {cand} | Delivered: {delivered} | Closed: {closed} | Closed-with-R: {closed_r}".format(
            cand=int(overall.get("candidates", 0)),
            delivered=int(overall.get("delivered", 0)),
            closed=int(overall.get("closed", 0)),
            closed_r=int(overall.get("closed_with_r", 0)),
        )
    )
    print(
        "Win rate: {wr:.1%} | AvgR: {avg:.3f} | TotalR: {total:.3f} | PF: {pf} | MaxDD: {dd:.3f}R".format(
            wr=float(overall.get("win_rate", 0.0)),
            avg=float(overall.get("avg_r", 0.0)),
            total=float(overall.get("total_r", 0.0)),
            pf=_fmt_pf(overall.get("profit_factor", 0.0)),
            dd=float(overall.get("max_drawdown_r", 0.0)),
        )
    )
    print(f"Status: {overall.get('status_counts', {})}")
    print(f"Exit reasons: {overall.get('exit_reason_counts', {})}")

    for title, key in (
        ("BY PAIR", "by_pair"),
        ("BY REGIME", "by_regime"),
        ("BY SESSION", "by_session"),
        ("BY SCORE BUCKET", "by_score_bucket"),
        ("BY PRE-TRADE SHADOW VERDICT", "by_pre_trade_shadow_verdict"),
        ("BY PRE-TRADE SHADOW REASON", "by_pre_trade_shadow_reason"),
    ):
        data = report.get(key, {})
        if isinstance(data, dict):
            print_group(title, data)


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    reporter = ForwardPerformanceReporter(build_settings(settings, args))
    report = reporter.build_report()
    if args.no_rows:
        report = dict(report)
        report.pop("rows", None)
    reporter.write_report(report)
    print_report(report)
    print(f"Report saved: {reporter.settings.report_path}")


if __name__ == "__main__":
    main()
