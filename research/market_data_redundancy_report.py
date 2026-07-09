from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from config import Settings


def parse_time(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("type") == "market_data_redundancy":
                rows.append(payload)
    return rows


def build_report(path: Path, recent_minutes: int) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=max(1, recent_minutes))
    rows = []
    for row in read_jsonl(path):
        observed_at = parse_time(row.get("observed_at"))
        if observed_at is not None and observed_at >= cutoff:
            rows.append(row)

    selected = Counter(str(row.get("selected_source") or "none") for row in rows)
    by_symbol = Counter(str(row.get("symbol") or "unknown") for row in rows)
    failed_attempts: Counter[str] = Counter()
    stale_attempts: Counter[str] = Counter()
    for row in rows:
        attempts = row.get("attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            source = str(attempt.get("source") or "unknown")
            if not attempt.get("ok"):
                failed_attempts[source] += 1
            elif attempt.get("fresh") is False:
                stale_attempts[source] += 1

    return {
        "type": "market_data_redundancy_summary",
        "generated_at": now.isoformat(),
        "log_path": str(path),
        "recent_minutes": recent_minutes,
        "requests": len(rows),
        "ok": sum(1 for row in rows if row.get("ok") is True),
        "failed": sum(1 for row in rows if row.get("ok") is not True),
        "selected_source": dict(selected),
        "by_symbol": dict(by_symbol),
        "failed_attempts": dict(failed_attempts),
        "stale_attempts": dict(stale_attempts),
    }


def print_report(report: Mapping[str, object]) -> None:
    print()
    print("MARKET DATA REDUNDANCY")
    print(
        "Requests: {requests} | OK: {ok} | Failed: {failed} | Selected: {selected}".format(
            requests=report.get("requests", 0),
            ok=report.get("ok", 0),
            failed=report.get("failed", 0),
            selected=report.get("selected_source", {}),
        )
    )
    print(f"By symbol: {report.get('by_symbol', {})}")
    print(f"Failed attempts: {report.get('failed_attempts', {})}")
    print(f"Stale attempts: {report.get('stale_attempts', {})}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize redundant market data provider decisions.")
    parser.add_argument("--log", default=None, help="Redundancy JSONL log path")
    parser.add_argument("--recent-minutes", type=int, default=1440, help="Recent window to analyze")
    args = parser.parse_args()

    settings = Settings.from_env()
    path = Path(args.log or settings.market_data_redundancy_log_path)
    report = build_report(path, args.recent_minutes)
    print_report(report)


if __name__ == "__main__":
    main()
