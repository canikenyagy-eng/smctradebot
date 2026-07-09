from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from config import Settings
from services.itick_websocket_shadow import ItickWebSocketShadowReporter
from services.live_bar_builder import LiveBarBuilderReporter


def _parse_time(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_redundancy_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
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


def _build_redundancy_report(path: Path, recent_minutes: int) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=max(1, int(recent_minutes)))
    rows = []
    for row in _read_redundancy_jsonl(path):
        observed_at = _parse_time(row.get("observed_at"))
        if observed_at is not None and observed_at >= cutoff:
            rows.append(row)

    selected = Counter(str(row.get("selected_source") or "none") for row in rows)
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
        "recent_minutes": int(recent_minutes),
        "requests": len(rows),
        "ok": sum(1 for row in rows if row.get("ok") is True),
        "failed": sum(1 for row in rows if row.get("ok") is not True),
        "selected_source": dict(selected),
        "failed_attempts": dict(failed_attempts),
        "stale_attempts": dict(stale_attempts),
    }


def _component(name: str, ok: bool, reason: str, details: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "name": name,
        "ok": bool(ok),
        "reason": reason,
        "details": details or {},
    }


def _itick_component(settings: Settings, *, recent_minutes: int) -> dict[str, object]:
    reporter = ItickWebSocketShadowReporter(
        log_path=settings.itick_websocket_log_path,
        summary_path=settings.itick_websocket_summary_path,
        recent_minutes=recent_minutes,
        stale_seconds=settings.itick_websocket_stale_seconds,
        max_latency_seconds=settings.itick_websocket_max_latency_seconds,
        max_stale_rate=settings.itick_websocket_max_stale_rate,
        max_slow_rate=settings.itick_websocket_max_slow_rate,
        max_connection_errors=settings.itick_websocket_max_connection_errors,
        max_latest_quote_age_seconds=settings.itick_websocket_max_latest_quote_age_seconds,
    )
    report = reporter.build_report()
    reporter.write_report(report)
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        return _component("itick_websocket", False, "summary missing")
    ok = overall.get("alert") is not True
    reason = (
        "healthy"
        if ok
        else (
            f"quotes={overall.get('quotes', 0)} stale={overall.get('stale', 0)} "
            f"slow_rate={float(overall.get('slow_rate', 0.0)):.3%} "
            f"errors={overall.get('connection_errors', 0)} "
            f"latest_age={overall.get('latest_quote_age_seconds')}"
        )
    )
    return _component(
        "itick_websocket",
        ok,
        reason,
        {
            "quotes": overall.get("quotes", 0),
            "stale": overall.get("stale", 0),
            "slow": overall.get("slow", 0),
            "slow_rate": overall.get("slow_rate", 0.0),
            "connection_errors": overall.get("connection_errors", 0),
            "latest_quote_age_seconds": overall.get("latest_quote_age_seconds"),
        },
    )


def _live_bar_component(
    settings: Settings,
    *,
    recent_minutes: int,
    max_bar_age_seconds: float,
    max_stale_rate: float,
) -> dict[str, object]:
    reporter = LiveBarBuilderReporter(
        log_path=settings.live_bar_builder_log_path,
        recent_minutes=recent_minutes,
        max_bar_age_seconds=max_bar_age_seconds,
    )
    report = reporter.build_report()
    output = Path(settings.live_bar_builder_summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        return _component("live_bar_builder", False, "summary missing")
    updates = int(overall.get("updates", 0) or 0)
    stale_updates = int(overall.get("stale_updates", 0) or 0)
    stale_rate = stale_updates / updates if updates else 1.0
    ok = updates > 0 and stale_rate <= max_stale_rate
    reason = (
        "healthy"
        if ok
        else (
            f"updates={updates} stale_updates={stale_updates} "
            f"stale_rate={stale_rate:.3%} max_stale_rate={max_stale_rate:.3%} "
            f"max_age={overall.get('max_bar_age_seconds')}"
        )
    )
    return _component(
        "live_bar_builder",
        ok,
        reason,
        {
            "updates": updates,
            "closed": overall.get("closed", 0),
            "stale_updates": stale_updates,
            "stale_rate": round(stale_rate, 6),
            "max_stale_rate": max_stale_rate,
            "max_bar_age_seconds": overall.get("max_bar_age_seconds", 0),
        },
    )


def _redundancy_component(settings: Settings, *, recent_minutes: int) -> dict[str, object]:
    report = _build_redundancy_report(Path(settings.market_data_redundancy_log_path), recent_minutes)
    failed = int(report.get("failed", 0) or 0)
    stale_attempts = report.get("stale_attempts", {})
    stale_count = sum(int(value) for value in stale_attempts.values()) if isinstance(stale_attempts, dict) else 0
    ok = failed == 0 and stale_count == 0
    reason = "healthy" if ok else f"failed={failed} stale_attempts={stale_count}"
    return _component(
        "market_data_redundancy",
        ok,
        reason,
        {
            "requests": report.get("requests", 0),
            "ok": report.get("ok", 0),
            "failed": failed,
            "selected_source": report.get("selected_source", {}),
            "stale_attempts": stale_attempts,
        },
    )


def build_feed_health_components(
    settings: Settings,
    *,
    enabled: bool | None = None,
    recent_minutes: int | None = None,
    check_itick_websocket: bool | None = None,
    check_live_bars: bool | None = None,
    check_redundancy: bool | None = None,
    live_bar_max_age_seconds: float | None = None,
    live_bar_max_stale_rate: float | None = None,
) -> list[dict[str, object]]:
    if not (settings.enable_feed_health_checks if enabled is None else bool(enabled)):
        return []

    window_minutes = max(1, int(recent_minutes or settings.feed_health_recent_minutes))
    live_bar_age = max(
        1.0,
        float(live_bar_max_age_seconds or settings.feed_health_live_bar_max_age_seconds),
    )
    live_bar_stale_rate = max(
        0.0,
        min(
            1.0,
            float(
                settings.feed_health_live_bar_max_stale_rate
                if live_bar_max_stale_rate is None
                else live_bar_max_stale_rate
            ),
        ),
    )
    source = settings.data_source.strip().lower()
    include_itick = settings.feed_health_check_itick_websocket if check_itick_websocket is None else bool(check_itick_websocket)
    include_live_bars = settings.feed_health_check_live_bars if check_live_bars is None else bool(check_live_bars)
    include_redundancy = settings.feed_health_check_redundancy if check_redundancy is None else bool(check_redundancy)

    components: list[dict[str, object]] = []
    uses_itick_stream = settings.enable_itick_websocket_shadow or settings.enable_live_bar_builder or source in {"live_bars", "redundant"}
    if include_itick and uses_itick_stream:
        components.append(_itick_component(settings, recent_minutes=window_minutes))
    if include_live_bars and (settings.enable_live_bar_builder or source in {"live_bars", "redundant"}):
        components.append(
            _live_bar_component(
                settings,
                recent_minutes=window_minutes,
                max_bar_age_seconds=live_bar_age,
                max_stale_rate=live_bar_stale_rate,
            )
        )
    if include_redundancy or source == "redundant":
        components.append(_redundancy_component(settings, recent_minutes=window_minutes))
    return components
