from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Iterable, Mapping

from config import Settings
from services.feed_health import build_feed_health_components
from services.forward_performance import ForwardPerformanceReporter, ForwardPerformanceSettings
from services.live_health import HealthCheckSettings, LiveHealthChecker, combine_health_components


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_utc(value: object | None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_jsonl(path: Path | str) -> list[dict[str, object]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, object]] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def recent_rows(rows: Iterable[Mapping[str, object]], *, recent_minutes: int, time_key: str = "observed_at") -> list[Mapping[str, object]]:
    cutoff = utc_now() - timedelta(minutes=max(1, int(recent_minutes)))
    recent: list[Mapping[str, object]] = []
    for row in rows:
        observed_at = parse_utc(row.get(time_key))
        if observed_at is not None and observed_at >= cutoff:
            recent.append(row)
    return recent


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pf(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return str(value)
    if numeric == float("inf"):
        return "inf"
    return f"{numeric:.2f}"


def _group_leader(groups: Mapping[str, object], *, reverse: bool) -> dict[str, object] | None:
    candidates: list[tuple[str, Mapping[str, object]]] = []
    for key, raw in groups.items():
        if not isinstance(raw, Mapping):
            continue
        if int(raw.get("closed_with_r", 0) or 0) <= 0:
            continue
        candidates.append((str(key), raw))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            float(item[1].get("avg_r", 0.0) or 0.0),
            float(item[1].get("total_r", 0.0) or 0.0),
            int(item[1].get("closed_with_r", 0) or 0),
        ),
        reverse=reverse,
    )
    key, row = candidates[0]
    return {
        "name": key,
        "closed_with_r": int(row.get("closed_with_r", 0) or 0),
        "win_rate": float(row.get("win_rate", 0.0) or 0.0),
        "avg_r": float(row.get("avg_r", 0.0) or 0.0),
        "profit_factor": row.get("profit_factor", 0.0),
        "total_r": float(row.get("total_r", 0.0) or 0.0),
    }


@dataclass(frozen=True)
class DailyForwardReportSettings:
    report_path: Path | str = Path("reports/daily_live_forward_report.json")
    recent_minutes: int = 1440
    sent_only: bool = False
    min_closed_trades: int = 10
    include_rows: bool = False

    def normalized(self) -> "DailyForwardReportSettings":
        return DailyForwardReportSettings(
            report_path=Path(self.report_path),
            recent_minutes=max(1, int(self.recent_minutes)),
            sent_only=bool(self.sent_only),
            min_closed_trades=max(0, int(self.min_closed_trades)),
            include_rows=bool(self.include_rows),
        )


class DailyForwardReportBuilder:
    def __init__(self, app_settings: Settings, report_settings: DailyForwardReportSettings) -> None:
        self.app_settings = app_settings
        self.settings = report_settings.normalized()

    def build_report(self, *, outcome_update: Mapping[str, object] | None = None) -> dict[str, object]:
        performance = self._build_performance_report()
        feed_quality = self._build_feed_quality()
        live_activity = self._build_live_activity()
        leaders = self._leaders(performance)
        recommendation = self._recommendation(performance, feed_quality)

        report: dict[str, object] = {
            "type": "daily_live_forward_report",
            "version": 1,
            "generated_at": iso_now(),
            "window": {
                "recent_minutes": self.settings.recent_minutes,
                "recent_hours": round(self.settings.recent_minutes / 60.0, 3),
            },
            "settings": {
                "sent_only": self.settings.sent_only,
                "min_closed_trades": self.settings.min_closed_trades,
                "report_path": str(self.settings.report_path),
                "forward_journal_path": self.app_settings.forward_journal_log_path,
                "forward_outcome_path": self.app_settings.forward_outcome_log_path,
                "forward_performance_path": self.app_settings.forward_performance_report_path,
            },
            "outcome_update": dict(outcome_update or {}),
            "performance": performance,
            "leaders": leaders,
            "feed_quality": feed_quality,
            "live_activity": live_activity,
            "recommendation": recommendation,
        }
        if not self.settings.include_rows:
            performance.pop("rows", None)
        return report

    def write_report(self, report: Mapping[str, object]) -> None:
        path = Path(self.settings.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def _build_performance_report(self) -> dict[str, object]:
        reporter = ForwardPerformanceReporter(
            ForwardPerformanceSettings(
                journal_path=self.app_settings.forward_journal_log_path,
                outcome_path=self.app_settings.forward_outcome_log_path,
                report_path=self.app_settings.forward_performance_report_path,
                sent_only=self.settings.sent_only,
                score_bucket_size=self.app_settings.forward_performance_score_bucket_size,
                min_closed_trades=self.settings.min_closed_trades,
                recent_minutes=self.settings.recent_minutes,
            )
        )
        report = reporter.build_report()
        reporter.write_report(report)
        return report

    def _build_feed_quality(self) -> dict[str, object]:
        components = build_feed_health_components(
            self.app_settings,
            enabled=True,
            recent_minutes=self.settings.recent_minutes,
            check_itick_websocket=True,
            check_live_bars=True,
            check_redundancy=self.app_settings.feed_safe_mode_check_redundancy
            or self.app_settings.data_source.strip().lower() == "redundant",
            live_bar_max_age_seconds=self.app_settings.feed_safe_mode_live_bar_max_age_seconds,
            live_bar_max_stale_rate=self.app_settings.feed_safe_mode_live_bar_max_stale_rate,
        )
        health = LiveHealthChecker(
            HealthCheckSettings(
                heartbeat_path=self.app_settings.live_heartbeat_path,
                max_scan_age_minutes=self.app_settings.health_max_scan_age_minutes,
            )
        ).check()
        combined = combine_health_components(health, components)
        safe_mode = self._safe_mode_summary()
        alert_state = self._read_json(self.app_settings.health_alert_state_path)

        return {
            "ok": bool(combined.ok) and int(safe_mode.get("active_count", 0)) == 0,
            "health": combined.to_dict(),
            "components": components,
            "safe_mode": safe_mode,
            "health_alert_state": alert_state,
        }

    def _safe_mode_summary(self) -> dict[str, object]:
        rows = recent_rows(
            [
                row
                for row in read_jsonl(self.app_settings.feed_safe_mode_log_path)
                if str(row.get("type")) == "feed_safe_mode"
            ],
            recent_minutes=self.settings.recent_minutes,
        )
        active = [row for row in rows if row.get("active") is True]
        blocking = [row for row in active if row.get("block_signals") is True]
        reasons = Counter(str(row.get("reason", "unknown")) for row in active)
        latest = rows[-1] if rows else {}
        return {
            "checks": len(rows),
            "active_count": len(active),
            "blocking_count": len(blocking),
            "active_rate": round(len(active) / len(rows), 6) if rows else 0.0,
            "latest_active": bool(latest.get("active")) if latest else None,
            "latest_reason": latest.get("reason") if latest else None,
            "active_reasons": dict(reasons),
        }

    def _build_live_activity(self) -> dict[str, object]:
        telemetry_rows = recent_rows(
            [
                row
                for row in read_jsonl(self.app_settings.live_telemetry_log_path)
                if str(row.get("type")) in {"live_scan_completed", "live_feed_safe_mode"}
            ],
            recent_minutes=self.settings.recent_minutes,
        )
        completed = [row for row in telemetry_rows if str(row.get("type")) == "live_scan_completed"]
        safe_mode_events = [row for row in telemetry_rows if str(row.get("type")) == "live_feed_safe_mode"]
        return {
            "scan_completed_count": len(completed),
            "signals_found": sum(int(row.get("found_count", 0) or 0) for row in completed),
            "signals_sent": sum(int(row.get("sent_count", 0) or 0) for row in completed),
            "safe_mode_events": len(safe_mode_events),
            "safe_mode_active_events": sum(1 for row in safe_mode_events if row.get("active") is True),
        }

    def _leaders(self, performance: Mapping[str, object]) -> dict[str, object]:
        by_pair = _as_dict(performance.get("by_pair"))
        by_regime = _as_dict(performance.get("by_regime"))
        by_session = _as_dict(performance.get("by_session"))
        return {
            "best_pair": _group_leader(by_pair, reverse=True),
            "worst_pair": _group_leader(by_pair, reverse=False),
            "best_regime": _group_leader(by_regime, reverse=True),
            "worst_regime": _group_leader(by_regime, reverse=False),
            "best_session": _group_leader(by_session, reverse=True),
            "worst_session": _group_leader(by_session, reverse=False),
        }

    def _recommendation(self, performance: Mapping[str, object], feed_quality: Mapping[str, object]) -> dict[str, object]:
        overall = _as_dict(performance.get("overall"))
        closed = int(overall.get("closed_with_r", 0) or 0)
        avg_r = float(overall.get("avg_r", 0.0) or 0.0)
        pf = _as_float(overall.get("profit_factor")) or 0.0
        safe_mode = _as_dict(feed_quality.get("safe_mode"))

        if not bool(feed_quality.get("ok")):
            return {
                "action": "FEED_REVIEW",
                "reason": "feed health or safe mode reported degraded data",
                "live_profile_change": "none",
            }
        if closed < self.settings.min_closed_trades:
            return {
                "action": "COLLECT_MORE_FORWARD_DATA",
                "reason": f"closed sample {closed} < minimum {self.settings.min_closed_trades}",
                "live_profile_change": "none",
            }
        if int(safe_mode.get("blocking_count", 0) or 0) > 0:
            return {
                "action": "REVIEW_BLOCKED_WINDOWS",
                "reason": "safe mode blocked at least one scan in the report window",
                "live_profile_change": "none",
            }
        if avg_r >= 0.15 and pf >= 1.3:
            return {
                "action": "KEEP_PROFILE",
                "reason": "forward expectancy and PF meet target thresholds",
                "live_profile_change": "none",
            }
        if avg_r < 0.0 or pf < 1.0:
            return {
                "action": "TIGHTEN_OR_PAUSE_PROFILE",
                "reason": "forward expectancy is negative or PF is below 1.0",
                "live_profile_change": "review pair/regime/session losers",
            }
        return {
            "action": "HOLD_PROFILE",
            "reason": "forward edge is positive but not strong enough to expand",
            "live_profile_change": "none",
        }

    @staticmethod
    def _read_json(path: Path | str) -> dict[str, object]:
        file_path = Path(path)
        if not file_path.exists():
            return {}
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}


def print_daily_forward_report(report: Mapping[str, object]) -> None:
    performance = _as_dict(report.get("performance"))
    overall = _as_dict(performance.get("overall"))
    feed = _as_dict(report.get("feed_quality"))
    safe_mode = _as_dict(feed.get("safe_mode"))
    leaders = _as_dict(report.get("leaders"))
    recommendation = _as_dict(report.get("recommendation"))

    print()
    print("DAILY LIVE FORWARD REPORT")
    window = _as_dict(report.get("window"))
    print(f"Window: last {window.get('recent_minutes')} minutes")
    print(
        "Forward Signals: {candidates} | Delivered: {delivered} | Closed Outcomes: {closed}".format(
            candidates=int(overall.get("candidates", 0) or 0),
            delivered=int(overall.get("delivered", 0) or 0),
            closed=int(overall.get("closed_with_r", 0) or 0),
        )
    )
    print(
        "Win Rate: {wr:.1%} | AvgR: {avg:.3f} | PF: {pf} | MaxDD: {dd:.3f}R | TotalR: {total:.3f}".format(
            wr=float(overall.get("win_rate", 0.0) or 0.0),
            avg=float(overall.get("avg_r", 0.0) or 0.0),
            pf=_fmt_pf(overall.get("profit_factor", 0.0)),
            dd=float(overall.get("max_drawdown_r", 0.0) or 0.0),
            total=float(overall.get("total_r", 0.0) or 0.0),
        )
    )
    print(
        "Feed: {feed} | Safe-mode blocks: {blocks}/{checks} | Latest safe-mode: {latest}".format(
            feed="OK" if feed.get("ok") else "ALERT",
            blocks=int(safe_mode.get("blocking_count", 0) or 0),
            checks=int(safe_mode.get("checks", 0) or 0),
            latest=safe_mode.get("latest_reason", "-"),
        )
    )
    print(f"Best pair: {_leader_text(leaders.get('best_pair'))}")
    print(f"Worst pair: {_leader_text(leaders.get('worst_pair'))}")
    print(f"Best regime: {_leader_text(leaders.get('best_regime'))}")
    print(f"Best session: {_leader_text(leaders.get('best_session'))}")
    print(f"Recommended action: {recommendation.get('action', 'UNKNOWN')} - {recommendation.get('reason', '-')}")


def _leader_text(value: object) -> str:
    if not isinstance(value, Mapping) or not value:
        return "n/a"
    return (
        f"{value.get('name')} "
        f"closed={value.get('closed_with_r')} "
        f"wr={float(value.get('win_rate', 0.0) or 0.0):.1%} "
        f"avgR={float(value.get('avg_r', 0.0) or 0.0):.3f} "
        f"pf={_fmt_pf(value.get('profit_factor', 0.0))}"
    )


def format_daily_forward_report_message(report: Mapping[str, object]) -> str:
    performance = _as_dict(report.get("performance"))
    overall = _as_dict(performance.get("overall"))
    feed = _as_dict(report.get("feed_quality"))
    safe_mode = _as_dict(feed.get("safe_mode"))
    leaders = _as_dict(report.get("leaders"))
    recommendation = _as_dict(report.get("recommendation"))
    window = _as_dict(report.get("window"))
    action = str(recommendation.get("action", "UNKNOWN"))
    icon = "✅" if feed.get("ok") and action in {"COLLECT_MORE_FORWARD_DATA", "KEEP_PROFILE", "HOLD_PROFILE"} else "⚠️"

    return (
        f"{icon} <b>SMC DAILY FORWARD REPORT</b>\n\n"
        f"<b>Window:</b> {escape(str(window.get('recent_minutes', '-')))} min\n"
        f"<b>Signals:</b> {int(overall.get('candidates', 0) or 0)} | "
        f"<b>Delivered:</b> {int(overall.get('delivered', 0) or 0)} | "
        f"<b>Closed:</b> {int(overall.get('closed_with_r', 0) or 0)}\n"
        f"<b>WR:</b> {float(overall.get('win_rate', 0.0) or 0.0):.1%} | "
        f"<b>AvgR:</b> {float(overall.get('avg_r', 0.0) or 0.0):.3f} | "
        f"<b>PF:</b> {escape(_fmt_pf(overall.get('profit_factor', 0.0)))} | "
        f"<b>DD:</b> {float(overall.get('max_drawdown_r', 0.0) or 0.0):.3f}R\n\n"
        f"<b>Feed:</b> {'OK' if feed.get('ok') else 'ALERT'} | "
        f"<b>Safe blocks:</b> {int(safe_mode.get('blocking_count', 0) or 0)}/{int(safe_mode.get('checks', 0) or 0)}\n"
        f"<b>Safe latest:</b> {escape(str(safe_mode.get('latest_reason', '-')))}\n\n"
        f"<b>Best pair:</b> {escape(_leader_text(leaders.get('best_pair')))}\n"
        f"<b>Worst pair:</b> {escape(_leader_text(leaders.get('worst_pair')))}\n"
        f"<b>Best regime:</b> {escape(_leader_text(leaders.get('best_regime')))}\n\n"
        f"<b>Action:</b> {escape(action)}\n"
        f"<b>Reason:</b> {escape(str(recommendation.get('reason', '-')))}"
    )
