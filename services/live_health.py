from __future__ import annotations

import json
import os
from html import escape
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


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


@dataclass(frozen=True)
class LiveHeartbeatSettings:
    enabled: bool = False
    path: Path | str = Path("logs/live_heartbeat.json")


@dataclass(frozen=True)
class HealthCheckSettings:
    heartbeat_path: Path | str = Path("logs/live_heartbeat.json")
    max_scan_age_minutes: int = 15

    def normalized(self) -> "HealthCheckSettings":
        return HealthCheckSettings(
            heartbeat_path=Path(self.heartbeat_path),
            max_scan_age_minutes=max(1, int(self.max_scan_age_minutes)),
        )


@dataclass(frozen=True)
class HealthAlertSettings:
    enabled: bool = False
    state_path: Path | str = Path("logs/live_health_alert_state.json")
    cooldown_minutes: int = 60

    def normalized(self) -> "HealthAlertSettings":
        return HealthAlertSettings(
            enabled=bool(self.enabled),
            state_path=Path(self.state_path),
            cooldown_minutes=max(1, int(self.cooldown_minutes)),
        )


@dataclass(frozen=True)
class HealthCheckResult:
    ok: bool
    status: str
    reason: str
    heartbeat_path: str
    observed_at: str
    age_seconds: float | None
    max_age_seconds: float
    heartbeat: dict[str, object]
    components: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "status": self.status,
            "reason": self.reason,
            "heartbeat_path": self.heartbeat_path,
            "observed_at": self.observed_at,
            "age_seconds": self.age_seconds,
            "max_age_seconds": self.max_age_seconds,
            "heartbeat": self.heartbeat,
            "components": list(self.components),
        }


class LiveHeartbeatWriter:
    def __init__(self, settings: LiveHeartbeatSettings) -> None:
        self.settings = settings
        self.path = Path(settings.path)

    def engine_started(
        self,
        *,
        pairs: Sequence[str],
        data_source: str,
        live_mode: str,
        scan_interval_minutes: int,
    ) -> None:
        self._write(
            {
                "status": "engine_started",
                "pairs": list(pairs),
                "pair_count": len(pairs),
                "data_source": data_source,
                "live_mode": live_mode,
                "scan_interval_minutes": int(scan_interval_minutes),
                "pid": os.getpid(),
            }
        )

    def scan_started(
        self,
        *,
        cycle_id: str,
        pairs: Sequence[str],
        scan_started_at: datetime,
        scan_interval_minutes: int,
    ) -> None:
        self._write(
            {
                "status": "scan_started",
                "cycle_id": cycle_id,
                "pairs": list(pairs),
                "pair_count": len(pairs),
                "last_scan_started_at": scan_started_at.astimezone(timezone.utc).isoformat(),
                "scan_interval_minutes": int(scan_interval_minutes),
                "pid": os.getpid(),
            }
        )

    def scan_completed(
        self,
        *,
        cycle_id: str,
        pairs: Sequence[str],
        scan_started_at: datetime,
        duration_seconds: float,
        found_count: int,
        sent_count: int,
        scan_interval_minutes: int,
    ) -> None:
        self._write(
            {
                "status": "scan_completed",
                "cycle_id": cycle_id,
                "pairs": list(pairs),
                "pair_count": len(pairs),
                "last_scan_started_at": scan_started_at.astimezone(timezone.utc).isoformat(),
                "last_scan_completed_at": iso_now(),
                "duration_seconds": round(float(duration_seconds), 6),
                "found_count": int(found_count),
                "sent_count": int(sent_count),
                "scan_interval_minutes": int(scan_interval_minutes),
                "pid": os.getpid(),
            }
        )

    def scan_failed(
        self,
        *,
        cycle_id: str,
        pairs: Sequence[str],
        scan_started_at: datetime,
        duration_seconds: float,
        error: Exception,
        scan_interval_minutes: int,
    ) -> None:
        self._write(
            {
                "status": "scan_failed",
                "cycle_id": cycle_id,
                "pairs": list(pairs),
                "pair_count": len(pairs),
                "last_scan_started_at": scan_started_at.astimezone(timezone.utc).isoformat(),
                "last_scan_failed_at": iso_now(),
                "duration_seconds": round(float(duration_seconds), 6),
                "error_type": error.__class__.__name__,
                "error": str(error),
                "scan_interval_minutes": int(scan_interval_minutes),
                "pid": os.getpid(),
            }
        )

    def _write(self, payload: dict[str, object]) -> None:
        if not self.settings.enabled:
            return
        full_payload = {
            "type": "live_heartbeat",
            "version": 1,
            "observed_at": iso_now(),
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(full_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp_path.replace(self.path)


class LiveHealthChecker:
    def __init__(self, settings: HealthCheckSettings) -> None:
        self.settings = settings.normalized()
        self.path = Path(self.settings.heartbeat_path)

    def check(self, *, now: datetime | None = None) -> HealthCheckResult:
        checked_at = now or utc_now()
        max_age_seconds = float(self.settings.max_scan_age_minutes * 60)
        if not self.path.exists():
            return HealthCheckResult(
                ok=False,
                status="missing",
                reason="heartbeat file missing",
                heartbeat_path=str(self.path),
                observed_at=checked_at.isoformat(),
                age_seconds=None,
                max_age_seconds=max_age_seconds,
                heartbeat={},
            )

        try:
            heartbeat = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return HealthCheckResult(
                ok=False,
                status="invalid",
                reason=f"heartbeat unreadable: {exc}",
                heartbeat_path=str(self.path),
                observed_at=checked_at.isoformat(),
                age_seconds=None,
                max_age_seconds=max_age_seconds,
                heartbeat={},
            )

        if not isinstance(heartbeat, dict):
            heartbeat = {}

        status = str(heartbeat.get("status", "unknown"))
        reference_time = (
            parse_utc(heartbeat.get("last_scan_completed_at"))
            or parse_utc(heartbeat.get("last_scan_failed_at"))
            or parse_utc(heartbeat.get("last_scan_started_at"))
            or parse_utc(heartbeat.get("observed_at"))
        )
        age_seconds = None if reference_time is None else max(0.0, (checked_at - reference_time).total_seconds())

        if status == "scan_failed":
            return HealthCheckResult(
                ok=False,
                status=status,
                reason=str(heartbeat.get("error") or "last scan failed"),
                heartbeat_path=str(self.path),
                observed_at=checked_at.isoformat(),
                age_seconds=age_seconds,
                max_age_seconds=max_age_seconds,
                heartbeat=heartbeat,
            )

        if age_seconds is None:
            return HealthCheckResult(
                ok=False,
                status=status,
                reason="heartbeat has no valid timestamp",
                heartbeat_path=str(self.path),
                observed_at=checked_at.isoformat(),
                age_seconds=None,
                max_age_seconds=max_age_seconds,
                heartbeat=heartbeat,
            )

        if age_seconds > max_age_seconds:
            return HealthCheckResult(
                ok=False,
                status=status,
                reason=f"heartbeat stale: {age_seconds:.0f}s > {max_age_seconds:.0f}s",
                heartbeat_path=str(self.path),
                observed_at=checked_at.isoformat(),
                age_seconds=age_seconds,
                max_age_seconds=max_age_seconds,
                heartbeat=heartbeat,
            )

        return HealthCheckResult(
            ok=True,
            status=status,
            reason="healthy",
            heartbeat_path=str(self.path),
            observed_at=checked_at.isoformat(),
            age_seconds=age_seconds,
            max_age_seconds=max_age_seconds,
            heartbeat=heartbeat,
        )


class HealthAlertState:
    def __init__(self, settings: HealthAlertSettings) -> None:
        self.settings = settings.normalized()
        self.path = Path(self.settings.state_path)

    def load(self) -> dict[str, object]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def should_send(self, result: HealthCheckResult, *, now: datetime | None = None) -> tuple[bool, str]:
        if not self.settings.enabled:
            return False, "alerts disabled"

        state = self.load()
        checked_at = now or utc_now()
        previous_ok = state.get("last_ok")
        previous_status = str(state.get("last_status", ""))
        previous_reason = str(state.get("last_reason", ""))
        last_alert_at = parse_utc(state.get("last_alert_at"))
        cooldown_seconds = self.settings.cooldown_minutes * 60

        if result.ok:
            if previous_ok is False:
                return True, "recovery"
            return False, "healthy"

        changed = previous_status != result.status or previous_reason != result.reason
        if last_alert_at is None:
            return True, "first unhealthy alert"
        if changed:
            return True, "health status changed"
        elapsed = (checked_at - last_alert_at).total_seconds()
        if elapsed >= cooldown_seconds:
            return True, "cooldown elapsed"
        return False, f"cooldown active: {elapsed:.0f}s < {cooldown_seconds:.0f}s"

    def update(self, result: HealthCheckResult, *, alert_sent: bool, alert_reason: str) -> None:
        payload = {
            "type": "live_health_alert_state",
            "version": 1,
            "updated_at": iso_now(),
            "last_ok": bool(result.ok),
            "last_status": result.status,
            "last_reason": result.reason,
            "last_age_seconds": result.age_seconds,
            "last_alert_sent": bool(alert_sent),
            "last_alert_reason": alert_reason,
        }
        previous = self.load()
        if alert_sent:
            payload["last_alert_at"] = payload["updated_at"]
        elif previous.get("last_alert_at"):
            payload["last_alert_at"] = previous.get("last_alert_at")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp_path.replace(self.path)


def format_health_message(result: HealthCheckResult) -> str:
    heartbeat = result.heartbeat
    pairs = heartbeat.get("pairs", [])
    pair_text = ", ".join(str(pair) for pair in pairs) if isinstance(pairs, list) else str(pairs or "-")
    age_text = "n/a" if result.age_seconds is None else f"{result.age_seconds:.0f}s"
    icon = "✅" if result.ok else "🚨"
    title = "SMC BOT HEALTH OK" if result.ok else "SMC BOT HEALTH ALERT"
    component_lines = ""
    if result.components:
        lines = []
        for component in result.components[:8]:
            ok = bool(component.get("ok"))
            marker = "OK" if ok else "ALERT"
            name = escape(str(component.get("name", "component")))
            reason = escape(str(component.get("reason", "-")))
            lines.append(f"{marker} {name}: {reason}")
        component_lines = "\n\n<b>Components:</b>\n" + "\n".join(lines)
    return (
        f"{icon} <b>{title}</b>\n\n"
        f"<b>Status:</b> {escape(result.status)}\n"
        f"<b>Reason:</b> {escape(result.reason)}\n"
        f"<b>Age:</b> {age_text} / {result.max_age_seconds:.0f}s\n"
        f"<b>Cycle:</b> {escape(str(heartbeat.get('cycle_id', '-')))}\n"
        f"<b>Pairs:</b> {escape(pair_text)}\n"
        f"<b>Found/Sent:</b> {escape(str(heartbeat.get('found_count', '-')))} / {escape(str(heartbeat.get('sent_count', '-')))}\n"
        f"<b>Heartbeat:</b> {escape(result.heartbeat_path)}\n"
        f"{component_lines}\n"
        f"<b>UTC:</b> {escape(result.observed_at)}"
    )


def combine_health_components(
    result: HealthCheckResult,
    components: Sequence[Mapping[str, object]],
) -> HealthCheckResult:
    normalized = tuple(dict(component) for component in components)
    failed = [component for component in normalized if component.get("ok") is not True]
    if not failed:
        return replace(result, components=normalized)
    if not result.ok:
        return replace(result, components=normalized)
    first = failed[0]
    return replace(
        result,
        ok=False,
        status="feed_alert",
        reason=f"{first.get('name', 'feed')}: {first.get('reason', 'unhealthy')}",
        components=normalized,
    )
