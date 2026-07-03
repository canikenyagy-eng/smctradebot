from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


SessionWindow = tuple[int, int]


def normalize_session_windows(windows: tuple[SessionWindow, ...] | list[SessionWindow] | None) -> tuple[SessionWindow, ...]:
    if not windows:
        return ()

    normalized: list[SessionWindow] = []
    for raw_start, raw_end in windows:
        start = max(0, min(23, int(raw_start)))
        end = max(0, min(24, int(raw_end)))
        if start == end:
            continue
        item = (start, end)
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class SessionGateSettings:
    enabled: bool = False
    windows_utc: tuple[SessionWindow, ...] = ()
    backtest_only: bool = True
    allow_live: bool = False

    def sanitized(self) -> "SessionGateSettings":
        return SessionGateSettings(
            enabled=bool(self.enabled),
            windows_utc=normalize_session_windows(self.windows_utc),
            backtest_only=bool(self.backtest_only),
            allow_live=bool(self.allow_live),
        )

    def is_active(self, runtime_mode: str) -> bool:
        if not self.enabled or not self.windows_utc:
            return False
        if not self.backtest_only:
            return True
        if runtime_mode.lower().strip() == "backtest":
            return True
        return self.allow_live

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "windows_utc": [f"{start:02d}-{end:02d}" for start, end in self.windows_utc],
            "backtest_only": self.backtest_only,
            "allow_live": self.allow_live,
        }


@dataclass(frozen=True)
class SessionGateResult:
    active: bool
    allowed: bool
    hour_utc: int | None
    matched_window: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "allowed": self.allowed,
            "hour_utc": self.hour_utc,
            "matched_window": self.matched_window,
            "reason": self.reason,
        }


class SessionGate:
    def __init__(self, settings: SessionGateSettings) -> None:
        self.settings = settings.sanitized()

    def evaluate(self, signal_time: datetime, runtime_mode: str) -> SessionGateResult:
        if not self.settings.is_active(runtime_mode):
            return SessionGateResult(
                active=False,
                allowed=True,
                hour_utc=None,
                matched_window=None,
                reason="session gate inactive",
            )

        utc_time = signal_time
        if utc_time.tzinfo is not None:
            utc_time = utc_time.astimezone(timezone.utc).replace(tzinfo=None)
        hour = int(utc_time.hour)

        for start, end in self.settings.windows_utc:
            if self._hour_in_window(hour, start, end):
                return SessionGateResult(
                    active=True,
                    allowed=True,
                    hour_utc=hour,
                    matched_window=f"{start:02d}-{end:02d}",
                    reason="inside allowed session",
                )

        return SessionGateResult(
            active=True,
            allowed=False,
            hour_utc=hour,
            matched_window=None,
            reason="outside allowed session",
        )

    @staticmethod
    def _hour_in_window(hour: int, start: int, end: int) -> bool:
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end
