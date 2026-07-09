from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from config import Settings
from services.feed_health import build_feed_health_components


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FeedSafeModeDecision:
    enabled: bool
    active: bool
    block_signals: bool
    reason: str
    observed_at: str
    components: tuple[dict[str, object], ...]

    def should_block(self) -> bool:
        return self.enabled and self.active and self.block_signals

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "active": self.active,
            "block_signals": self.block_signals,
            "reason": self.reason,
            "observed_at": self.observed_at,
            "components": list(self.components),
        }


class FeedSafeModeGuard:
    """Blocks live signal scans when feed-quality diagnostics fail."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log_path = Path(settings.feed_safe_mode_log_path)

    def evaluate(self) -> FeedSafeModeDecision:
        observed_at = _utc_now()
        if not self.settings.enable_feed_safe_mode:
            return FeedSafeModeDecision(
                enabled=False,
                active=False,
                block_signals=False,
                reason="disabled",
                observed_at=observed_at,
                components=(),
            )

        components = tuple(
            build_feed_health_components(
                self.settings,
                enabled=True,
                recent_minutes=self.settings.feed_safe_mode_recent_minutes,
                check_itick_websocket=self.settings.feed_safe_mode_check_itick_websocket,
                check_live_bars=self.settings.feed_safe_mode_check_live_bars,
                check_redundancy=self.settings.feed_safe_mode_check_redundancy,
                live_bar_max_age_seconds=self.settings.feed_safe_mode_live_bar_max_age_seconds,
                live_bar_max_stale_rate=self.settings.feed_safe_mode_live_bar_max_stale_rate,
            )
        )
        failed = self._failed_components(components)
        if not components:
            failed = ({"name": "feed_safe_mode", "ok": False, "reason": "no feed components configured", "details": {}},)

        active = bool(failed)
        reason = "healthy" if not active else f"{failed[0].get('name', 'feed')}: {failed[0].get('reason', 'unhealthy')}"
        decision = FeedSafeModeDecision(
            enabled=True,
            active=active,
            block_signals=bool(self.settings.feed_safe_mode_block_signals),
            reason=reason,
            observed_at=observed_at,
            components=components or failed,
        )
        self.write_decision(decision)
        return decision

    def write_decision(self, decision: FeedSafeModeDecision) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "type": "feed_safe_mode",
            "version": 1,
            **decision.to_dict(),
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    @staticmethod
    def _failed_components(components: Sequence[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(component for component in components if component.get("ok") is not True)
