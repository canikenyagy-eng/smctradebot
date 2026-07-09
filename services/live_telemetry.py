from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from core.signal_engine import TradeSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveTelemetrySettings:
    enabled: bool = False
    log_path: Path | str = Path("logs/live_telemetry.jsonl")
    include_signal_details: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveTelemetryLogger:
    """JSONL telemetry for live scan cycles without changing signal behavior."""

    def __init__(self, settings: LiveTelemetrySettings) -> None:
        self.settings = settings
        self.log_path = Path(settings.log_path)
        self._cycle_counter = 0

    def next_cycle_id(self) -> str:
        self._cycle_counter += 1
        return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{self._cycle_counter:06d}"

    def _write(self, payload: dict[str, object]) -> None:
        if not self.settings.enabled:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def event(self, event_type: str, **fields: object) -> None:
        payload: dict[str, object] = {
            "type": event_type,
            "observed_at": _utc_now(),
            **fields,
        }
        self._write(payload)

    def engine_started(
        self,
        *,
        pairs: Sequence[str],
        data_source: str,
        live_mode: str,
        scan_interval_minutes: int,
        exit_profile: str,
        pre_trade_shadow_enabled: bool,
    ) -> None:
        self.event(
            "live_engine_started",
            pairs=list(pairs),
            pair_count=len(pairs),
            data_source=data_source,
            live_mode=live_mode,
            scan_interval_minutes=scan_interval_minutes,
            exit_profile=exit_profile,
            pre_trade_shadow_enabled=pre_trade_shadow_enabled,
        )

    def scan_started(self, *, cycle_id: str, pairs: Sequence[str]) -> None:
        self.event(
            "live_scan_started",
            cycle_id=cycle_id,
            pairs=list(pairs),
            pair_count=len(pairs),
        )

    def signals_found(self, *, cycle_id: str, signals: Iterable[TradeSignal]) -> None:
        signal_list = list(signals)
        payload: dict[str, object] = {
            "cycle_id": cycle_id,
            "signal_count": len(signal_list),
            "symbols": [signal.symbol for signal in signal_list],
        }
        if self.settings.include_signal_details:
            payload["signals"] = [self._signal_payload(signal) for signal in signal_list]
        self.event("live_signals_found", **payload)

    def pre_trade_shadow_summary(self, *, cycle_id: str, rows: Sequence[dict[str, object]]) -> None:
        if not rows:
            return
        blocked = [row for row in rows if bool(row.get("would_block"))]
        self.event(
            "live_pre_trade_shadow_summary",
            cycle_id=cycle_id,
            evaluated_count=len(rows),
            would_block_count=len(blocked),
            blocked_symbols=[str(row.get("symbol")) for row in blocked],
            blocked_reasons=[str(row.get("reason")) for row in blocked],
        )

    def telegram_delivery(
        self,
        *,
        cycle_id: str,
        signal: TradeSignal,
        delivered: bool,
        latency_seconds: float,
    ) -> None:
        self.event(
            "live_telegram_delivery",
            cycle_id=cycle_id,
            delivered=bool(delivered),
            latency_seconds=round(float(latency_seconds), 6),
            signal=self._signal_payload(signal),
        )

    def scan_completed(
        self,
        *,
        cycle_id: str,
        duration_seconds: float,
        pair_count: int,
        found_count: int,
        sent_count: int,
        shadow_would_block_count: int,
    ) -> None:
        self.event(
            "live_scan_completed",
            cycle_id=cycle_id,
            duration_seconds=round(float(duration_seconds), 6),
            pair_count=int(pair_count),
            found_count=int(found_count),
            sent_count=int(sent_count),
            shadow_would_block_count=int(shadow_would_block_count),
        )

    def scan_failed(self, *, cycle_id: str, duration_seconds: float, error: Exception) -> None:
        self.event(
            "live_scan_failed",
            cycle_id=cycle_id,
            duration_seconds=round(float(duration_seconds), 6),
            error_type=error.__class__.__name__,
            error=str(error),
        )

    @staticmethod
    def _signal_payload(signal: TradeSignal) -> dict[str, object]:
        return {
            "symbol": signal.symbol,
            "fingerprint": signal.fingerprint(),
            "side": signal.side,
            "score": signal.score,
            "entry": signal.entry,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "entry_mode": signal.entry_mode,
            "entry_source": signal.entry_source,
            "regime_label": signal.regime_label,
            "regime_direction": signal.regime_direction,
            "trigger_event": signal.trigger_event,
            "trigger_strength": signal.trigger_strength,
            "zone": signal.zone,
            "generated_at": signal.generated_at.isoformat(),
        }
