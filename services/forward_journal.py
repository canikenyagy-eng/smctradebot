from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.signal_engine import TradeSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForwardJournalSettings:
    enabled: bool = False
    log_path: Path | str = Path("logs/forward_journal.jsonl")
    include_score_breakdown: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _planned_risk_reward(signal: TradeSignal) -> float | None:
    risk = abs(float(signal.entry) - float(signal.stop_loss))
    if risk <= 0:
        return None
    reward = abs(float(signal.take_profit) - float(signal.entry))
    return round(reward / risk, 6)


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    return str(value)


class ForwardSignalJournal:
    """Append-only live signal journal for forward-test outcome tracking."""

    def __init__(self, settings: ForwardJournalSettings) -> None:
        self.settings = settings
        self.log_path = Path(settings.log_path)

    def build_journal_id(self, *, cycle_id: str, signal: TradeSignal) -> str:
        raw = f"{cycle_id}|{signal.symbol}|{signal.fingerprint()}|{signal.generated_at.isoformat()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def record_candidate(
        self,
        *,
        cycle_id: str,
        signal: TradeSignal,
        pre_trade_shadow: Mapping[str, object] | None = None,
    ) -> str:
        journal_id = self.build_journal_id(cycle_id=cycle_id, signal=signal)
        if not self.settings.enabled:
            return journal_id

        payload = self._candidate_payload(
            cycle_id=cycle_id,
            journal_id=journal_id,
            signal=signal,
            pre_trade_shadow=pre_trade_shadow,
        )
        self._write(payload)
        logger.info(
            "Forward journal candidate recorded | symbol=%s side=%s score=%s journal_id=%s log=%s",
            signal.symbol,
            signal.side,
            signal.score,
            journal_id,
            self.log_path,
        )
        return journal_id

    def record_delivery(
        self,
        *,
        cycle_id: str,
        journal_id: str,
        signal: TradeSignal,
        delivered: bool,
        latency_seconds: float,
    ) -> None:
        if not self.settings.enabled:
            return

        self._write(
            {
                "type": "forward_signal_delivery",
                "version": 1,
                "observed_at": _utc_now(),
                "cycle_id": cycle_id,
                "journal_id": journal_id,
                "fingerprint": signal.fingerprint(),
                "symbol": signal.symbol,
                "side": signal.side,
                "status": "sent" if delivered else "not_sent",
                "delivered": bool(delivered),
                "latency_seconds": round(float(latency_seconds), 6),
            }
        )

    def _write(self, payload: dict[str, object]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def _candidate_payload(
        self,
        *,
        cycle_id: str,
        journal_id: str,
        signal: TradeSignal,
        pre_trade_shadow: Mapping[str, object] | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "forward_signal_candidate",
            "version": 1,
            "observed_at": _utc_now(),
            "cycle_id": cycle_id,
            "journal_id": journal_id,
            "status": "candidate",
            "source": "live_signal_engine",
            "signal": self._signal_payload(signal),
        }
        if self.settings.include_score_breakdown:
            payload["score_breakdown"] = signal.score_breakdown.contribution_dict()
            payload["score_total"] = int(signal.score_breakdown.total)
        if pre_trade_shadow is not None:
            payload["pre_trade_shadow"] = {
                "would_block": bool(pre_trade_shadow.get("would_block")),
                "reason": str(pre_trade_shadow.get("reason", "")),
                "portfolio_sleeve": str(pre_trade_shadow.get("portfolio_sleeve", "")),
                "block_expansion_continuation": bool(
                    pre_trade_shadow.get("block_expansion_continuation", False)
                ),
                "block_expansion_continuation_fallback": bool(
                    pre_trade_shadow.get("block_expansion_continuation_fallback", False)
                ),
            }
        return payload

    @staticmethod
    def _signal_payload(signal: TradeSignal) -> dict[str, object]:
        return {
            "symbol": signal.symbol,
            "fingerprint": signal.fingerprint(),
            "side": signal.side,
            "generated_at": signal.generated_at.isoformat(),
            "entry": float(signal.entry),
            "stop_loss": float(signal.stop_loss),
            "take_profit": float(signal.take_profit),
            "planned_rr": _planned_risk_reward(signal),
            "score": int(signal.score),
            "htf_bias": signal.htf_bias,
            "regime_label": signal.regime_label,
            "regime_direction": signal.regime_direction,
            "zone": signal.zone,
            "trigger_direction": signal.trigger_direction,
            "trigger_event": signal.trigger_event,
            "trigger_strength": int(signal.trigger_strength),
            "structure_event": signal.structure_event,
            "structure_trend": signal.structure_trend,
            "entry_mode": signal.entry_mode,
            "entry_source": signal.entry_source,
            "entry_summary": signal.entry_summary,
            "management_summary": signal.management_summary,
            "partial_take_profit": signal.partial_take_profit,
            "partial_take_fraction": float(signal.partial_take_fraction),
            "break_even_r": float(signal.break_even_r),
            "trailing_enabled": bool(signal.trailing_enabled),
            "trailing_start_r": float(signal.trailing_start_r),
            "trailing_lookback_bars": int(signal.trailing_lookback_bars),
            "time_stop_bars": int(signal.time_stop_bars),
            "meta": _safe_value(signal.meta),
        }
