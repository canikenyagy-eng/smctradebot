from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from data.market_data import MarketDataClient

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"closed", "entry_not_filled"}


@dataclass(frozen=True)
class ForwardOutcomeSettings:
    journal_path: Path | str = Path("logs/forward_journal.jsonl")
    output_path: Path | str = Path("logs/forward_outcomes.jsonl")
    timeframe: str = "M15"
    history_limit: int = 1500
    sent_only: bool = False
    max_hold_bars: int = 48
    entry_expiry_bars: int = 0
    ambiguous_policy: str = "ambiguous"
    skip_terminal_existing: bool = True

    def normalized(self) -> "ForwardOutcomeSettings":
        policy = self.ambiguous_policy.strip().lower()
        if policy not in {"ambiguous", "stop_first", "target_first"}:
            policy = "ambiguous"
        return ForwardOutcomeSettings(
            journal_path=Path(self.journal_path),
            output_path=Path(self.output_path),
            timeframe=self.timeframe.strip().upper() or "M15",
            history_limit=max(50, int(self.history_limit)),
            sent_only=bool(self.sent_only),
            max_hold_bars=max(1, int(self.max_hold_bars)),
            entry_expiry_bars=max(0, int(self.entry_expiry_bars)),
            ambiguous_policy=policy,
            skip_terminal_existing=bool(self.skip_terminal_existing),
        )


@dataclass(frozen=True)
class ForwardCandidate:
    journal_id: str
    cycle_id: str
    fingerprint: str
    symbol: str
    side: str
    generated_at: pd.Timestamp
    entry: float
    stop_loss: float
    take_profit: float
    planned_rr: float | None
    score: int
    entry_mode: str
    entry_source: str
    time_stop_bars: int
    delivered: bool | None
    candidate_event: Mapping[str, object]
    delivery_event: Mapping[str, object] | None


@dataclass(frozen=True)
class ForwardOutcome:
    payload: dict[str, object]

    @property
    def journal_id(self) -> str:
        return str(self.payload.get("journal_id", ""))

    @property
    def status(self) -> str:
        return str(self.payload.get("status", ""))

    @property
    def exit_reason(self) -> str:
        return str(self.payload.get("exit_reason", ""))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def timestamp_or_none(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    return value.tz_convert("UTC").isoformat()


def read_jsonl(path: Path | str) -> list[dict[str, object]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    rows: list[dict[str, object]] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL row %s:%s: %s", file_path, line_no, exc)
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_candidates(journal_path: Path | str, *, sent_only: bool = False) -> list[ForwardCandidate]:
    events = read_jsonl(journal_path)
    candidates: dict[str, dict[str, object]] = {}
    deliveries: dict[str, dict[str, object]] = {}

    for event in events:
        event_type = str(event.get("type", ""))
        journal_id = str(event.get("journal_id", ""))
        if not journal_id:
            continue
        if event_type == "forward_signal_candidate":
            candidates[journal_id] = event
        elif event_type == "forward_signal_delivery":
            deliveries[journal_id] = event

    parsed: list[ForwardCandidate] = []
    for journal_id, event in candidates.items():
        signal = event.get("signal")
        if not isinstance(signal, dict):
            continue
        delivery = deliveries.get(journal_id)
        delivered = None if delivery is None else bool(delivery.get("delivered"))
        if sent_only and delivered is not True:
            continue
        try:
            parsed.append(
                ForwardCandidate(
                    journal_id=journal_id,
                    cycle_id=str(event.get("cycle_id", "")),
                    fingerprint=str(signal.get("fingerprint", event.get("fingerprint", ""))),
                    symbol=str(signal.get("symbol", "")).upper().replace("/", ""),
                    side=str(signal.get("side", "")).upper(),
                    generated_at=parse_timestamp(signal.get("generated_at")),
                    entry=float(signal.get("entry")),
                    stop_loss=float(signal.get("stop_loss")),
                    take_profit=float(signal.get("take_profit")),
                    planned_rr=(
                        float(signal.get("planned_rr")) if signal.get("planned_rr") is not None else None
                    ),
                    score=int(signal.get("score", 0)),
                    entry_mode=str(signal.get("entry_mode", "MARKET")).upper(),
                    entry_source=str(signal.get("entry_source", "")),
                    time_stop_bars=max(0, int(signal.get("time_stop_bars", 0) or 0)),
                    delivered=delivered,
                    candidate_event=event,
                    delivery_event=delivery,
                )
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed forward candidate %s: %s", journal_id, exc)

    return sorted(parsed, key=lambda item: (item.generated_at, item.symbol, item.journal_id))


def load_latest_outcomes(path: Path | str) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for event in read_jsonl(path):
        if str(event.get("type", "")) != "forward_signal_outcome":
            continue
        journal_id = str(event.get("journal_id", ""))
        if journal_id:
            latest[journal_id] = event
    return latest


class ForwardOutcomeTracker:
    def __init__(self, settings: ForwardOutcomeSettings) -> None:
        self.settings = settings.normalized()

    def run(self, market_data: MarketDataClient) -> list[ForwardOutcome]:
        candidates = load_candidates(self.settings.journal_path, sent_only=self.settings.sent_only)
        if not candidates:
            return []

        frames = self._load_symbol_frames(market_data, candidates)
        return [self.evaluate_candidate(candidate, frames.get(candidate.symbol)) for candidate in candidates]

    def append_outcomes(self, outcomes: Iterable[ForwardOutcome]) -> int:
        output_path = Path(self.settings.output_path)
        latest = load_latest_outcomes(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with output_path.open("a", encoding="utf-8") as fh:
            for outcome in outcomes:
                previous = latest.get(outcome.journal_id)
                if self._should_skip(outcome, previous):
                    continue
                fh.write(json.dumps(outcome.payload, sort_keys=True, default=str) + "\n")
                written += 1
        return written

    def evaluate_candidate(
        self,
        candidate: ForwardCandidate,
        frame: pd.DataFrame | None,
    ) -> ForwardOutcome:
        if frame is None or frame.empty:
            return self._base_outcome(
                candidate,
                status="insufficient_data",
                entry_status="unknown",
                exit_reason="no_market_data",
                bars_observed=0,
            )

        frame = self._standardize_frame(frame)
        future = frame[frame.index > candidate.generated_at]
        if future.empty:
            return self._base_outcome(
                candidate,
                status="waiting_for_data",
                entry_status="unknown",
                exit_reason="no_future_bars",
                bars_observed=0,
            )

        fill_position = self._find_fill_position(candidate, future)
        expiry_bars = self._entry_expiry_bars(candidate)
        if fill_position is None:
            if len(future) < expiry_bars:
                return self._base_outcome(
                    candidate,
                    status="pending_entry",
                    entry_status="pending",
                    exit_reason="entry_not_touched_yet",
                    bars_observed=len(future),
                    bars_to_expiry=max(0, expiry_bars - len(future)),
                )
            return self._base_outcome(
                candidate,
                status="entry_not_filled",
                entry_status="not_filled",
                exit_reason="entry_not_touched",
                bars_observed=len(future),
            )

        return self._evaluate_filled_candidate(candidate, future, fill_position)

    def summarize(self, outcomes: Iterable[ForwardOutcome]) -> dict[str, object]:
        payloads = [outcome.payload for outcome in outcomes]
        status_counts = Counter(str(row.get("status", "unknown")) for row in payloads)
        reason_counts = Counter(str(row.get("exit_reason", "unknown")) for row in payloads)
        closed = [row for row in payloads if row.get("status") == "closed" and row.get("r_multiple") is not None]
        r_values = [float(row["r_multiple"]) for row in closed]
        wins = [value for value in r_values if value > 0]
        losses = [value for value in r_values if value < 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

        by_symbol: dict[str, dict[str, object]] = {}
        for symbol, rows in self._group_by_symbol(payloads).items():
            symbol_closed = [row for row in rows if row.get("status") == "closed" and row.get("r_multiple") is not None]
            symbol_r = [float(row["r_multiple"]) for row in symbol_closed]
            symbol_wins = [value for value in symbol_r if value > 0]
            symbol_losses = [value for value in symbol_r if value < 0]
            symbol_gross_loss = abs(sum(symbol_losses))
            by_symbol[symbol] = {
                "candidates": len(rows),
                "closed": len(symbol_closed),
                "win_rate": round(len(symbol_wins) / len(symbol_closed), 6) if symbol_closed else 0.0,
                "avg_r": round(sum(symbol_r) / len(symbol_r), 6) if symbol_r else 0.0,
                "profit_factor": (
                    round(sum(symbol_wins) / symbol_gross_loss, 6)
                    if symbol_gross_loss > 0
                    else ("inf" if symbol_wins else 0.0)
                ),
                "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in rows)),
            }

        return {
            "type": "forward_outcome_summary",
            "version": 1,
            "generated_at": utc_now(),
            "settings": {
                "journal_path": str(self.settings.journal_path),
                "output_path": str(self.settings.output_path),
                "timeframe": self.settings.timeframe,
                "history_limit": self.settings.history_limit,
                "sent_only": self.settings.sent_only,
                "max_hold_bars": self.settings.max_hold_bars,
                "entry_expiry_bars": self.settings.entry_expiry_bars,
                "ambiguous_policy": self.settings.ambiguous_policy,
            },
            "candidates": len(payloads),
            "closed": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(closed), 6) if closed else 0.0,
            "avg_r": round(sum(r_values) / len(r_values), 6) if r_values else 0.0,
            "profit_factor": round(profit_factor, 6) if profit_factor != float("inf") else "inf",
            "status_counts": dict(status_counts),
            "exit_reason_counts": dict(reason_counts),
            "by_symbol": by_symbol,
        }

    def _load_symbol_frames(
        self,
        market_data: MarketDataClient,
        candidates: Iterable[ForwardCandidate],
    ) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for symbol in sorted({candidate.symbol for candidate in candidates if candidate.symbol}):
            try:
                frames[symbol] = market_data.fetch_ohlcv(
                    symbol,
                    self.settings.timeframe,
                    limit=self.settings.history_limit,
                )
            except Exception as exc:
                logger.warning("Failed to fetch outcome data for %s %s: %s", symbol, self.settings.timeframe, exc)
        return frames

    def _standardize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        if normalized.index.tz is None:
            normalized.index = normalized.index.tz_localize("UTC")
        else:
            normalized.index = normalized.index.tz_convert("UTC")
        return normalized.sort_index()

    def _find_fill_position(self, candidate: ForwardCandidate, future: pd.DataFrame) -> int | None:
        if candidate.entry_mode != "MITIGATION_LIMIT":
            return 0

        expiry_bars = self._entry_expiry_bars(candidate)
        search = future.iloc[:expiry_bars]
        for position, (_, candle) in enumerate(search.iterrows()):
            high = float(candle["high"])
            low = float(candle["low"])
            if low <= candidate.entry <= high:
                return position
        return None

    def _entry_expiry_bars(self, candidate: ForwardCandidate) -> int:
        if self.settings.entry_expiry_bars > 0:
            return self.settings.entry_expiry_bars
        return max(1, candidate.time_stop_bars or self.settings.max_hold_bars)

    def _hold_bars(self, candidate: ForwardCandidate) -> int:
        return max(1, candidate.time_stop_bars or self.settings.max_hold_bars)

    def _evaluate_filled_candidate(
        self,
        candidate: ForwardCandidate,
        future: pd.DataFrame,
        fill_position: int,
    ) -> ForwardOutcome:
        risk = abs(candidate.entry - candidate.stop_loss)
        if risk <= 0:
            return self._base_outcome(
                candidate,
                status="invalid_candidate",
                entry_status="filled",
                exit_reason="invalid_risk",
                bars_observed=len(future),
            )

        fill_time = pd.Timestamp(future.index[fill_position])
        hold_bars = self._hold_bars(candidate)
        exit_end_position = min(len(future) - 1, fill_position + hold_bars)
        evaluation = future.iloc[fill_position : exit_end_position + 1]
        if evaluation.empty:
            return self._base_outcome(
                candidate,
                status="waiting_for_data",
                entry_status="filled",
                exit_reason="no_post_fill_bars",
                entry_time=fill_time,
                bars_observed=len(future),
            )

        for offset, (timestamp, candle) in enumerate(evaluation.iterrows()):
            high = float(candle["high"])
            low = float(candle["low"])
            stop_hit, target_hit = self._hit_flags(candidate, high=high, low=low)
            if stop_hit and target_hit:
                return self._ambiguous_outcome(
                    candidate,
                    timestamp=parse_timestamp(timestamp),
                    fill_time=fill_time,
                    bars_held=offset,
                    risk=risk,
                )
            if stop_hit:
                return self._closed_outcome(
                    candidate,
                    status="closed",
                    entry_status="filled",
                    exit_reason="stop_loss",
                    entry_time=fill_time,
                    exit_time=parse_timestamp(timestamp),
                    exit_price=candidate.stop_loss,
                    r_multiple=-1.0,
                    bars_held=offset,
                    bars_observed=len(future),
                )
            if target_hit:
                return self._closed_outcome(
                    candidate,
                    status="closed",
                    entry_status="filled",
                    exit_reason="take_profit",
                    entry_time=fill_time,
                    exit_time=parse_timestamp(timestamp),
                    exit_price=candidate.take_profit,
                    r_multiple=self._target_r(candidate, risk),
                    bars_held=offset,
                    bars_observed=len(future),
                )

        if len(future) <= fill_position + hold_bars:
            return self._base_outcome(
                candidate,
                status="open",
                entry_status="filled",
                exit_reason="waiting_for_exit",
                entry_time=fill_time,
                bars_observed=len(future),
                bars_to_time_stop=max(0, fill_position + hold_bars + 1 - len(future)),
            )

        exit_candle = future.iloc[exit_end_position]
        exit_time = pd.Timestamp(future.index[exit_end_position])
        exit_price = float(exit_candle["close"])
        r_multiple = self._r_at_price(candidate, exit_price, risk)
        return self._closed_outcome(
            candidate,
            status="closed",
            entry_status="filled",
            exit_reason="time_stop",
            entry_time=fill_time,
            exit_time=exit_time,
            exit_price=exit_price,
            r_multiple=r_multiple,
            bars_held=hold_bars,
            bars_observed=len(future),
        )

    def _hit_flags(self, candidate: ForwardCandidate, *, high: float, low: float) -> tuple[bool, bool]:
        if candidate.side == "BUY":
            return low <= candidate.stop_loss, high >= candidate.take_profit
        return high >= candidate.stop_loss, low <= candidate.take_profit

    def _target_r(self, candidate: ForwardCandidate, risk: float) -> float:
        if candidate.side == "BUY":
            return round((candidate.take_profit - candidate.entry) / risk, 6)
        return round((candidate.entry - candidate.take_profit) / risk, 6)

    def _r_at_price(self, candidate: ForwardCandidate, price: float, risk: float) -> float:
        if candidate.side == "BUY":
            return round((price - candidate.entry) / risk, 6)
        return round((candidate.entry - price) / risk, 6)

    def _ambiguous_outcome(
        self,
        candidate: ForwardCandidate,
        *,
        timestamp: pd.Timestamp,
        fill_time: pd.Timestamp,
        bars_held: int,
        risk: float,
    ) -> ForwardOutcome:
        target_r = self._target_r(candidate, risk)
        if self.settings.ambiguous_policy == "stop_first":
            return self._closed_outcome(
                candidate,
                status="closed",
                entry_status="filled",
                exit_reason="ambiguous_stop_first",
                entry_time=fill_time,
                exit_time=timestamp,
                exit_price=candidate.stop_loss,
                r_multiple=-1.0,
                bars_held=bars_held,
                bars_observed=bars_held + 1,
                r_min=-1.0,
                r_max=target_r,
            )
        if self.settings.ambiguous_policy == "target_first":
            return self._closed_outcome(
                candidate,
                status="closed",
                entry_status="filled",
                exit_reason="ambiguous_target_first",
                entry_time=fill_time,
                exit_time=timestamp,
                exit_price=candidate.take_profit,
                r_multiple=target_r,
                bars_held=bars_held,
                bars_observed=bars_held + 1,
                r_min=-1.0,
                r_max=target_r,
            )
        return self._closed_outcome(
            candidate,
            status="closed",
            entry_status="filled",
            exit_reason="ambiguous_tp_sl",
            entry_time=fill_time,
            exit_time=timestamp,
            exit_price=None,
            r_multiple=None,
            bars_held=bars_held,
            bars_observed=bars_held + 1,
            r_min=-1.0,
            r_max=target_r,
        )

    def _closed_outcome(
        self,
        candidate: ForwardCandidate,
        *,
        status: str,
        entry_status: str,
        exit_reason: str,
        entry_time: pd.Timestamp | None,
        exit_time: pd.Timestamp | None,
        exit_price: float | None,
        r_multiple: float | None,
        bars_held: int,
        bars_observed: int,
        r_min: float | None = None,
        r_max: float | None = None,
    ) -> ForwardOutcome:
        payload = self._base_payload(
            candidate,
            status=status,
            entry_status=entry_status,
            exit_reason=exit_reason,
            bars_observed=bars_observed,
        )
        payload.update(
            {
                "entry_time": timestamp_or_none(entry_time),
                "exit_time": timestamp_or_none(exit_time),
                "exit_price": round(float(exit_price), 6) if exit_price is not None else None,
                "r_multiple": round(float(r_multiple), 6) if r_multiple is not None else None,
                "r_min": round(float(r_min), 6) if r_min is not None else None,
                "r_max": round(float(r_max), 6) if r_max is not None else None,
                "bars_held": int(bars_held),
            }
        )
        return ForwardOutcome(payload)

    def _base_outcome(self, candidate: ForwardCandidate, **fields: object) -> ForwardOutcome:
        payload = self._base_payload(candidate, **fields)
        return ForwardOutcome(payload)

    def _base_payload(self, candidate: ForwardCandidate, **fields: object) -> dict[str, object]:
        signal_context = candidate.candidate_event.get("signal")
        if not isinstance(signal_context, dict):
            signal_context = {}
        payload: dict[str, object] = {
            "type": "forward_signal_outcome",
            "version": 1,
            "model": "static_tp_sl_time_stop_v1",
            "observed_at": utc_now(),
            "journal_id": candidate.journal_id,
            "cycle_id": candidate.cycle_id,
            "fingerprint": candidate.fingerprint,
            "symbol": candidate.symbol,
            "side": candidate.side,
            "generated_at": timestamp_or_none(candidate.generated_at),
            "timeframe": self.settings.timeframe,
            "entry": round(candidate.entry, 6),
            "stop_loss": round(candidate.stop_loss, 6),
            "take_profit": round(candidate.take_profit, 6),
            "planned_rr": candidate.planned_rr,
            "score": candidate.score,
            "entry_mode": candidate.entry_mode,
            "entry_source": candidate.entry_source,
            "time_stop_bars": candidate.time_stop_bars,
            "delivered": candidate.delivered,
            "ambiguous_policy": self.settings.ambiguous_policy,
            "regime_label": signal_context.get("regime_label"),
            "trigger_event": signal_context.get("trigger_event"),
            "zone": signal_context.get("zone"),
        }
        payload.update(fields)
        return payload

    def _should_skip(
        self,
        outcome: ForwardOutcome,
        previous: Mapping[str, object] | None,
    ) -> bool:
        if previous is None:
            return False
        if not self.settings.skip_terminal_existing:
            return False
        previous_status = str(previous.get("status", ""))
        if previous_status in TERMINAL_STATUSES:
            return True
        return (
            previous_status == outcome.status
            and str(previous.get("exit_reason", "")) == outcome.exit_reason
            and previous.get("bars_observed") == outcome.payload.get("bars_observed")
        )

    @staticmethod
    def _group_by_symbol(payloads: Iterable[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for payload in payloads:
            grouped[str(payload.get("symbol", "UNKNOWN"))].append(payload)
        return grouped
