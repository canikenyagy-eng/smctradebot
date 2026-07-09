from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

from services.forward_outcomes import ForwardCandidate, load_candidates, load_latest_outcomes


@dataclass(frozen=True)
class ForwardPerformanceSettings:
    journal_path: Path | str = Path("logs/forward_journal.jsonl")
    outcome_path: Path | str = Path("logs/forward_outcomes.jsonl")
    report_path: Path | str = Path("reports/forward_performance_report.json")
    sent_only: bool = False
    score_bucket_size: int = 5
    min_closed_trades: int = 0
    recent_minutes: int | None = None

    def normalized(self) -> "ForwardPerformanceSettings":
        recent_minutes = None
        if self.recent_minutes is not None:
            recent_minutes = max(1, int(self.recent_minutes))
        return ForwardPerformanceSettings(
            journal_path=Path(self.journal_path),
            outcome_path=Path(self.outcome_path),
            report_path=Path(self.report_path),
            sent_only=bool(self.sent_only),
            score_bucket_size=max(1, int(self.score_bucket_size)),
            min_closed_trades=max(0, int(self.min_closed_trades)),
            recent_minutes=recent_minutes,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _profit_factor(r_values: list[float]) -> float | str:
    gross_win = sum(value for value in r_values if value > 0)
    gross_loss = abs(sum(value for value in r_values if value < 0))
    if gross_loss > 0:
        return round(gross_win / gross_loss, 6)
    if gross_win > 0:
        return "inf"
    return 0.0


def _max_drawdown(r_values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in r_values:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return round(max_dd, 6)


def session_bucket(timestamp: pd.Timestamp) -> str:
    ts = timestamp.tz_convert("UTC") if timestamp.tzinfo is not None else timestamp.tz_localize("UTC")
    hour = int(ts.hour)
    if 0 <= hour < 7:
        return "asia_00_07_utc"
    if 7 <= hour < 12:
        return "london_07_12_utc"
    if 12 <= hour < 16:
        return "london_ny_overlap_12_16_utc"
    if 16 <= hour < 21:
        return "new_york_late_16_21_utc"
    return "rollover_21_24_utc"


def score_bucket(score: int, bucket_size: int) -> str:
    bucket = max(0, int(score) // bucket_size * bucket_size)
    upper = bucket + bucket_size - 1
    return f"score_{bucket:03d}_{upper:03d}"


def pre_trade_shadow_verdict(candidate: ForwardCandidate) -> str:
    shadow = _as_dict(candidate.candidate_event.get("pre_trade_shadow"))
    if not shadow:
        return "missing"
    return "would_block" if bool(shadow.get("would_block")) else "allowed"


def pre_trade_shadow_reason(candidate: ForwardCandidate) -> str:
    shadow = _as_dict(candidate.candidate_event.get("pre_trade_shadow"))
    if not shadow:
        return "missing"
    reason = str(shadow.get("reason", "")).strip()
    return reason or "unknown"


class ForwardPerformanceReporter:
    def __init__(self, settings: ForwardPerformanceSettings) -> None:
        self.settings = settings.normalized()

    def build_report(self) -> dict[str, object]:
        candidates = load_candidates(self.settings.journal_path, sent_only=self.settings.sent_only)
        if self.settings.recent_minutes is not None:
            cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(minutes=self.settings.recent_minutes))
            candidates = [candidate for candidate in candidates if candidate.generated_at >= cutoff]
        outcomes = load_latest_outcomes(self.settings.outcome_path)
        rows = [self._row(candidate, outcomes.get(candidate.journal_id)) for candidate in candidates]

        return {
            "type": "forward_performance_report",
            "version": 1,
            "generated_at": utc_now(),
            "settings": {
                "journal_path": str(self.settings.journal_path),
                "outcome_path": str(self.settings.outcome_path),
                "report_path": str(self.settings.report_path),
                "sent_only": self.settings.sent_only,
                "score_bucket_size": self.settings.score_bucket_size,
                "min_closed_trades": self.settings.min_closed_trades,
                "recent_minutes": self.settings.recent_minutes,
            },
            "overall": self._stats(rows),
            "by_pair": self._group(rows, lambda row: str(row["symbol"])),
            "by_regime": self._group(rows, lambda row: str(row["regime_label"])),
            "by_session": self._group(rows, lambda row: str(row["session_bucket"])),
            "by_score_bucket": self._group(rows, lambda row: str(row["score_bucket"])),
            "by_pre_trade_shadow_verdict": self._group(
                rows,
                lambda row: str(row["pre_trade_shadow_verdict"]),
            ),
            "by_pre_trade_shadow_reason": self._group(
                rows,
                lambda row: str(row["pre_trade_shadow_reason"]),
            ),
            "rows": rows,
        }

    def write_report(self, report: Mapping[str, object]) -> None:
        import json

        path = Path(self.settings.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def _row(self, candidate: ForwardCandidate, outcome: Mapping[str, object] | None) -> dict[str, object]:
        signal = _as_dict(candidate.candidate_event.get("signal"))
        shadow = _as_dict(candidate.candidate_event.get("pre_trade_shadow"))
        outcome_payload = _as_dict(outcome)
        status = str(outcome_payload.get("status", "no_outcome")) if outcome_payload else "no_outcome"
        exit_reason = str(outcome_payload.get("exit_reason", "no_outcome")) if outcome_payload else "no_outcome"
        r_multiple = _as_float(outcome_payload.get("r_multiple"))
        generated_at = candidate.generated_at

        return {
            "journal_id": candidate.journal_id,
            "cycle_id": candidate.cycle_id,
            "fingerprint": candidate.fingerprint,
            "symbol": candidate.symbol,
            "side": candidate.side,
            "generated_at": generated_at.isoformat(),
            "session_bucket": session_bucket(generated_at),
            "utc_hour": int(generated_at.hour),
            "score": candidate.score,
            "score_bucket": score_bucket(candidate.score, self.settings.score_bucket_size),
            "regime_label": str(signal.get("regime_label") or outcome_payload.get("regime_label") or "UNKNOWN"),
            "regime_direction": str(signal.get("regime_direction") or "UNKNOWN"),
            "trigger_event": str(signal.get("trigger_event") or outcome_payload.get("trigger_event") or "UNKNOWN"),
            "zone": str(signal.get("zone") or outcome_payload.get("zone") or "UNKNOWN"),
            "entry_mode": candidate.entry_mode,
            "entry_source": candidate.entry_source,
            "planned_rr": candidate.planned_rr,
            "delivered": candidate.delivered,
            "pre_trade_shadow_verdict": pre_trade_shadow_verdict(candidate),
            "pre_trade_shadow_reason": pre_trade_shadow_reason(candidate),
            "pre_trade_shadow_sleeve": str(shadow.get("portfolio_sleeve", "missing")) if shadow else "missing",
            "outcome_status": status,
            "exit_reason": exit_reason,
            "r_multiple": _round(r_multiple),
            "r_min": _round(_as_float(outcome_payload.get("r_min"))),
            "r_max": _round(_as_float(outcome_payload.get("r_max"))),
            "bars_held": outcome_payload.get("bars_held"),
            "bars_observed": outcome_payload.get("bars_observed"),
            "entry_time": outcome_payload.get("entry_time"),
            "exit_time": outcome_payload.get("exit_time"),
        }

    def _group(
        self,
        rows: Iterable[Mapping[str, object]],
        key_func: Callable[[Mapping[str, object]], str],
    ) -> dict[str, dict[str, object]]:
        grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[key_func(row)].append(row)
        return {key: self._stats(group_rows) for key, group_rows in sorted(grouped.items())}

    def _stats(self, rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
        row_list = list(rows)
        closed_rows = [row for row in row_list if row.get("outcome_status") == "closed"]
        r_rows = [row for row in closed_rows if row.get("r_multiple") is not None]
        r_values = [float(row["r_multiple"]) for row in r_rows]
        wins = [value for value in r_values if value > 0]
        losses = [value for value in r_values if value < 0]
        breakeven = [value for value in r_values if value == 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        status_counts = Counter(str(row.get("outcome_status", "unknown")) for row in row_list)
        reason_counts = Counter(str(row.get("exit_reason", "unknown")) for row in row_list)
        delivered_count = sum(1 for row in row_list if row.get("delivered") is True)
        ambiguous_count = sum(1 for row in row_list if str(row.get("exit_reason", "")).startswith("ambiguous"))
        enough_sample = len(r_values) >= self.settings.min_closed_trades

        return {
            "candidates": len(row_list),
            "delivered": delivered_count,
            "delivery_rate": round(delivered_count / len(row_list), 6) if row_list else 0.0,
            "closed": len(closed_rows),
            "closed_with_r": len(r_values),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "win_rate": round(len(wins) / len(r_values), 6) if r_values else 0.0,
            "avg_r": round(sum(r_values) / len(r_values), 6) if r_values else 0.0,
            "total_r": round(sum(r_values), 6) if r_values else 0.0,
            "avg_win_r": round(avg_win, 6),
            "avg_loss_r": round(avg_loss, 6),
            "profit_factor": _profit_factor(r_values),
            "max_drawdown_r": _max_drawdown(r_values),
            "best_r": round(max(r_values), 6) if r_values else 0.0,
            "worst_r": round(min(r_values), 6) if r_values else 0.0,
            "ambiguous_count": ambiguous_count,
            "open_count": status_counts.get("open", 0),
            "pending_entry_count": status_counts.get("pending_entry", 0),
            "no_outcome_count": status_counts.get("no_outcome", 0),
            "status_counts": dict(status_counts),
            "exit_reason_counts": dict(reason_counts),
            "sample_ok": bool(enough_sample),
            "min_closed_trades": self.settings.min_closed_trades,
        }
