from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping


@dataclass(frozen=True)
class MarketDataDiagnosticsReportSettings:
    log_path: Path | str = Path("logs/market_data_diagnostics.jsonl")
    summary_path: Path | str = Path("reports/market_data_diagnostics_summary.json")
    recent_minutes: int = 1440
    max_latency_seconds: float = 5.0
    max_candle_age_seconds: int = 1800

    def normalized(self) -> "MarketDataDiagnosticsReportSettings":
        return MarketDataDiagnosticsReportSettings(
            log_path=Path(self.log_path),
            summary_path=Path(self.summary_path),
            recent_minutes=max(1, int(self.recent_minutes)),
            max_latency_seconds=max(0.1, float(self.max_latency_seconds)),
            max_candle_age_seconds=max(60, int(self.max_candle_age_seconds)),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_key(value: object) -> str:
    text = str(value or "unknown").strip()
    return text or "unknown"


class MarketDataDiagnosticsReporter:
    def __init__(self, settings: MarketDataDiagnosticsReportSettings) -> None:
        self.settings = settings.normalized()

    def build_report(self) -> dict[str, object]:
        now = utc_now()
        cutoff = now - timedelta(minutes=self.settings.recent_minutes)
        rows = [row for row in read_jsonl(self.settings.log_path) if str(row.get("type")) == "market_data_fetch"]
        recent = []
        for row in rows:
            observed_at = parse_utc(row.get("observed_at"))
            if observed_at is None or observed_at < cutoff:
                continue
            recent.append(row)

        report = {
            "type": "market_data_diagnostics_summary",
            "version": 1,
            "generated_at": now.isoformat(),
            "settings": {
                "log_path": str(self.settings.log_path),
                "summary_path": str(self.settings.summary_path),
                "recent_minutes": self.settings.recent_minutes,
                "max_latency_seconds": self.settings.max_latency_seconds,
                "max_candle_age_seconds": self.settings.max_candle_age_seconds,
            },
            "overall": self._stats(recent),
            "by_pair": self._group(recent, lambda row: _safe_key(row.get("pair"))),
            "by_timeframe": self._group(recent, lambda row: _safe_key(row.get("timeframe"))),
            "by_source": self._group(recent, lambda row: _safe_key(row.get("data_source"))),
            "by_pair_timeframe": self._group(
                recent,
                lambda row: f"{_safe_key(row.get('pair'))}_{_safe_key(row.get('timeframe'))}",
            ),
            "latest_by_pair_timeframe": self._latest_by_pair_timeframe(recent),
        }
        return report

    def write_report(self, report: Mapping[str, object]) -> None:
        path = Path(self.settings.summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def _group(
        self,
        rows: Iterable[Mapping[str, object]],
        key_func,
    ) -> dict[str, dict[str, object]]:
        grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[key_func(row)].append(row)
        return {key: self._stats(group_rows) for key, group_rows in sorted(grouped.items())}

    def _stats(self, rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
        row_list = list(rows)
        latencies = [value for row in row_list if (value := _as_float(row.get("latency_seconds"))) is not None]
        ages = [value for row in row_list if (value := _as_float(row.get("candle_age_seconds"))) is not None]
        errors = [row for row in row_list if row.get("ok") is False]
        stale = [row for row in row_list if bool(row.get("stale"))]
        slow = [row for row in row_list if bool(row.get("slow")) or (_as_float(row.get("latency_seconds")) or 0.0) > self.settings.max_latency_seconds]
        served_from = Counter(str(row.get("served_from", "unknown")) for row in row_list)
        observed_times = [value for row in row_list if (value := parse_utc(row.get("observed_at"))) is not None]
        candle_times = [value for row in row_list if (value := parse_utc(row.get("last_candle_time"))) is not None]
        latest_observed = max(observed_times) if observed_times else None
        latest_candle = max(candle_times) if candle_times else None

        return {
            "fetches": len(row_list),
            "ok": len(row_list) - len(errors),
            "errors": len(errors),
            "stale": len(stale),
            "slow": len(slow),
            "error_rate": round(len(errors) / len(row_list), 6) if row_list else 0.0,
            "stale_rate": round(len(stale) / len(row_list), 6) if row_list else 0.0,
            "slow_rate": round(len(slow) / len(row_list), 6) if row_list else 0.0,
            "avg_latency_seconds": round(mean(latencies), 6) if latencies else 0.0,
            "p95_latency_seconds": round(percentile(latencies, 0.95), 6) if latencies else 0.0,
            "max_latency_seconds": round(max(latencies), 6) if latencies else 0.0,
            "avg_candle_age_seconds": round(mean(ages), 3) if ages else 0.0,
            "max_candle_age_seconds": round(max(ages), 3) if ages else 0.0,
            "latest_observed_at": latest_observed.isoformat() if latest_observed else None,
            "latest_candle_time": latest_candle.isoformat() if latest_candle else None,
            "served_from": dict(served_from),
            "alert": bool(errors or stale or slow),
        }

    def _latest_by_pair_timeframe(self, rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
        latest: dict[str, Mapping[str, object]] = {}
        for row in rows:
            key = f"{_safe_key(row.get('pair'))}_{_safe_key(row.get('timeframe'))}"
            observed_at = parse_utc(row.get("observed_at"))
            previous = latest.get(key)
            previous_at = parse_utc(previous.get("observed_at")) if previous else None
            if observed_at is not None and (previous_at is None or observed_at > previous_at):
                latest[key] = row
        return {
            key: {
                "observed_at": row.get("observed_at"),
                "data_source": row.get("data_source"),
                "served_from": row.get("served_from"),
                "ok": row.get("ok"),
                "latency_seconds": row.get("latency_seconds"),
                "last_candle_time": row.get("last_candle_time"),
                "candle_age_seconds": row.get("candle_age_seconds"),
                "stale": row.get("stale"),
                "slow": row.get("slow"),
                "rows": row.get("rows"),
                "error": row.get("error"),
            }
            for key, row in sorted(latest.items())
        }
