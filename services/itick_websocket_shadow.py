"""Shadow-only iTick WebSocket quote monitor."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItickWebSocketShadowSettings:
    enabled: bool = False
    api_key: str = ""
    url: str = "wss://api.itick.org/forex"
    api_key_header: str = "token"
    auth_scheme: str = ""
    symbol_format: str = "{base}{quote}"
    region: str = "GB"
    subscription_types: str = "quote"
    log_path: Path | str = Path("logs/itick_websocket_shadow.jsonl")
    heartbeat_seconds: float = 30.0
    reconnect_seconds: float = 5.0
    stale_seconds: float = 5.0
    max_latency_seconds: float = 2.0
    connect_timeout_seconds: float = 10.0

    def normalized(self) -> "ItickWebSocketShadowSettings":
        return ItickWebSocketShadowSettings(
            enabled=bool(self.enabled),
            api_key=self.api_key.strip(),
            url=self.url.strip() or "wss://api.itick.org/forex",
            api_key_header=self.api_key_header.strip() or "token",
            auth_scheme=self.auth_scheme.strip(),
            symbol_format=self.symbol_format.strip() or "{base}{quote}",
            region=self.region.strip().upper() or "GB",
            subscription_types=self.subscription_types.strip() or "quote",
            log_path=Path(self.log_path),
            heartbeat_seconds=max(5.0, float(self.heartbeat_seconds)),
            reconnect_seconds=max(1.0, float(self.reconnect_seconds)),
            stale_seconds=max(1.0, float(self.stale_seconds)),
            max_latency_seconds=max(0.1, float(self.max_latency_seconds)),
            connect_timeout_seconds=max(1.0, float(self.connect_timeout_seconds)),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_provider_time(value: object | None) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric > 1e17:
                return datetime.fromtimestamp(numeric / 1e9, timezone.utc)
            if numeric > 1e14:
                return datetime.fromtimestamp(numeric / 1e6, timezone.utc)
            if numeric > 1e11:
                return datetime.fromtimestamp(numeric / 1e3, timezone.utc)
            return datetime.fromtimestamp(numeric, timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OSError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_pair(pair: str) -> str:
    return pair.upper().replace("/", "").strip()


def _split_pair(pair: str) -> tuple[str, str]:
    clean = _clean_pair(pair)
    if len(clean) != 6:
        raise ValueError(f"Unsupported forex symbol for iTick WebSocket shadow: {pair}")
    return clean[:3], clean[3:]


def _format_symbol(pair: str, symbol_format: str) -> str:
    base, quote = _split_pair(pair)
    return symbol_format.format(pair=f"{base}{quote}", base=base, quote=quote)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


class ItickWebSocketShadowClient:
    """Streams iTick quotes for observability without feeding signal decisions."""

    def __init__(
        self,
        settings: ItickWebSocketShadowSettings,
        quote_consumers: Iterable[Callable[[Mapping[str, object]], None]] | None = None,
    ) -> None:
        self.settings = settings.normalized()
        self.quote_consumers = tuple(quote_consumers or ())
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self, pairs: Iterable[str]) -> None:
        if not self.settings.enabled:
            return
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self.run(pairs), name="itick-websocket-shadow")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def run(self, pairs: Iterable[str], *, run_seconds: float | None = None) -> None:
        pairs_tuple = tuple(dict.fromkeys(_clean_pair(pair) for pair in pairs if pair))
        if not self.settings.enabled:
            return
        if not self.settings.api_key:
            self._write_event({"event": "disabled", "reason": "missing api key"})
            logger.warning("iTick WebSocket shadow is enabled but ITICK_API_KEY is missing")
            return
        if not pairs_tuple:
            self._write_event({"event": "disabled", "reason": "no pairs"})
            return

        deadline = time.monotonic() + run_seconds if run_seconds is not None else None
        self._write_event({"event": "starting", "pairs": list(pairs_tuple)})

        while not self._stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                await self._run_connection(pairs_tuple, deadline=deadline)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._write_event({"event": "connection_error", "error_type": exc.__class__.__name__, "error": str(exc)})
                logger.warning("iTick WebSocket shadow connection failed: %s", exc)

            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.reconnect_seconds)
            except asyncio.TimeoutError:
                continue

        self._write_event({"event": "stopped", "pairs": list(pairs_tuple)})

    async def _run_connection(self, pairs: tuple[str, ...], *, deadline: float | None) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets package is required for iTick WebSocket shadow") from exc

        headers = self._headers()
        connect_kwargs = {
            "ping_interval": None,
            "close_timeout": 5,
            "open_timeout": self.settings.connect_timeout_seconds,
        }
        signature = inspect.signature(websockets.connect)
        if "additional_headers" in signature.parameters:
            connect_kwargs["additional_headers"] = headers
        else:
            connect_kwargs["extra_headers"] = headers

        async with websockets.connect(self.settings.url, **connect_kwargs) as websocket:
            self._write_event({"event": "connected", "url": self.settings.url})
            await websocket.send(json.dumps(self._subscription_payload(pairs), separators=(",", ":")))
            self._write_event(
                {
                    "event": "subscribe_sent",
                    "pairs": list(pairs),
                    "region": self.settings.region,
                    "subscription_types": self.settings.subscription_types,
                }
            )
            next_ping = time.monotonic() + self.settings.heartbeat_seconds
            while not self._stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    break

                timeout = max(0.5, min(self.settings.heartbeat_seconds, next_ping - time.monotonic()))
                try:
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    self._handle_message(raw_message)
                except asyncio.TimeoutError:
                    pass

                if time.monotonic() >= next_ping:
                    await self._send_ping(websocket)
                    next_ping = time.monotonic() + self.settings.heartbeat_seconds

    def _headers(self) -> dict[str, str]:
        token_value = self.settings.api_key
        if self.settings.auth_scheme:
            token_value = f"{self.settings.auth_scheme} {token_value}"
        return {self.settings.api_key_header: token_value}

    def _subscription_payload(self, pairs: tuple[str, ...]) -> dict[str, str]:
        params = ",".join(f"{_format_symbol(pair, self.settings.symbol_format)}${self.settings.region}" for pair in pairs)
        return {"ac": "subscribe", "params": params, "types": self.settings.subscription_types}

    async def _send_ping(self, websocket: object) -> None:
        ping_value = str(int(time.time() * 1000))
        await websocket.send(json.dumps({"ac": "ping", "params": ping_value}, separators=(",", ":")))
        self._write_event({"event": "ping_sent", "params": ping_value})

    def _handle_message(self, raw_message: object) -> None:
        observed_at = utc_now()
        try:
            payload = json.loads(raw_message if isinstance(raw_message, str) else raw_message.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            self._write_event({"event": "invalid_json", "raw": str(raw_message)[:500]})
            return

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    self._handle_payload(item, observed_at)
            return
        if isinstance(payload, dict):
            self._handle_payload(payload, observed_at)

    def _handle_payload(self, payload: Mapping[str, object], observed_at: datetime) -> None:
        data = payload.get("data")
        if isinstance(data, dict) and data.get("type"):
            self._write_quote_payload(payload, data, observed_at)
            return

        event = str(payload.get("resAc") or payload.get("msg") or "control").strip().lower().replace(" ", "_")
        row = {
            "event": event or "control",
            "code": payload.get("code"),
            "resAc": payload.get("resAc"),
            "msg": payload.get("msg"),
        }
        if isinstance(data, dict):
            row["data"] = {key: value for key, value in data.items() if key != "params"}
        self._write_event(row, observed_at=observed_at)

    def _write_quote_payload(self, payload: Mapping[str, object], data: Mapping[str, object], observed_at: datetime) -> None:
        provider_time = parse_provider_time(data.get("t") or data.get("ts") or data.get("time"))
        latency_seconds = None
        stale = None
        slow = None
        if provider_time is not None:
            latency_seconds = (observed_at - provider_time).total_seconds()
            stale = latency_seconds > self.settings.stale_seconds
            slow = latency_seconds > self.settings.max_latency_seconds

        last_price = data.get("ld")
        if last_price is None:
            last_price = data.get("last") or data.get("price") or data.get("p") or data.get("c")

        row = {
            "event": "quote",
            "code": payload.get("code"),
            "pair": str(data.get("s") or "").upper(),
            "region": data.get("r"),
            "quote_type": data.get("type"),
            "provider_time": provider_time.isoformat() if provider_time else None,
            "latency_seconds": round(latency_seconds, 6) if latency_seconds is not None else None,
            "stale": stale,
            "slow": slow,
            "max_latency_seconds": self.settings.max_latency_seconds,
            "stale_seconds": self.settings.stale_seconds,
            "last_price": _as_float(last_price),
            "open": _as_float(data.get("o")),
            "high": _as_float(data.get("h")),
            "low": _as_float(data.get("l")),
            "volume": _as_float(data.get("v")),
        }
        for consumer in self.quote_consumers:
            try:
                consumer(row)
            except Exception as exc:
                logger.warning("iTick WebSocket quote consumer failed: %s", exc)
        self._write_event(row, observed_at=observed_at)

    def _write_event(self, payload: dict[str, object], *, observed_at: datetime | None = None) -> None:
        if not self.settings.enabled:
            return
        row = {
            "type": "itick_websocket_shadow",
            "version": 1,
            "observed_at": (observed_at or utc_now()).isoformat(),
            "source": "itick_websocket",
            **payload,
        }
        path = Path(self.settings.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


class ItickWebSocketShadowReporter:
    def __init__(
        self,
        *,
        log_path: Path | str,
        summary_path: Path | str,
        recent_minutes: int = 1440,
        stale_seconds: float = 5.0,
        max_latency_seconds: float = 2.0,
    ) -> None:
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)
        self.recent_minutes = max(1, int(recent_minutes))
        self.stale_seconds = max(1.0, float(stale_seconds))
        self.max_latency_seconds = max(0.1, float(max_latency_seconds))

    def build_report(self) -> dict[str, object]:
        now = utc_now()
        cutoff = now - timedelta(minutes=self.recent_minutes)
        rows = []
        for row in read_jsonl(self.log_path):
            if str(row.get("type")) != "itick_websocket_shadow":
                continue
            observed_at = parse_provider_time(row.get("observed_at"))
            if observed_at is None or observed_at < cutoff:
                continue
            rows.append(row)

        quote_rows = [row for row in rows if row.get("event") == "quote"]
        return {
            "type": "itick_websocket_shadow_summary",
            "version": 1,
            "generated_at": now.isoformat(),
            "settings": {
                "log_path": str(self.log_path),
                "summary_path": str(self.summary_path),
                "recent_minutes": self.recent_minutes,
                "stale_seconds": self.stale_seconds,
                "max_latency_seconds": self.max_latency_seconds,
            },
            "overall": self._stats(quote_rows),
            "events": dict(Counter(str(row.get("event", "unknown")) for row in rows)),
            "by_pair": self._group(quote_rows, "pair"),
            "latest_by_pair": self._latest_by_pair(quote_rows),
        }

    def write_report(self, report: Mapping[str, object]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def _group(self, rows: Iterable[Mapping[str, object]], key: str) -> dict[str, dict[str, object]]:
        grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(key) or "unknown")].append(row)
        return {name: self._stats(group_rows) for name, group_rows in sorted(grouped.items())}

    def _stats(self, rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
        row_list = list(rows)
        latencies = [value for row in row_list if (value := _as_float(row.get("latency_seconds"))) is not None]
        stale = [row for row in row_list if row.get("stale") is True or (_as_float(row.get("latency_seconds")) or 0.0) > self.stale_seconds]
        slow = [row for row in row_list if row.get("slow") is True or (_as_float(row.get("latency_seconds")) or 0.0) > self.max_latency_seconds]
        return {
            "quotes": len(row_list),
            "stale": len(stale),
            "slow": len(slow),
            "stale_rate": round(len(stale) / len(row_list), 6) if row_list else 0.0,
            "slow_rate": round(len(slow) / len(row_list), 6) if row_list else 0.0,
            "avg_latency_seconds": round(mean(latencies), 6) if latencies else 0.0,
            "p95_latency_seconds": round(_percentile(latencies, 0.95), 6) if latencies else 0.0,
            "max_latency_seconds": round(max(latencies), 6) if latencies else 0.0,
            "alert": bool(stale or slow or not row_list),
        }

    def _latest_by_pair(self, rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
        latest: dict[str, Mapping[str, object]] = {}
        for row in rows:
            pair = str(row.get("pair") or "unknown")
            observed_at = parse_provider_time(row.get("observed_at"))
            previous = latest.get(pair)
            previous_at = parse_provider_time(previous.get("observed_at")) if previous else None
            if observed_at is not None and (previous_at is None or observed_at > previous_at):
                latest[pair] = row
        return {
            pair: {
                "observed_at": row.get("observed_at"),
                "provider_time": row.get("provider_time"),
                "latency_seconds": row.get("latency_seconds"),
                "last_price": row.get("last_price"),
                "stale": row.get("stale"),
                "slow": row.get("slow"),
            }
            for pair, row in sorted(latest.items())
        }
