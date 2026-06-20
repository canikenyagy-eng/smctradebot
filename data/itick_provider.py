"""
iTick-compatible market data provider.

The provider is intentionally endpoint-configurable because iTick API deployments
can expose different REST paths/plans. It validates every response into the
engine's canonical OHLCV/tick schema and fails closed when required settings are
missing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse, urlunparse, parse_qsl
from urllib.request import Request, urlopen

import pandas as pd

from data.market_data_base import MarketDataProvider, TIMEFRAME_MAP, register_provider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItickConfig:
    api_key: str = ""
    base_url: str = ""
    ohlcv_path_template: str = ""
    ticks_path_template: str = ""
    api_key_header: str = "Authorization"
    api_key_query_param: str = ""
    auth_scheme: str = "Bearer"
    symbol_format: str = "{base}{quote}"
    timeout_seconds: float = 10.0
    timeframe_map: dict[str, str] | None = None
    extra_headers: dict[str, str] | None = None

    @classmethod
    def from_env(cls) -> "ItickConfig":
        timeframe_map = _parse_json_object(os.getenv("ITICK_TIMEFRAME_MAP_JSON", ""))
        extra_headers = _parse_json_object(os.getenv("ITICK_EXTRA_HEADERS_JSON", ""))
        return cls(
            api_key=os.getenv("ITICK_API_KEY", "").strip(),
            base_url=os.getenv("ITICK_BASE_URL", "").strip(),
            ohlcv_path_template=os.getenv("ITICK_OHLCV_PATH_TEMPLATE", "").strip(),
            ticks_path_template=os.getenv("ITICK_TICKS_PATH_TEMPLATE", "").strip(),
            api_key_header=os.getenv("ITICK_API_KEY_HEADER", "Authorization").strip(),
            api_key_query_param=os.getenv("ITICK_API_KEY_QUERY_PARAM", "").strip(),
            auth_scheme=os.getenv("ITICK_AUTH_SCHEME", "Bearer").strip(),
            symbol_format=os.getenv("ITICK_SYMBOL_FORMAT", "{base}{quote}").strip() or "{base}{quote}",
            timeout_seconds=max(1.0, float(os.getenv("ITICK_TIMEOUT_SECONDS", "10"))),
            timeframe_map={str(k).upper(): str(v) for k, v in timeframe_map.items()} if timeframe_map else None,
            extra_headers={str(k): str(v) for k, v in extra_headers.items()} if extra_headers else None,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ItickConfig":
        if not payload:
            return cls.from_env()
        env_cfg = cls.from_env()
        timeframe_map = payload.get("timeframe_map") if isinstance(payload.get("timeframe_map"), dict) else env_cfg.timeframe_map
        extra_headers = payload.get("extra_headers") if isinstance(payload.get("extra_headers"), dict) else env_cfg.extra_headers
        return cls(
            api_key=str(payload.get("api_key", env_cfg.api_key)).strip(),
            base_url=str(payload.get("base_url", env_cfg.base_url)).strip(),
            ohlcv_path_template=str(payload.get("ohlcv_path_template", env_cfg.ohlcv_path_template)).strip(),
            ticks_path_template=str(payload.get("ticks_path_template", env_cfg.ticks_path_template)).strip(),
            api_key_header=str(payload.get("api_key_header", env_cfg.api_key_header)).strip(),
            api_key_query_param=str(payload.get("api_key_query_param", env_cfg.api_key_query_param)).strip(),
            auth_scheme=str(payload.get("auth_scheme", env_cfg.auth_scheme)).strip(),
            symbol_format=str(payload.get("symbol_format", env_cfg.symbol_format)).strip() or "{base}{quote}",
            timeout_seconds=max(1.0, float(payload.get("timeout_seconds", env_cfg.timeout_seconds))),
            timeframe_map={str(k).upper(): str(v) for k, v in timeframe_map.items()} if timeframe_map else None,
            extra_headers={str(k): str(v) for k, v in extra_headers.items()} if extra_headers else None,
        )


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON object in iTick config: %s", text[:120])
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lower_map = {str(k).lower(): v for k, v in item.items()}
    for key in keys:
        if key in item:
            return item[key]
        value = lower_map.get(key.lower())
        if value is not None:
            return value
    return None


def _parse_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        raise ValueError("Missing timestamp")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1e17:
            return pd.to_datetime(int(numeric), unit="ns", utc=True)
        if numeric > 1e14:
            return pd.to_datetime(int(numeric), unit="us", utc=True)
        if numeric > 1e11:
            return pd.to_datetime(int(numeric), unit="ms", utc=True)
        return pd.to_datetime(numeric, unit="s", utc=True)
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("data", "bars", "candles", "ohlcv", "results", "items", "values"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _extract_records(value)
            if nested:
                return nested

    if any(str(key).lower() in {"open", "o", "close", "c", "timestamp", "time", "t"} for key in payload):
        return [payload]
    return []


class ItickMarketDataProvider(MarketDataProvider):
    """REST market data provider for iTick-compatible APIs."""

    def __init__(self, config: ItickConfig | None = None, history_limit: int = 500) -> None:
        super().__init__(history_limit)
        self.config = config or ItickConfig.from_env()
        self._initialized = bool(self.config.base_url and self.config.api_key)

    @staticmethod
    def _split_pair(pair: str) -> tuple[str, str]:
        clean = pair.upper().replace("/", "")
        if len(clean) != 6:
            raise ValueError(f"Unsupported forex symbol: {pair}")
        return clean[:3], clean[3:]

    def _format_symbol(self, pair: str) -> str:
        base, quote = self._split_pair(pair)
        return self.config.symbol_format.format(pair=f"{base}{quote}", base=base, quote=quote)

    def _format_interval(self, timeframe: str) -> str:
        key = timeframe.upper()
        if key not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        default_map = {
            "M1": "1m",
            "M5": "5m",
            "M15": "15m",
            "M30": "30m",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d",
        }
        mapping = self.config.timeframe_map or default_map
        return str(mapping.get(key, default_map[key]))

    def _build_url(self, template: str, pair: str, timeframe: str, limit: int) -> str:
        if not self.config.base_url:
            raise ConnectionError("ITICK_BASE_URL is required")
        if not template:
            raise ConnectionError("ITICK_OHLCV_PATH_TEMPLATE or ITICK_TICKS_PATH_TEMPLATE is required")

        base, quote = self._split_pair(pair)
        path = template.format(
            pair=f"{base}{quote}",
            symbol=self._format_symbol(pair),
            base=base,
            quote=quote,
            timeframe=timeframe.upper(),
            interval=self._format_interval(timeframe),
            limit=limit,
            api_key=self.config.api_key,
        )
        url = path if path.startswith("http://") or path.startswith("https://") else urljoin(self.config.base_url.rstrip("/") + "/", path.lstrip("/"))

        if self.config.api_key_query_param and self.config.api_key:
            parsed = urlparse(url)
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            query[self.config.api_key_query_param] = self.config.api_key
            url = urlunparse(parsed._replace(query=urlencode(query)))
        return url

    def _request_json(self, url: str) -> Any:
        headers = {"Accept": "application/json", "User-Agent": "SMCSignalEngine/1.0"}
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        if self.config.api_key_header and self.config.api_key and not self.config.api_key_query_param:
            scheme = self.config.auth_scheme.strip()
            token_value = f"{scheme} {self.config.api_key}" if scheme else self.config.api_key
            headers[self.config.api_key_header] = token_value

        request = Request(url=url, headers=headers, method="GET")
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise ConnectionError(f"iTick HTTP {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise ConnectionError(f"iTick request failed: {exc.reason}") from exc
        latency = time.monotonic() - started
        logger.debug("iTick response latency %.3fs for %s", latency, url.split("?")[0])
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("iTick response is not valid JSON") from exc

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        max_rows = limit or self.history_limit
        url = self._build_url(self.config.ohlcv_path_template, symbol, timeframe, max_rows)
        payload = self._request_json(url)
        records = _extract_records(payload)
        rows: list[dict[str, Any]] = []
        for item in records:
            timestamp_value = _first_present(item, ("timestamp", "time", "datetime", "date", "t", "ts"))
            open_value = _first_present(item, ("open", "o"))
            high_value = _first_present(item, ("high", "h"))
            low_value = _first_present(item, ("low", "l"))
            close_value = _first_present(item, ("close", "c", "last"))
            if open_value is None or high_value is None or low_value is None or close_value is None:
                continue
            rows.append(
                {
                    "timestamp": _parse_timestamp(timestamp_value),
                    "open": float(open_value),
                    "high": float(high_value),
                    "low": float(low_value),
                    "close": float(close_value),
                    "volume": float(_first_present(item, ("volume", "vol", "v", "tick_volume")) or 0.0),
                }
            )

        if not rows:
            raise ValueError(f"No iTick OHLCV rows for {symbol} {timeframe}")

        frame = pd.DataFrame(rows).set_index("timestamp")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame = self._validate_dataframe(frame, source="itick")
        self._log_data_integrity(frame, symbol, timeframe)
        return frame.tail(max_rows).copy()

    def fetch_ticks(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        if not self.config.ticks_path_template:
            raise NotImplementedError("ITICK_TICKS_PATH_TEMPLATE is not configured")
        url = self._build_url(self.config.ticks_path_template, symbol, "M1", limit)
        payload = self._request_json(url)
        records = _extract_records(payload)
        rows: list[dict[str, Any]] = []
        for item in records:
            timestamp_value = _first_present(item, ("timestamp", "time", "datetime", "date", "t", "ts"))
            bid = _first_present(item, ("bid", "b"))
            ask = _first_present(item, ("ask", "a"))
            last = _first_present(item, ("last", "price", "p", "mid"))
            if bid is None and ask is None and last is None:
                continue
            bid_float = float(bid) if bid is not None else None
            ask_float = float(ask) if ask is not None else None
            if last is None and bid_float is not None and ask_float is not None:
                last = (bid_float + ask_float) / 2.0
            rows.append(
                {
                    "timestamp": _parse_timestamp(timestamp_value),
                    "bid": bid_float,
                    "ask": ask_float,
                    "last": float(last) if last is not None else None,
                }
            )
        if not rows:
            raise ValueError(f"No iTick tick rows for {symbol}")
        frame = pd.DataFrame(rows).set_index("timestamp")
        return frame.sort_index().tail(limit).copy()

    def health_check(self) -> bool:
        if not self.config.base_url or not self.config.api_key or not self.config.ohlcv_path_template:
            return False
        try:
            self.fetch_ohlcv("EURUSD", "M5", limit=1)
            return True
        except Exception as exc:
            logger.warning("iTick health check failed: %s", exc)
            return False

    def close(self) -> None:
        super().close()


register_provider("itick", ItickMarketDataProvider)
