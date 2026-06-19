from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtest.engine import BacktestAccountSettings, BacktestPairReport, BacktestRunResult, BacktestTrade


@dataclass(frozen=True)
class TradeCacheSettings:
    enabled: bool = False
    cache_dir: Path | str = Path("data/cache/backtests")
    version: str = "trade_cache_v1"

    def sanitized(self) -> "TradeCacheSettings":
        return TradeCacheSettings(
            enabled=bool(self.enabled),
            cache_dir=Path(self.cache_dir),
            version=str(self.version or "trade_cache_v1").strip() or "trade_cache_v1",
        )


def stable_cache_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _trade_to_payload(trade: BacktestTrade) -> dict[str, object]:
    return _json_safe(asdict(trade))


def _trade_from_payload(raw: dict[str, object]) -> BacktestTrade:
    trade_fields = {field.name for field in fields(BacktestTrade)}
    values = {name: raw.get(name) for name in trade_fields}
    for key in ("signal_time", "entry_time", "exit_time"):
        values[key] = _parse_datetime(values[key])
    if values.get("feature_breakdown") is None:
        values["feature_breakdown"] = {}
    if values.get("smc_features") is None:
        values["smc_features"] = {}
    return BacktestTrade(**values)  # type: ignore[arg-type]


def result_to_payload(
    result: BacktestRunResult,
    *,
    key: str,
    key_payload: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    account = (result.account_settings or BacktestAccountSettings(enabled=False)).sanitized()
    return {
        "schema": "backtest_trade_cache_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "key_payload": _json_safe(key_payload),
        "metadata": _json_safe(metadata or {}),
        "result": {
            "parameters": _json_safe(result.parameters),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "news_mode": result.news_mode,
            "account_settings": _json_safe(asdict(account)),
            "pair_reports": [
                {
                    "pair": report.pair,
                    "trades": [_trade_to_payload(trade) for trade in report.trades],
                    "rejection_counts": _json_safe(report.rejection_counts),
                    "evaluations": report.evaluations,
                    "bars_processed": report.bars_processed,
                    "error": report.error,
                    "regime_evaluations": _json_safe(report.regime_evaluations or {}),
                    "regime_acceptances": _json_safe(report.regime_acceptances or {}),
                    "score_observations": list(report.score_observations or []),
                }
                for report in result.pair_reports
            ],
        },
    }


def result_from_payload(payload: dict[str, object]) -> BacktestRunResult:
    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        raise ValueError("Invalid trade cache payload: missing result")

    account_raw = result_payload.get("account_settings")
    account = BacktestAccountSettings(**account_raw).sanitized() if isinstance(account_raw, dict) else BacktestAccountSettings(enabled=False)
    reports: list[BacktestPairReport] = []
    for raw_report in result_payload.get("pair_reports", []):
        if not isinstance(raw_report, dict):
            continue
        trades = [
            _trade_from_payload(raw_trade)
            for raw_trade in raw_report.get("trades", [])
            if isinstance(raw_trade, dict)
        ]
        reports.append(
            BacktestPairReport(
                pair=str(raw_report.get("pair", "")),
                trades=trades,
                rejection_counts=dict(raw_report.get("rejection_counts") or {}),
                evaluations=int(raw_report.get("evaluations") or 0),
                bars_processed=int(raw_report.get("bars_processed") or 0),
                account_settings=account,
                error=raw_report.get("error") if raw_report.get("error") is not None else None,
                regime_evaluations=dict(raw_report.get("regime_evaluations") or {}),
                regime_acceptances=dict(raw_report.get("regime_acceptances") or {}),
                score_observations=[int(value) for value in raw_report.get("score_observations", [])],
            )
        )

    return BacktestRunResult(
        pair_reports=reports,
        parameters=dict(result_payload.get("parameters") or {}),
        started_at=_parse_datetime(result_payload.get("started_at")),
        finished_at=_parse_datetime(result_payload.get("finished_at")),
        news_mode=str(result_payload.get("news_mode") or "Cached"),
        account_settings=account,
    )


class BacktestTradeCache:
    def __init__(self, settings: TradeCacheSettings | None = None) -> None:
        self.settings = (settings or TradeCacheSettings()).sanitized()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def path_for_key(self, key: str) -> Path:
        return Path(self.settings.cache_dir) / f"{key}.json"

    def build_key(self, payload: dict[str, object]) -> str:
        wrapped = {"version": self.settings.version, "payload": payload}
        return stable_cache_key(wrapped)

    def load(self, key: str) -> BacktestRunResult | None:
        if not self.enabled:
            return None
        path = self.path_for_key(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = result_from_payload(payload)
        result.parameters["trade_cache_status"] = "hit"
        result.parameters["trade_cache_key"] = key
        result.parameters["trade_cache_path"] = str(path)
        return result

    def store(
        self,
        key: str,
        key_payload: dict[str, object],
        result: BacktestRunResult,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        path = self.path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = result_to_payload(result, key=key, key_payload=key_payload, metadata=metadata)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
        tmp_path.replace(path)
        result.parameters["trade_cache_status"] = "stored"
        result.parameters["trade_cache_key"] = key
        result.parameters["trade_cache_path"] = str(path)
        return path
