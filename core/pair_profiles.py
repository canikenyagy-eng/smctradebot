from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.regime_gate import RegimeGateSettings
from core.session_gate import SessionGateSettings, SessionWindow, normalize_session_windows


def clean_pair(pair: str) -> str:
    return str(pair).upper().replace("/", "").strip()


def _as_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_regimes(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        return ()

    regimes: list[str] = []
    for item in values:
        clean = str(item).strip().upper()
        if clean and clean not in regimes:
            regimes.append(clean)
    return tuple(regimes)


def _window_from_text(value: str) -> SessionWindow | None:
    text = value.strip()
    if not text:
        return None
    separator = "-" if "-" in text else ":"
    if separator not in text:
        return None
    left, right = text.split(separator, 1)
    try:
        start = int(left.strip())
        end = int(right.strip())
    except ValueError:
        return None
    return (start, end)


def _as_windows(value: object) -> tuple[SessionWindow, ...]:
    if value is None:
        return ()
    raw_windows: list[SessionWindow] = []
    if isinstance(value, str):
        for item in value.split(","):
            parsed = _window_from_text(item)
            if parsed is not None:
                raw_windows.append(parsed)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                parsed = _window_from_text(item)
                if parsed is not None:
                    raw_windows.append(parsed)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    raw_windows.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
    return normalize_session_windows(tuple(raw_windows))


@dataclass(frozen=True)
class PairRuntimeProfile:
    pair: str
    min_score: int | None = None
    evaluation_step: int | None = None
    session_gate_settings: SessionGateSettings | None = None
    regime_gate_settings: RegimeGateSettings | None = None
    description: str = ""

    def sanitized(self) -> "PairRuntimeProfile":
        min_score = None if self.min_score is None else max(0, min(100, int(self.min_score)))
        evaluation_step = None if self.evaluation_step is None else max(1, int(self.evaluation_step))
        return PairRuntimeProfile(
            pair=clean_pair(self.pair),
            min_score=min_score,
            evaluation_step=evaluation_step,
            session_gate_settings=self.session_gate_settings.sanitized()
            if self.session_gate_settings is not None
            else None,
            regime_gate_settings=self.regime_gate_settings.sanitized()
            if self.regime_gate_settings is not None
            else None,
            description=str(self.description or "").strip(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pair": clean_pair(self.pair),
            "min_score": self.min_score,
            "evaluation_step": self.evaluation_step,
            "session_gate": self.session_gate_settings.to_dict()
            if self.session_gate_settings is not None
            else None,
            "regime_gate": self.regime_gate_settings.to_dict()
            if self.regime_gate_settings is not None
            else None,
            "description": self.description,
        }


def build_pair_runtime_profiles(
    raw_profiles: dict[str, Any] | None,
    *,
    enabled: bool,
    session_backtest_only: bool = True,
    allow_live_session: bool = False,
    regime_backtest_only: bool = True,
    allow_live_regime: bool = False,
) -> dict[str, PairRuntimeProfile]:
    if not enabled or not raw_profiles:
        return {}

    profiles: dict[str, PairRuntimeProfile] = {}
    for raw_pair, raw_value in raw_profiles.items():
        pair = clean_pair(str(raw_pair))
        if len(pair) != 6 or not isinstance(raw_value, dict):
            continue
        if not _as_bool(raw_value.get("enabled", True), default=True):
            continue

        min_score = _as_optional_int(raw_value.get("min_score", raw_value.get("score_threshold")))
        evaluation_step = _as_optional_int(
            raw_value.get(
                "evaluation_step",
                raw_value.get("backtest_evaluation_step", raw_value.get("cadence_step")),
            )
        )

        windows = _as_windows(
            raw_value.get(
                "session_windows_utc",
                raw_value.get("session_gate_windows_utc", raw_value.get("session")),
            )
        )
        session_gate_settings = (
            SessionGateSettings(
                enabled=True,
                windows_utc=windows,
                backtest_only=session_backtest_only,
                allow_live=allow_live_session,
            ).sanitized()
            if windows
            else None
        )

        blocked_regimes = _as_regimes(
            raw_value.get(
                "regime_blocklist",
                raw_value.get("blocked_regimes", raw_value.get("block_regimes")),
            )
        )
        regime_gate_settings = (
            RegimeGateSettings(
                enabled=True,
                blocked_regimes=blocked_regimes,
                backtest_only=regime_backtest_only,
                allow_live=allow_live_regime,
            ).sanitized()
            if blocked_regimes
            else None
        )

        profile = PairRuntimeProfile(
            pair=pair,
            min_score=min_score,
            evaluation_step=evaluation_step,
            session_gate_settings=session_gate_settings,
            regime_gate_settings=regime_gate_settings,
            description=str(raw_value.get("description", "") or "").strip(),
        ).sanitized()
        if (
            profile.min_score is None
            and profile.evaluation_step is None
            and profile.session_gate_settings is None
            and profile.regime_gate_settings is None
        ):
            continue
        profiles[pair] = profile

    return profiles
