from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal


LiveSafety = Literal["LIVE_SAFE", "DELAYED_LIVE_SAFE", "RESEARCH_ONLY", "UNKNOWN"]


def normalize_direction(raw: object) -> str:
    if raw is None:
        return "neutral"
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        if numeric > 0:
            return "bullish"
        if numeric < 0:
            return "bearish"
    value = str(raw).strip().lower()
    if value in {"1", "bull", "bullish", "buy", "long"}:
        return "bullish"
    if value in {"-1", "bear", "bearish", "sell", "short"}:
        return "bearish"
    if value in {"high", "swing_high"}:
        return "bearish"
    if value in {"low", "swing_low"}:
        return "bullish"
    return "neutral"


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class SMCEvent:
    source: str
    event_type: str
    direction: str
    timestamp: datetime | None
    index: int
    confirmation_timestamp: datetime | None = None
    confirmation_index: int | None = None
    level: float | None = None
    top: float | None = None
    bottom: float | None = None
    strength: float | None = None
    mitigated_index: int | None = None
    swept_index: int | None = None
    live_safety: LiveSafety = "UNKNOWN"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", self.source.strip().lower())
        object.__setattr__(self, "event_type", self.event_type.strip().upper())
        object.__setattr__(self, "direction", normalize_direction(self.direction))
        if self.level is not None:
            object.__setattr__(self, "level", float(self.level))
        if self.top is not None:
            object.__setattr__(self, "top", float(self.top))
        if self.bottom is not None:
            object.__setattr__(self, "bottom", float(self.bottom))
        if self.strength is not None:
            object.__setattr__(self, "strength", float(self.strength))

    @property
    def comparable_level(self) -> float | None:
        if self.level is not None:
            return self.level
        if self.top is not None and self.bottom is not None:
            return (self.top + self.bottom) / 2.0
        if self.top is not None:
            return self.top
        if self.bottom is not None:
            return self.bottom
        return None

    @property
    def known_at_index(self) -> int:
        if self.confirmation_index is not None:
            return int(self.confirmation_index)
        return int(self.index)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    available: bool
    reason: str | None = None
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def timestamp_at(frame_index: Any, index: int) -> datetime | None:
    if index < 0 or index >= len(frame_index):
        return None
    value = frame_index[index]
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return None
