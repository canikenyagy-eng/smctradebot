from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable

from core.signal_engine import SignalEvaluation


SnapshotCacheKey = tuple[Hashable, ...]


@dataclass(frozen=True)
class SnapshotCacheSettings:
    enabled: bool = False
    max_entries: int = 50_000

    def sanitized(self) -> "SnapshotCacheSettings":
        return SnapshotCacheSettings(
            enabled=bool(self.enabled),
            max_entries=max(1_000, int(self.max_entries)),
        )


class SnapshotCache:
    def __init__(self, settings: SnapshotCacheSettings | None = None) -> None:
        self.settings = (settings or SnapshotCacheSettings()).sanitized()
        self._data: OrderedDict[SnapshotCacheKey, SignalEvaluation] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.skips = 0

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def get(self, key: SnapshotCacheKey) -> SignalEvaluation | None:
        if not self.enabled:
            return None
        value = self._data.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._data.move_to_end(key)
        return value

    def put(self, key: SnapshotCacheKey, value: SignalEvaluation) -> None:
        if not self.enabled:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        self.stores += 1
        while len(self._data) > self.settings.max_entries:
            self._data.popitem(last=False)

    def skip(self) -> None:
        self.skips += 1

    def clear(self) -> None:
        self._data.clear()
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.skips = 0

    def stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "entries": len(self._data),
            "max_entries": self.settings.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "skips": self.skips,
        }
