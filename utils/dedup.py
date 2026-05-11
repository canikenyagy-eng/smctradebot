from __future__ import annotations

from collections import deque


class SignalDeduplicator:
    def __init__(self, max_cache: int = 2000) -> None:
        self.max_cache = max_cache
        self._items = set()
        self._queue = deque()

    def seen(self, fingerprint: str) -> bool:
        return fingerprint in self._items

    def remember(self, fingerprint: str) -> None:
        if fingerprint in self._items:
            return

        self._items.add(fingerprint)
        self._queue.append(fingerprint)

        while len(self._queue) > self.max_cache:
            old = self._queue.popleft()
            self._items.discard(old)
