from __future__ import annotations

from dataclasses import dataclass


def normalize_regimes(regimes: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not regimes:
        return ()
    cleaned: list[str] = []
    for regime in regimes:
        item = str(regime).strip().upper()
        if item and item not in cleaned:
            cleaned.append(item)
    return tuple(cleaned)


@dataclass(frozen=True)
class RegimeGateSettings:
    enabled: bool = False
    blocked_regimes: tuple[str, ...] = ()
    backtest_only: bool = True
    allow_live: bool = False

    def sanitized(self) -> "RegimeGateSettings":
        return RegimeGateSettings(
            enabled=bool(self.enabled),
            blocked_regimes=normalize_regimes(self.blocked_regimes),
            backtest_only=bool(self.backtest_only),
            allow_live=bool(self.allow_live),
        )

    def is_active(self, runtime_mode: str) -> bool:
        if not self.enabled or not self.blocked_regimes:
            return False
        if not self.backtest_only:
            return True
        if runtime_mode.lower().strip() == "backtest":
            return True
        return self.allow_live

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "blocked_regimes": list(self.blocked_regimes),
            "backtest_only": self.backtest_only,
            "allow_live": self.allow_live,
        }


@dataclass(frozen=True)
class RegimeGateResult:
    active: bool
    allowed: bool
    regime_label: str
    blocked_regimes: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "allowed": self.allowed,
            "regime_label": self.regime_label,
            "blocked_regimes": list(self.blocked_regimes),
            "reason": self.reason,
        }


class RegimeGate:
    def __init__(self, settings: RegimeGateSettings) -> None:
        self.settings = settings.sanitized()

    def evaluate(self, regime_label: str, runtime_mode: str) -> RegimeGateResult:
        clean_label = str(regime_label).strip().upper()
        if not self.settings.is_active(runtime_mode):
            return RegimeGateResult(
                active=False,
                allowed=True,
                regime_label=clean_label,
                blocked_regimes=self.settings.blocked_regimes,
                reason="regime gate inactive",
            )

        blocked = clean_label in self.settings.blocked_regimes
        return RegimeGateResult(
            active=True,
            allowed=not blocked,
            regime_label=clean_label,
            blocked_regimes=self.settings.blocked_regimes,
            reason="blocked regime" if blocked else "allowed regime",
        )
