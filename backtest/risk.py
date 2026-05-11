from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ATRRiskSettings:
    enabled: bool = False
    period: int = 14
    multiplier: float = 1.5

    def sanitized(self) -> "ATRRiskSettings":
        return ATRRiskSettings(
            enabled=bool(self.enabled),
            period=max(2, int(self.period)),
            multiplier=max(0.1, float(self.multiplier)),
        )


@dataclass(frozen=True)
class EquityProtectionSettings:
    enabled: bool = False
    max_drawdown_limit: float = 10.0
    drawdown_risk_reduction_factor: float = 0.5
    max_consecutive_losses: int = 4
    min_risk_multiplier: float = 0.25

    def sanitized(self) -> "EquityProtectionSettings":
        return EquityProtectionSettings(
            enabled=bool(self.enabled),
            max_drawdown_limit=max(0.01, float(self.max_drawdown_limit)),
            drawdown_risk_reduction_factor=max(0.0, min(1.0, float(self.drawdown_risk_reduction_factor))),
            max_consecutive_losses=max(1, int(self.max_consecutive_losses)),
            min_risk_multiplier=max(0.01, min(1.0, float(self.min_risk_multiplier))),
        )


@dataclass
class EquityProtectionState:
    settings: EquityProtectionSettings
    equity_r: float = 0.0
    peak_equity_r: float = 0.0
    consecutive_losses: int = 0
    halted: bool = False

    def drawdown_r(self) -> float:
        return max(0.0, self.peak_equity_r - self.equity_r)

    def current_risk_multiplier(self) -> float:
        if not self.settings.enabled:
            return 1.0

        dd = self.drawdown_r()
        dd_ratio = min(1.0, dd / max(1e-9, self.settings.max_drawdown_limit))
        base_multiplier = 1.0 - dd_ratio * self.settings.drawdown_risk_reduction_factor
        multiplier = max(self.settings.min_risk_multiplier, base_multiplier)

        if self.consecutive_losses >= self.settings.max_consecutive_losses:
            multiplier = min(multiplier, self.settings.min_risk_multiplier)

        return max(self.settings.min_risk_multiplier, min(1.0, multiplier))

    def allow_new_trade(self) -> bool:
        if not self.settings.enabled:
            return True
        if self.halted:
            return False
        return self.drawdown_r() < self.settings.max_drawdown_limit

    def register_trade(self, adjusted_r_multiple: float) -> None:
        if adjusted_r_multiple < 0:
            self.consecutive_losses += 1
        elif adjusted_r_multiple > 0:
            self.consecutive_losses = 0

        self.equity_r += adjusted_r_multiple
        if self.equity_r > self.peak_equity_r:
            self.peak_equity_r = self.equity_r

        if self.settings.enabled and self.drawdown_r() >= self.settings.max_drawdown_limit:
            self.halted = True


def compute_atr_series(frame: pd.DataFrame, period: int) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=max(2, period), min_periods=1).mean()


def atr_value_at(frame: pd.DataFrame, index: int, period: int) -> float | None:
    if frame.empty or index < 0 or index >= len(frame):
        return None
    atr_series = compute_atr_series(frame.iloc[: index + 1], period=period)
    if atr_series.empty:
        return None
    value = float(atr_series.iloc[-1])
    if value <= 0:
        return None
    return value

