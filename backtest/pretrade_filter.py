from __future__ import annotations

from dataclasses import dataclass

from backtest.portfolio_layer import PortfolioDecision
from core.signal_engine import TradeSignal


@dataclass(frozen=True)
class PreTradeFilterSettings:
    enabled: bool = False
    block_expansion_continuation: bool = False
    block_expansion_continuation_fallback: bool = False

    def sanitized(self) -> "PreTradeFilterSettings":
        return PreTradeFilterSettings(
            enabled=bool(self.enabled),
            block_expansion_continuation=bool(self.block_expansion_continuation),
            block_expansion_continuation_fallback=bool(self.block_expansion_continuation_fallback),
        )


@dataclass(frozen=True)
class PreTradeFilterDecision:
    allowed: bool
    reason: str = "allowed"


class PreTradeFilter:
    def __init__(self, settings: PreTradeFilterSettings | None = None) -> None:
        self.settings = (settings or PreTradeFilterSettings()).sanitized()

    def evaluate(self, signal: TradeSignal, portfolio: PortfolioDecision) -> PreTradeFilterDecision:
        if not self.settings.enabled:
            return PreTradeFilterDecision(allowed=True)

        if self.settings.block_expansion_continuation:
            if (
                (signal.regime_label or "").upper() == "EXPANSION"
                and (portfolio.sleeve or "").lower() == "continuation"
            ):
                return PreTradeFilterDecision(
                    allowed=False,
                    reason="pre_trade:expansion_continuation",
                )

        if self.settings.block_expansion_continuation_fallback:
            if (
                (signal.regime_label or "").upper() == "EXPANSION"
                and (portfolio.sleeve or "").lower() == "continuation"
                and (signal.entry_source or "").lower() == "fallback"
            ):
                return PreTradeFilterDecision(
                    allowed=False,
                    reason="pre_trade:expansion_continuation_fallback",
                )

        return PreTradeFilterDecision(allowed=True)
