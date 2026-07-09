from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from backtest.portfolio_layer import PortfolioDecision
from backtest.pretrade_filter import PreTradeFilter, PreTradeFilterSettings
from core.signal_engine import TradeSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreTradeShadowSettings:
    enabled: bool = False
    log_path: Path | str = Path("logs/pre_trade_filter_shadow.jsonl")
    block_expansion_continuation: bool = False
    block_expansion_continuation_fallback: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_sleeve(signal: TradeSignal) -> str:
    regime = (signal.regime_label or "").strip().lower()
    event = (signal.trigger_event or "").strip().upper()

    if regime in {"trend", "expansion"} and event in {"BOS", "BOS_CONTINUATION"}:
        return "continuation"
    if regime == "expansion":
        return "breakout_expansion"
    if regime in {"range", "contraction"}:
        return "mean_reversion"
    return "liquidity_reversal"


class PreTradeShadowLogger:
    """Logs what the pre-trade filter would do without blocking live signals."""

    def __init__(self, settings: PreTradeShadowSettings) -> None:
        self.settings = settings
        self.log_path = Path(settings.log_path)
        self.filter = PreTradeFilter(
            PreTradeFilterSettings(
                enabled=True,
                block_expansion_continuation=settings.block_expansion_continuation,
                block_expansion_continuation_fallback=settings.block_expansion_continuation_fallback,
            )
        )

    def _write(self, payload: dict[str, object]) -> None:
        if not self.settings.enabled:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def evaluate_signals(self, signals: Iterable[TradeSignal]) -> list[dict[str, object]]:
        if not self.settings.enabled:
            return []

        rows: list[dict[str, object]] = []
        for signal in signals:
            sleeve = classify_sleeve(signal)
            portfolio = PortfolioDecision(
                sleeve=sleeve,
                multiplier=1.0,
                applied=False,
                reason="shadow_live",
            )
            decision = self.filter.evaluate(signal, portfolio)
            row: dict[str, object] = {
                "type": "pre_trade_filter_shadow",
                "observed_at": _utc_now(),
                "symbol": signal.symbol,
                "fingerprint": signal.fingerprint(),
                "side": signal.side,
                "score": signal.score,
                "regime_label": signal.regime_label,
                "trigger_event": signal.trigger_event,
                "trigger_strength": signal.trigger_strength,
                "entry_mode": signal.entry_mode,
                "entry_source": signal.entry_source,
                "portfolio_sleeve": sleeve,
                "would_block": not decision.allowed,
                "reason": decision.reason,
                "block_expansion_continuation": self.settings.block_expansion_continuation,
                "block_expansion_continuation_fallback": self.settings.block_expansion_continuation_fallback,
            }
            self._write(row)
            rows.append(row)

        blocked = sum(1 for row in rows if row["would_block"])
        if rows:
            logger.info(
                "Pre-trade shadow evaluated signals=%s would_block=%s log=%s",
                len(rows),
                blocked,
                self.log_path,
            )
        return rows
