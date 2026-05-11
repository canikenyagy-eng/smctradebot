from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeManagementPlan:
    partial_tp_enabled: bool
    partial_take_profit: float | None
    partial_take_fraction: float
    break_even_r: float
    trailing_enabled: bool
    trailing_start_r: float
    trailing_lookback_bars: int
    time_stop_bars: int
    summary: str


def _round_price(value: float) -> float:
    return round(float(value), 5)


def build_trade_management_plan(
    *,
    side: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    partial_tp_enabled: bool,
    partial_tp_r: float,
    partial_tp_fraction: float,
    break_even_r: float,
    trailing_enabled: bool,
    trailing_start_r: float,
    trailing_lookback_bars: int,
    time_stop_bars: int,
) -> TradeManagementPlan:
    risk = abs(entry - stop_loss)
    partial_price: float | None = None

    if partial_tp_enabled and risk > 0 and partial_tp_r > 0 and partial_tp_fraction > 0:
        if side.upper() == "BUY":
            candidate = entry + risk * partial_tp_r
            if candidate < take_profit:
                partial_price = candidate
        else:
            candidate = entry - risk * partial_tp_r
            if candidate > take_profit:
                partial_price = candidate

    partial_enabled = partial_price is not None
    partial_fraction = partial_tp_fraction if partial_enabled else 0.0

    summary = (
        f"partial={'ON' if partial_enabled else 'OFF'}"
        f"{'' if partial_price is None else f'@{_round_price(partial_price)}({int(partial_fraction * 100)}%)'}"
        f" | BE@{break_even_r:.2f}R"
        f" | trail={'ON' if trailing_enabled else 'OFF'}@{trailing_start_r:.2f}R"
        f" | lookback={max(1, trailing_lookback_bars)}"
        f" | time_stop={max(0, time_stop_bars)}"
    )

    return TradeManagementPlan(
        partial_tp_enabled=partial_enabled,
        partial_take_profit=_round_price(partial_price) if partial_price is not None else None,
        partial_take_fraction=round(partial_fraction, 4),
        break_even_r=max(0.0, break_even_r),
        trailing_enabled=trailing_enabled,
        trailing_start_r=max(0.0, trailing_start_r),
        trailing_lookback_bars=max(1, trailing_lookback_bars),
        time_stop_bars=max(0, time_stop_bars),
        summary=summary,
    )
