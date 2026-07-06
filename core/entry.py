from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.shadow import ShadowFeatureContext
from smc.zones import PriceZone


@dataclass(frozen=True)
class EntryPlan:
    mode: str
    source: str
    entry: float
    stop_loss: float
    take_profit: float
    summary: str
    zone_kind: str | None
    zone_frame: str | None
    zone_lower: float | None
    zone_upper: float | None
    wait_for_fill: bool


def _side_direction(side: str) -> str:
    return "bullish" if side.upper() == "BUY" else "bearish"


def _round_price(value: float) -> float:
    return round(float(value), 5)


def _build_market_plan(frame: pd.DataFrame, side: str, entry: float, risk_reward: float) -> EntryPlan | None:
    scoped = frame.tail(25)
    buffer = max(entry * 0.00035, 0.00005)

    if side.upper() == "BUY":
        sl = float(scoped["low"].min()) - buffer
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + risk * risk_reward
        summary = f"MARKET fallback entry={_round_price(entry)} sl={_round_price(sl)} tp={_round_price(tp)}"
        return EntryPlan(
            mode="MARKET",
            source="fallback",
            entry=_round_price(entry),
            stop_loss=_round_price(sl),
            take_profit=_round_price(tp),
            summary=summary,
            zone_kind=None,
            zone_frame=None,
            zone_lower=None,
            zone_upper=None,
            wait_for_fill=False,
        )

    sl = float(scoped["high"].max()) + buffer
    risk = sl - entry
    if risk <= 0:
        return None
    tp = entry - risk * risk_reward
    summary = f"MARKET fallback entry={_round_price(entry)} sl={_round_price(sl)} tp={_round_price(tp)}"
    return EntryPlan(
        mode="MARKET",
        source="fallback",
        entry=_round_price(entry),
        stop_loss=_round_price(sl),
        take_profit=_round_price(tp),
        summary=summary,
        zone_kind=None,
        zone_frame=None,
        zone_lower=None,
        zone_upper=None,
        wait_for_fill=False,
    )


def _build_mitigation_plan(
    side: str,
    shadow: ShadowFeatureContext,
    risk_reward: float,
    *,
    min_entry_score: int,
    min_reaction_strength: float,
) -> EntryPlan | None:
    zone = shadow.entry_zone
    mitigation = shadow.entry_mitigation
    if zone is None or mitigation is None:
        return None

    side_dir = _side_direction(side)
    if zone.direction != side_dir:
        return None
    if zone.invalidated or not mitigation.touched:
        return None
    if mitigation.reaction_direction != side_dir:
        return None
    if shadow.entry_score < min_entry_score:
        return None
    if mitigation.reaction_strength < min_reaction_strength:
        return None

    entry = zone.midpoint
    buffer = max(entry * 0.00035, 0.00005, zone.width * 0.05)

    if side.upper() == "BUY":
        sl = zone.lower - buffer
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + risk * risk_reward
    else:
        sl = zone.upper + buffer
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - risk * risk_reward

    summary = (
        f"MITIGATION_LIMIT via {zone.kind.upper()}[{shadow.entry_frame or 'NA'}] "
        f"entry={_round_price(entry)} sl={_round_price(sl)} tp={_round_price(tp)} "
        f"score={shadow.entry_score} react={mitigation.reaction_strength:.0f}"
    )
    return EntryPlan(
        mode="MITIGATION_LIMIT",
        source=zone.kind.upper(),
        entry=_round_price(entry),
        stop_loss=_round_price(sl),
        take_profit=_round_price(tp),
        summary=summary,
        zone_kind=zone.kind.upper(),
        zone_frame=shadow.entry_frame.upper() if shadow.entry_frame else None,
        zone_lower=_round_price(zone.lower),
        zone_upper=_round_price(zone.upper),
        wait_for_fill=True,
    )


def build_entry_plan(
    *,
    side: str,
    current_price: float,
    ltf_frame: pd.DataFrame,
    risk_reward: float,
    shadow: ShadowFeatureContext | None = None,
    enable_mitigation_entry: bool = True,
    allow_market_fallback: bool = True,
    min_entry_score: int = 5,
    min_reaction_strength: float = 55.0,
) -> EntryPlan | None:
    if enable_mitigation_entry and shadow is not None:
        mitigation_plan = _build_mitigation_plan(
            side,
            shadow,
            risk_reward,
            min_entry_score=min_entry_score,
            min_reaction_strength=min_reaction_strength,
        )
        if mitigation_plan is not None:
            return mitigation_plan

    if not allow_market_fallback:
        return None

    return _build_market_plan(ltf_frame, side, current_price, risk_reward)
