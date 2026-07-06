from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smc.liquidity import LiquidityContext, analyze_liquidity
from smc.structure import StructureState, detect_bos_choch


@dataclass(frozen=True)
class TriggerContext:
    direction: str
    structure_event: str | None
    structure_trend: str
    liquidity: LiquidityContext
    strength: int

    @property
    def bullish(self) -> bool:
        return self.direction == "bullish"

    @property
    def bearish(self) -> bool:
        return self.direction == "bearish"


def _resolve_direction(structure: StructureState, liquidity: LiquidityContext) -> str:
    if structure.direction in {"bullish", "bearish"}:
        return structure.direction
    if liquidity.displacement_direction in {"bullish", "bearish"}:
        return liquidity.displacement_direction
    if liquidity.sweep_direction in {"bullish", "bearish"}:
        return liquidity.sweep_direction
    return "neutral"


def analyze_trigger(
    frame: pd.DataFrame,
    swing_window: int = 2,
    *,
    pair: str | None = None,
    liquidity_tolerance_pips: float | None = None,
    liquidity_atr_tolerance_factor: float = 0.0,
) -> TriggerContext:
    if frame.empty:
        return TriggerContext(
            direction="neutral",
            structure_event=None,
            structure_trend="NEUTRAL",
            liquidity=LiquidityContext(
                equal_highs=False,
                equal_lows=False,
                equal_high_level=None,
                equal_low_level=None,
                sweep=False,
                sweep_direction=None,
                displacement=False,
                displacement_direction=None,
            ),
            strength=0,
        )

    structure = detect_bos_choch(frame, window=max(2, swing_window))
    liquidity = analyze_liquidity(
        frame,
        swing_window=max(2, swing_window),
        pair=pair,
        tolerance_pips=liquidity_tolerance_pips,
        atr_tolerance_factor=liquidity_atr_tolerance_factor,
    )
    direction = _resolve_direction(structure, liquidity)

    strength = 0
    if direction in {"bullish", "bearish"}:
        strength += 4
    if structure.event is not None:
        strength += 5
    if structure.direction in {"bullish", "bearish"} and structure.direction == direction:
        strength += 3
    if liquidity.displacement:
        strength += 6
    if liquidity.sweep:
        strength += 4
    if liquidity.equal_highs or liquidity.equal_lows:
        strength += 1
    if liquidity.displacement and liquidity.displacement_direction == direction:
        strength += 2
    if liquidity.sweep and liquidity.sweep_direction == direction:
        strength += 2

    return TriggerContext(
        direction=direction,
        structure_event=structure.event,
        structure_trend=structure.trend.upper(),
        liquidity=liquidity,
        strength=min(20, strength),
    )
