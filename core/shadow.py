from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smc.fvg import latest_fvg
from smc.mitigation import MitigationState, evaluate_mitigation
from smc.order_block import latest_order_block
from smc.smt import SMTDivergence, detect_smt_divergence
from smc.zones import PriceZone


@dataclass(frozen=True)
class ShadowFeatureContext:
    pair: str
    side: str
    current_price: float
    fvg_zone: PriceZone | None
    fvg_frame: str | None
    fvg_mitigation: MitigationState | None
    fvg_summary: str
    order_block_zone: PriceZone | None
    order_block_frame: str | None
    order_block_mitigation: MitigationState | None
    order_block_summary: str
    smt: SMTDivergence | None
    smt_summary: str
    reference_pair: str | None
    entry_zone: PriceZone | None
    entry_frame: str | None
    entry_mitigation: MitigationState | None
    entry_score: int
    entry_summary: str


@dataclass(frozen=True)
class ShadowScoreBreakdown:
    fvg_alignment: int
    order_block_alignment: int
    mitigation_alignment: int
    smt_alignment: int
    total: int

    def contribution_dict(self) -> dict[str, int]:
        return {
            "shadow_fvg": int(self.fvg_alignment),
            "shadow_ob": int(self.order_block_alignment),
            "shadow_mitigation": int(self.mitigation_alignment),
            "shadow_smt": int(self.smt_alignment),
        }


def _side_direction(side: str) -> str:
    return "bullish" if side.upper() == "BUY" else "bearish"


def _zone_proximity_score(current_price: float, zone: PriceZone) -> int:
    if zone.invalidated or zone.direction not in {"bullish", "bearish"}:
        return 0

    width = max(zone.width, 1e-9)
    midpoint_distance = abs(current_price - zone.midpoint) / width

    if zone.contains(current_price):
        score = 4
    elif midpoint_distance <= 0.25:
        score = 3
    elif midpoint_distance <= 0.5:
        score = 2
    elif midpoint_distance <= 1.0:
        score = 1
    else:
        score = 0

    score += min(2, int(round(max(0.0, min(1.0, zone.strength)) * 2)))

    if zone.is_fresh:
        score += 1
    if zone.touch_count == 0:
        score += 1
    elif zone.touch_count >= 2:
        score -= 1
    if zone.fill_ratio >= 0.75:
        score -= 1

    return max(0, min(7, score))


def _mitigation_score(side: str, mitigation: MitigationState | None) -> int:
    if mitigation is None or not mitigation.touched:
        return 0

    side_dir = _side_direction(side)
    if mitigation.reaction_direction != side_dir:
        if mitigation.reaction_direction == "neutral" and mitigation.mitigated:
            return 1
        return 0

    score = 2
    if mitigation.reaction_strength >= 80:
        score += 2
    elif mitigation.reaction_strength >= 60:
        score += 1

    if mitigation.mitigated:
        score += 1
    if mitigation.touch_depth <= 0.45:
        score += 1
    if mitigation.touch_depth <= 0.25 and not mitigation.mitigated:
        score += 1

    return max(0, min(6, score))


def _select_zone_candidate(
    side: str,
    current_price: float,
    frame: pd.DataFrame,
    zone: PriceZone | None,
    frame_label: str,
) -> tuple[PriceZone | None, str | None, MitigationState | None, int, str]:
    if zone is None:
        return None, None, None, 0, f"{frame_label}: no zone"

    if zone.direction != _side_direction(side):
        return None, None, None, 0, f"{frame_label}: opposite {zone.kind} ignored"

    mitigation = evaluate_mitigation(frame, zone)
    zone_score = _zone_proximity_score(current_price, zone)
    mitigation_score = _mitigation_score(side, mitigation)
    total_score = 0 if zone.invalidated else zone_score + mitigation_score
    summary = (
        f"{frame_label}: {zone.kind.upper()} {zone.direction.upper()} "
        f"score={zone_score} touch={zone.touch_count} fill={zone.fill_ratio:.2f} "
        f"mit={mitigation.reaction_direction.upper()}:{mitigation.reaction_strength:.0f}"
    )
    if zone.invalidated:
        summary += " invalidated"
    return zone, frame_label, mitigation, total_score, summary + f" | mit_score={mitigation_score} total={total_score}"


def _pick_best_candidate(
    first: tuple[PriceZone | None, str | None, MitigationState | None, int, str],
    second: tuple[PriceZone | None, str | None, MitigationState | None, int, str],
) -> tuple[PriceZone | None, str | None, MitigationState | None, int, str]:
    if first[3] > second[3]:
        return first
    if second[3] > first[3]:
        return second

    first_zone = first[0]
    second_zone = second[0]
    first_strength = first_zone.strength if first_zone is not None else -1.0
    second_strength = second_zone.strength if second_zone is not None else -1.0
    if first_strength > second_strength:
        return first
    if second_strength > first_strength:
        return second

    first_touch = first_zone.touch_count if first_zone is not None else 999
    second_touch = second_zone.touch_count if second_zone is not None else 999
    if first_touch <= second_touch:
        return first
    return second


def analyze_shadow_context(
    pair: str,
    side: str,
    current_price: float,
    ltf_frame: pd.DataFrame,
    trigger_frame: pd.DataFrame,
    *,
    reference_pair: str | None = None,
    reference_frame: pd.DataFrame | None = None,
    include_order_block: bool = True,
) -> ShadowFeatureContext:
    side_u = side.upper()
    side_dir = _side_direction(side_u)

    ltf_fvg = _select_zone_candidate(
        side_u,
        current_price,
        ltf_frame,
        latest_fvg(ltf_frame, direction=side_dir),
        "LTF",
    )
    trigger_fvg = _select_zone_candidate(
        side_u,
        current_price,
        trigger_frame,
        latest_fvg(trigger_frame, direction=side_dir),
        "TRIGGER",
    )
    fvg_candidate = _pick_best_candidate(ltf_fvg, trigger_fvg)
    fvg_zone, fvg_frame, fvg_mitigation, fvg_score, fvg_summary = fvg_candidate

    if include_order_block:
        ltf_ob = _select_zone_candidate(
            side_u,
            current_price,
            ltf_frame,
            latest_order_block(ltf_frame, direction=side_dir),
            "LTF",
        )
        trigger_ob = _select_zone_candidate(
            side_u,
            current_price,
            trigger_frame,
            latest_order_block(trigger_frame, direction=side_dir),
            "TRIGGER",
        )
        ob_candidate = _pick_best_candidate(ltf_ob, trigger_ob)
    else:
        ob_candidate = (None, None, None, 0, "ORDER_BLOCK: disabled by runtime settings")
    ob_zone, ob_frame, ob_mitigation, ob_score, ob_summary = ob_candidate
    entry_zone, entry_frame, entry_mitigation, entry_score, entry_summary = _pick_best_candidate(
        fvg_candidate,
        ob_candidate,
    )

    mitigation_candidates = [candidate for candidate in (fvg_mitigation, ob_mitigation) if candidate is not None and candidate.touched]
    mitigation_score = max((_mitigation_score(side_u, candidate) for candidate in mitigation_candidates), default=0)

    smt: SMTDivergence | None = None
    smt_score = 0
    smt_summary = "SMT unavailable"
    if reference_pair is not None and reference_frame is not None and not reference_frame.empty:
        smt = detect_smt_divergence(
            trigger_frame,
            reference_frame,
            target_pair=pair,
            reference_pair=reference_pair,
            swing_window=3,
            lookback=min(80, len(trigger_frame), len(reference_frame)),
        )
        if smt is None:
            smt_summary = f"{reference_pair}: no SMT divergence"
        elif smt.direction == side_dir:
            smt_score = min(8, 4 + int(round(max(0.0, min(100.0, smt.strength)) / 20.0)))
            smt_summary = f"{reference_pair}: {smt.kind.upper()} {smt.direction.upper()} score={smt_score}"
        else:
            smt_score = 0
            smt_summary = f"{reference_pair}: opposite SMT ignored"
    elif reference_pair is not None:
        smt_summary = f"{reference_pair}: reference frame unavailable"

    return ShadowFeatureContext(
        pair=pair,
        side=side_u,
        current_price=current_price,
        fvg_zone=fvg_zone,
        fvg_frame=fvg_frame,
        fvg_mitigation=fvg_mitigation,
        fvg_summary=fvg_summary,
        order_block_zone=ob_zone,
        order_block_frame=ob_frame,
        order_block_mitigation=ob_mitigation,
        order_block_summary=ob_summary,
        smt=smt,
        smt_summary=smt_summary,
        reference_pair=reference_pair,
        entry_zone=entry_zone,
        entry_frame=entry_frame,
        entry_mitigation=entry_mitigation,
        entry_score=entry_score,
        entry_summary=entry_summary,
    )


def score_shadow_context(context: ShadowFeatureContext) -> ShadowScoreBreakdown:
    fvg_alignment = 0
    if context.fvg_zone is not None:
        fvg_alignment = _zone_proximity_score(context.current_price, context.fvg_zone)

    order_block_alignment = 0
    if context.order_block_zone is not None:
        order_block_alignment = _zone_proximity_score(context.current_price, context.order_block_zone)

    mitigation_alignment = max(
        _mitigation_score(context.side, context.fvg_mitigation),
        _mitigation_score(context.side, context.order_block_mitigation),
    )

    smt_alignment = 0
    if context.smt is not None and context.smt.direction == _side_direction(context.side):
        smt_alignment = min(8, 4 + int(round(max(0.0, min(100.0, context.smt.strength)) / 20.0)))

    total = max(0, min(30, fvg_alignment + order_block_alignment + mitigation_alignment + smt_alignment))
    return ShadowScoreBreakdown(
        fvg_alignment=fvg_alignment,
        order_block_alignment=order_block_alignment,
        mitigation_alignment=mitigation_alignment,
        smt_alignment=smt_alignment,
        total=total,
    )
