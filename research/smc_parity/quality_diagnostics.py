from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from research.smc_parity.event_schema import SMCEvent, json_safe
from research.smc_parity.parity_engine import ParitySettings, pip_size


@dataclass(frozen=True)
class StructureQualitySettings:
    min_break_distance_pips: float = 2.0
    level_bucket_pips: float = 2.0


@dataclass(frozen=True)
class OrderBlockQualitySettings:
    min_strength: float = 35.0
    max_width_pips: float = 20.0
    max_age_bars: int = 500


@dataclass(frozen=True)
class QualityDiagnosticSettings:
    structure: StructureQualitySettings = StructureQualitySettings()
    order_block: OrderBlockQualitySettings = OrderBlockQualitySettings()

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def _bucket_level(pair: str, level: float | None, bucket_pips: float) -> int | None:
    if level is None:
        return None
    size = max(pip_size(pair) * bucket_pips, 1e-9)
    return round(float(level) / size)


def _width_pips(event: SMCEvent, pair: str) -> float | None:
    if event.top is None or event.bottom is None:
        return None
    return abs(event.top - event.bottom) / pip_size(pair)


def _distance_pips(left: SMCEvent, right: SMCEvent, pair: str) -> float | None:
    left_level = left.comparable_level
    right_level = right.comparable_level
    if left_level is None or right_level is None:
        return None
    return abs(left_level - right_level) / pip_size(pair)


def _nearest_distance(
    target: SMCEvent,
    candidates: list[SMCEvent],
    *,
    pair: str,
    same_direction: bool = True,
) -> float | None:
    distances: list[float] = []
    for candidate in candidates:
        if same_direction and candidate.direction != target.direction:
            continue
        distance = _distance_pips(target, candidate, pair)
        if distance is not None:
            distances.append(distance)
    return min(distances) if distances else None


def structure_quality(
    events: list[SMCEvent],
    *,
    pair: str,
    settings: StructureQualitySettings,
) -> dict[str, Any]:
    structure_events = [event for event in events if event.event_type in {"BOS", "CHOCH"}]
    by_type = Counter(event.event_type for event in structure_events)
    repeated_by_type = Counter()
    strict_pass_by_type = Counter()
    micro_break_by_type = Counter()
    seen_levels: set[tuple[str, str, int | None]] = set()
    break_distances: dict[str, list[float]] = defaultdict(list)

    for event in structure_events:
        distance = event.metadata.get("break_distance_pips") if isinstance(event.metadata, dict) else None
        try:
            parsed_distance = float(distance)
        except (TypeError, ValueError):
            parsed_distance = None
        if parsed_distance is not None:
            break_distances[event.event_type].append(parsed_distance)
        level_bucket = _bucket_level(pair, event.level, settings.level_bucket_pips)
        key = (event.event_type, event.direction, level_bucket)
        repeated = key in seen_levels
        seen_levels.add(key)
        if repeated:
            repeated_by_type[event.event_type] += 1
        if parsed_distance is not None and parsed_distance >= settings.min_break_distance_pips and not repeated:
            strict_pass_by_type[event.event_type] += 1
        else:
            micro_break_by_type[event.event_type] += 1

    output: dict[str, Any] = {
        "total": len(structure_events),
        "by_type": dict(by_type),
        "repeated_breaks_by_type": dict(repeated_by_type),
        "strict_pass_by_type": dict(strict_pass_by_type),
        "micro_or_duplicate_by_type": dict(micro_break_by_type),
    }
    for event_type, values in sorted(break_distances.items()):
        output[f"{event_type.lower()}_avg_break_distance_pips"] = round(mean(values), 4) if values else None
    return output


def order_block_quality(
    events: list[SMCEvent],
    *,
    pair: str,
    frame_length: int,
    settings: OrderBlockQualitySettings,
) -> dict[str, Any]:
    ob_events = [event for event in events if event.event_type == "ORDER_BLOCK"]
    strength_pass = 0
    width_pass = 0
    age_pass = 0
    strict_pass = 0
    widths: list[float] = []
    strengths: list[float] = []
    ages: list[int] = []

    for event in ob_events:
        width = _width_pips(event, pair)
        strength = event.strength
        age = max(0, frame_length - 1 - event.index)
        if width is not None:
            widths.append(width)
        if strength is not None:
            strengths.append(strength)
        ages.append(age)
        width_ok = width is not None and width <= settings.max_width_pips
        strength_ok = strength is not None and strength >= settings.min_strength
        age_ok = age <= settings.max_age_bars
        strength_pass += int(strength_ok)
        width_pass += int(width_ok)
        age_pass += int(age_ok)
        strict_pass += int(width_ok and strength_ok and age_ok)

    total = len(ob_events)
    return {
        "total": total,
        "strength_pass": strength_pass,
        "width_pass": width_pass,
        "age_pass": age_pass,
        "strict_pass": strict_pass,
        "strict_pass_rate": round(strict_pass / total, 4) if total else 0.0,
        "avg_width_pips": round(mean(widths), 4) if widths else None,
        "avg_strength": round(mean(strengths), 4) if strengths else None,
        "avg_age_bars": round(mean(ages), 4) if ages else None,
    }


def _matched_reference_indices(
    *,
    pair: str,
    internal_events: list[SMCEvent],
    reference_events: list[SMCEvent],
    event_type: str,
    settings: ParitySettings,
) -> set[int]:
    internal = [event for event in internal_events if event.event_type == event_type]
    reference = [event for event in reference_events if event.event_type == event_type]
    unmatched_internal = set(range(len(internal)))
    matched_reference: set[int] = set()

    for ref_idx, ref in enumerate(reference):
        candidates: list[tuple[float, int]] = []
        for internal_idx in list(unmatched_internal):
            item = internal[internal_idx]
            time_delta = abs(item.known_at_index - ref.known_at_index)
            if time_delta > settings.max_time_delta_bars:
                continue
            level_delta = _distance_pips(item, ref, pair)
            if level_delta is not None and level_delta > settings.max_level_distance_pips:
                continue
            direction_penalty = 0 if item.direction == ref.direction else 10_000
            score = direction_penalty + time_delta * 10.0 + (level_delta if level_delta is not None else 0.0)
            candidates.append((score, internal_idx))
        if not candidates:
            continue
        _, internal_idx = min(candidates, key=lambda row: row[0])
        unmatched_internal.remove(internal_idx)
        matched_reference.add(ref_idx)
    return matched_reference


def fvg_reference_diagnostics(
    *,
    pair: str,
    internal_events: list[SMCEvent],
    reference_events: list[SMCEvent],
    parity_settings: ParitySettings,
) -> dict[str, Any]:
    internal_fvgs = [event for event in internal_events if event.event_type == "FVG"]
    reference_fvgs = [event for event in reference_events if event.event_type == "FVG"]
    matched_ref_indices = _matched_reference_indices(
        pair=pair,
        internal_events=internal_events,
        reference_events=reference_events,
        event_type="FVG",
        settings=parity_settings,
    )
    missed = [event for idx, event in enumerate(reference_fvgs) if idx not in matched_ref_indices]
    nearest_distances = [
        value
        for value in (_nearest_distance(event, internal_fvgs, pair=pair, same_direction=True) for event in missed)
        if value is not None
    ]
    widths = [width for width in (_width_pips(event, pair) for event in missed) if width is not None]
    mitigated = [event for event in missed if event.mitigated_index is not None]
    time_to_mitigation = [
        int(event.mitigated_index - event.index)
        for event in mitigated
        if event.mitigated_index is not None and event.mitigated_index >= event.index
    ]
    return {
        "internal_fvg_count": len(internal_fvgs),
        "reference_fvg_count": len(reference_fvgs),
        "reference_extra_fvg_count": len(missed),
        "reference_extra_fvg_rate": round(len(missed) / len(reference_fvgs), 4) if reference_fvgs else 0.0,
        "avg_missed_fvg_distance_to_internal_pips": round(mean(nearest_distances), 4) if nearest_distances else None,
        "avg_missed_fvg_width_pips": round(mean(widths), 4) if widths else None,
        "missed_fvg_mitigation_rate": round(len(mitigated) / len(missed), 4) if missed else 0.0,
        "avg_missed_fvg_time_to_mitigation_bars": round(mean(time_to_mitigation), 4)
        if time_to_mitigation
        else None,
    }


def build_quality_diagnostics(
    *,
    pair: str,
    frame_length: int,
    internal_events: list[SMCEvent],
    reference_events: list[SMCEvent],
    parity_settings: ParitySettings,
    settings: QualityDiagnosticSettings | None = None,
) -> dict[str, Any]:
    cfg = settings or QualityDiagnosticSettings()
    return {
        "settings": cfg.to_dict(),
        "internal_structure": structure_quality(internal_events, pair=pair, settings=cfg.structure),
        "reference_structure": structure_quality(reference_events, pair=pair, settings=cfg.structure),
        "internal_order_block": order_block_quality(
            internal_events,
            pair=pair,
            frame_length=frame_length,
            settings=cfg.order_block,
        ),
        "reference_order_block": order_block_quality(
            reference_events,
            pair=pair,
            frame_length=frame_length,
            settings=cfg.order_block,
        ),
        "fvg_reference": fvg_reference_diagnostics(
            pair=pair,
            internal_events=internal_events,
            reference_events=reference_events,
            parity_settings=parity_settings,
        ),
    }
