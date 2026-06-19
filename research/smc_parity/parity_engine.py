from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from research.smc_parity.event_schema import SMCEvent, json_safe


def pip_size(pair: str) -> float:
    clean = pair.upper().replace("/", "")
    return 0.01 if clean.endswith("JPY") else 0.0001


@dataclass(frozen=True)
class ParitySettings:
    max_time_delta_bars: int = 5
    max_level_distance_pips: float = 5.0
    include_event_samples: bool = False
    max_event_samples: int = 10


@dataclass(frozen=True)
class EventMatch:
    event_type: str
    reference_index: int
    internal_index: int
    reference_direction: str
    internal_direction: str
    time_delta_bars: int
    level_distance_pips: float | None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def event_counts(events: list[SMCEvent]) -> dict[str, int]:
    return dict(Counter(event.event_type for event in events))


def live_safety_counts(events: list[SMCEvent]) -> dict[str, int]:
    return dict(Counter(event.live_safety for event in events))


def _level_distance_pips(left: SMCEvent, right: SMCEvent, pair: str) -> float | None:
    left_level = left.comparable_level
    right_level = right.comparable_level
    if left_level is None or right_level is None:
        return None
    return abs(left_level - right_level) / pip_size(pair)


def _match_type_events(
    *,
    event_type: str,
    internal_events: list[SMCEvent],
    reference_events: list[SMCEvent],
    pair: str,
    settings: ParitySettings,
) -> tuple[list[EventMatch], list[SMCEvent], list[SMCEvent]]:
    unmatched_internal = set(range(len(internal_events)))
    matches: list[EventMatch] = []
    missing_reference: list[SMCEvent] = []

    for ref in reference_events:
        candidates: list[tuple[float, int, int, float | None]] = []
        for internal_idx in list(unmatched_internal):
            item = internal_events[internal_idx]
            time_delta = abs(item.known_at_index - ref.known_at_index)
            if time_delta > settings.max_time_delta_bars:
                continue
            level_delta = _level_distance_pips(item, ref, pair)
            if level_delta is not None and level_delta > settings.max_level_distance_pips:
                continue
            direction_penalty = 0 if item.direction == ref.direction else 10_000
            level_score = level_delta if level_delta is not None else 0.0
            score = direction_penalty + time_delta * 10.0 + level_score
            candidates.append((score, internal_idx, time_delta, level_delta))

        if not candidates:
            missing_reference.append(ref)
            continue

        _, internal_idx, time_delta, level_delta = min(candidates, key=lambda row: row[0])
        matched = internal_events[internal_idx]
        unmatched_internal.remove(internal_idx)
        matches.append(
            EventMatch(
                event_type=event_type,
                reference_index=ref.index,
                internal_index=matched.index,
                reference_direction=ref.direction,
                internal_direction=matched.direction,
                time_delta_bars=time_delta,
                level_distance_pips=round(level_delta, 4) if level_delta is not None else None,
            )
        )

    extra_internal = [internal_events[idx] for idx in sorted(unmatched_internal)]
    return matches, missing_reference, extra_internal


def compare_event_sets(
    *,
    pair: str,
    timeframe: str,
    internal_events: list[SMCEvent],
    reference_events: list[SMCEvent],
    settings: ParitySettings | None = None,
) -> dict[str, Any]:
    cfg = settings or ParitySettings()
    event_types = sorted({event.event_type for event in internal_events} | {event.event_type for event in reference_events})
    by_internal: dict[str, list[SMCEvent]] = defaultdict(list)
    by_reference: dict[str, list[SMCEvent]] = defaultdict(list)
    for event in internal_events:
        by_internal[event.event_type].append(event)
    for event in reference_events:
        by_reference[event.event_type].append(event)

    modules: dict[str, Any] = {}
    all_matches: list[EventMatch] = []
    for event_type in event_types:
        matches, missing, extra = _match_type_events(
            event_type=event_type,
            internal_events=by_internal.get(event_type, []),
            reference_events=by_reference.get(event_type, []),
            pair=pair,
            settings=cfg,
        )
        all_matches.extend(matches)
        direction_agreement = (
            sum(1 for match in matches if match.internal_direction == match.reference_direction) / len(matches)
            if matches
            else 0.0
        )
        level_distances = [match.level_distance_pips for match in matches if match.level_distance_pips is not None]
        time_deltas = [match.time_delta_bars for match in matches]
        modules[event_type] = {
            "internal_events": len(by_internal.get(event_type, [])),
            "reference_events": len(by_reference.get(event_type, [])),
            "matched_events": len(matches),
            "missing_reference_events": len(missing),
            "extra_internal_events": len(extra),
            "direction_agreement": round(direction_agreement, 4),
            "avg_level_distance_pips": round(mean(level_distances), 4) if level_distances else None,
            "avg_time_delta_bars": round(mean(time_deltas), 4) if time_deltas else None,
            "match_rate_vs_reference": round(len(matches) / len(by_reference.get(event_type, [])), 4)
            if by_reference.get(event_type)
            else None,
            "match_rate_vs_internal": round(len(matches) / len(by_internal.get(event_type, [])), 4)
            if by_internal.get(event_type)
            else None,
        }
        if cfg.include_event_samples:
            modules[event_type]["matches"] = [match.to_dict() for match in matches[: cfg.max_event_samples]]
            modules[event_type]["missing_reference_samples"] = [
                event.to_dict() for event in missing[: cfg.max_event_samples]
            ]
            modules[event_type]["extra_internal_samples"] = [event.to_dict() for event in extra[: cfg.max_event_samples]]

    direction_matches = sum(1 for match in all_matches if match.internal_direction == match.reference_direction)
    return {
        "pair": pair.upper().replace("/", ""),
        "timeframe": timeframe.upper(),
        "settings": asdict(cfg),
        "internal_event_count": len(internal_events),
        "reference_event_count": len(reference_events),
        "matched_event_count": len(all_matches),
        "overall_direction_agreement": round(direction_matches / len(all_matches), 4) if all_matches else 0.0,
        "internal_counts": event_counts(internal_events),
        "reference_counts": event_counts(reference_events),
        "internal_live_safety": live_safety_counts(internal_events),
        "reference_live_safety": live_safety_counts(reference_events),
        "modules": modules,
    }


def internal_event_summary(*, pair: str, timeframe: str, events: list[SMCEvent]) -> dict[str, Any]:
    return {
        "pair": pair.upper().replace("/", ""),
        "timeframe": timeframe.upper(),
        "event_count": len(events),
        "counts": event_counts(events),
        "live_safety": live_safety_counts(events),
    }
