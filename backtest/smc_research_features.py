from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

import pandas as pd

from smc.fvg import detect_fvg_zones
from smc.order_block import detect_order_blocks
from smc.structure import detect_bos_choch
from smc.structure_quality import StructureQualitySettings, evaluate_structure_quality
from smc.zones import PriceZone, assess_zone_lifecycle


def pip_size(pair: str) -> float:
    clean = pair.upper().replace("/", "")
    return 0.01 if clean.endswith("JPY") else 0.0001


@dataclass(frozen=True)
class SMCResearchFeatureSettings:
    enabled: bool = False
    structure_scan_bars: int = 300
    structure_min_break_pips: float = 2.0
    structure_level_bucket_pips: float = 2.0
    ob_lookback_bars: int = 300
    ob_max_age_bars: int = 300
    ob_max_width_pips: float = 20.0
    ob_max_distance_pips: float = 30.0
    relaxed_fvg_lookback_bars: int = 300
    relaxed_fvg_min_gap_pips: float = 0.1
    relaxed_fvg_max_distance_pips: float = 30.0

    def sanitized(self) -> "SMCResearchFeatureSettings":
        return SMCResearchFeatureSettings(
            enabled=bool(self.enabled),
            structure_scan_bars=max(80, int(self.structure_scan_bars)),
            structure_min_break_pips=max(0.0, float(self.structure_min_break_pips)),
            structure_level_bucket_pips=max(0.1, float(self.structure_level_bucket_pips)),
            ob_lookback_bars=max(50, int(self.ob_lookback_bars)),
            ob_max_age_bars=max(1, int(self.ob_max_age_bars)),
            ob_max_width_pips=max(0.1, float(self.ob_max_width_pips)),
            ob_max_distance_pips=max(0.1, float(self.ob_max_distance_pips)),
            relaxed_fvg_lookback_bars=max(50, int(self.relaxed_fvg_lookback_bars)),
            relaxed_fvg_min_gap_pips=max(0.0, float(self.relaxed_fvg_min_gap_pips)),
            relaxed_fvg_max_distance_pips=max(0.1, float(self.relaxed_fvg_max_distance_pips)),
        )


def _side_direction(side: str) -> str:
    return "bullish" if side.upper() == "BUY" else "bearish"


def _level_bucket(pair: str, level: float | None, bucket_pips: float) -> int | None:
    if level is None:
        return None
    return round(float(level) / max(pip_size(pair) * bucket_pips, 1e-9))


def _distance_pips(pair: str, left: float, right: float) -> float:
    return abs(float(left) - float(right)) / pip_size(pair)


def _zone_distance_pips(pair: str, zone: PriceZone, price: float) -> float:
    if zone.contains(price):
        return 0.0
    if price < zone.lower:
        return _distance_pips(pair, price, zone.lower)
    return _distance_pips(pair, price, zone.upper)


def _score_linear(value: float, *, good: float, bad: float, invert: bool = False) -> float:
    if bad == good:
        return 100.0
    if invert:
        if value <= good:
            return 100.0
        if value >= bad:
            return 0.0
        return 100.0 * (bad - value) / (bad - good)
    if value >= good:
        return 100.0
    if value <= bad:
        return 0.0
    return 100.0 * (value - bad) / (good - bad)


def _structure_events(
    frame: pd.DataFrame,
    *,
    pair: str,
    scan_bars: int,
    swing_window: int,
    bucket_pips: float,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, int | None]]]:
    scoped = frame.tail(scan_bars).copy()
    if len(scoped) < max(40, swing_window * 8):
        return [], set()

    offset = len(frame) - len(scoped)
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    repeated: set[tuple[str, str, int | None]] = set()

    for local_idx in range(max(20, swing_window * 4), len(scoped)):
        state = detect_bos_choch(scoped.iloc[: local_idx + 1], window=swing_window)
        if state.event is None or state.direction is None:
            continue
        level = state.last_swing_high if state.direction == "bullish" else state.last_swing_low
        level_bucket = _level_bucket(pair, level, bucket_pips)
        key = (state.event, state.direction, level_bucket)
        if key in seen:
            repeated.add(key)
        seen.add(key)
        close = float(scoped["close"].iloc[local_idx])
        distance = _distance_pips(pair, close, float(level)) if level is not None else None
        events.append(
            {
                "event": state.event,
                "direction": state.direction,
                "index": offset + local_idx,
                "level": level,
                "level_bucket": level_bucket,
                "break_distance_pips": distance,
                "repeated": key in repeated,
            }
        )

    return events, repeated


def _structure_strict_features(
    *,
    pair: str,
    side: str,
    frame: pd.DataFrame,
    structure_event: str,
    swing_window: int,
    settings: SMCResearchFeatureSettings,
) -> dict[str, Any]:
    result = evaluate_structure_quality(
        pair=pair,
        side=side,
        frame=frame,
        structure_event=structure_event,
        swing_window=swing_window,
        settings=StructureQualitySettings(
            enabled=True,
            scan_bars=settings.structure_scan_bars,
            min_break_pips=settings.structure_min_break_pips,
            level_bucket_pips=settings.structure_level_bucket_pips,
            min_score_for_bonus=60.0,
            max_bonus=8,
            backtest_only=False,
            allow_live=True,
        ),
        runtime_mode="backtest",
    )
    return {
        "structure_strict_score": round(float(result.score), 2),
        "structure_quality_bonus": int(result.bonus),
        "structure_event_current": result.event_current,
        "structure_direction_current": result.direction_current,
        "structure_event_aligns_side": bool(result.event_aligns_side),
        "structure_break_distance_pips": round(float(result.break_distance_pips), 4),
        "structure_duplicate_level": bool(result.duplicate_level),
        "structure_recent_event_count": int(result.recent_event_count),
        "structure_repeated_level_count": int(result.repeated_level_count),
    }


def _fresh_ob_features(
    *,
    pair: str,
    side: str,
    entry: float,
    frame: pd.DataFrame,
    settings: SMCResearchFeatureSettings,
) -> dict[str, Any]:
    zones = detect_order_blocks(frame, lookback=min(len(frame), settings.ob_lookback_bars))
    direction = _side_direction(side)
    aligned = [zone for zone in zones if zone.direction == direction]
    if not aligned:
        return {
            "fresh_ob_score": 0.0,
            "fresh_ob_found": False,
            "fresh_ob_aligned_count": 0,
            "fresh_ob_age_bars": None,
            "fresh_ob_width_pips": None,
            "fresh_ob_strength": None,
            "fresh_ob_distance_pips": None,
            "fresh_ob_is_fresh": False,
        }

    def candidate_score(zone: PriceZone) -> float:
        age = max(0, len(frame) - 1 - int(zone.created_index or 0))
        width = zone.width / pip_size(pair)
        distance = _zone_distance_pips(pair, zone, entry)
        strength = max(0.0, min(100.0, zone.strength * 100.0))
        width_score = _score_linear(width, good=3.0, bad=settings.ob_max_width_pips, invert=True)
        age_score = _score_linear(age, good=0.0, bad=settings.ob_max_age_bars, invert=True)
        distance_score = _score_linear(distance, good=0.0, bad=settings.ob_max_distance_pips, invert=True)
        freshness_score = 100.0 if zone.is_fresh else max(0.0, 100.0 * (1.0 - zone.fill_ratio))
        return (
            strength * 0.30
            + width_score * 0.20
            + age_score * 0.20
            + distance_score * 0.20
            + freshness_score * 0.10
        )

    best = max(aligned, key=candidate_score)
    age = max(0, len(frame) - 1 - int(best.created_index or 0))
    width = best.width / pip_size(pair)
    distance = _zone_distance_pips(pair, best, entry)
    return {
        "fresh_ob_score": round(candidate_score(best), 2),
        "fresh_ob_found": True,
        "fresh_ob_aligned_count": len(aligned),
        "fresh_ob_age_bars": int(age),
        "fresh_ob_width_pips": round(width, 4),
        "fresh_ob_strength": round(best.strength * 100.0, 2),
        "fresh_ob_distance_pips": round(distance, 4),
        "fresh_ob_is_fresh": bool(best.is_fresh),
        "fresh_ob_fill_ratio": round(best.fill_ratio, 4),
    }


def detect_relaxed_fvg_zones(
    frame: pd.DataFrame,
    *,
    pair: str,
    lookback: int = 300,
    min_gap_pips: float = 0.1,
) -> list[PriceZone]:
    if frame.empty or len(frame) < 3:
        return []

    start = max(1, len(frame) - max(3, lookback) - 1)
    min_gap = min_gap_pips * pip_size(pair)
    zones: list[PriceZone] = []

    for mid_idx in range(start, len(frame) - 1):
        prev_idx = mid_idx - 1
        next_idx = mid_idx + 1
        prev = frame.iloc[prev_idx]
        mid = frame.iloc[mid_idx]
        nxt = frame.iloc[next_idx]
        prev_high = float(prev["high"])
        prev_low = float(prev["low"])
        next_high = float(nxt["high"])
        next_low = float(nxt["low"])
        mid_open = float(mid["open"])
        mid_close = float(mid["close"])
        body = abs(mid_close - mid_open)
        avg_range = float((frame["high"].iloc[max(0, mid_idx - 20) : mid_idx] - frame["low"].iloc[max(0, mid_idx - 20) : mid_idx]).mean() or 0.0)

        if next_low > prev_high and (next_low - prev_high) >= min_gap:
            strength = min(1.0, ((next_low - prev_high) / max(avg_range, min_gap, 1e-9)) * 0.65 + (body / max(avg_range, 1e-9)) * 0.35)
            zone = PriceZone(
                kind="relaxed_fvg",
                direction="bullish",
                lower=prev_high,
                upper=next_low,
                created_at=frame.index[next_idx].to_pydatetime(),
                created_index=next_idx,
                source_index=mid_idx,
                strength=strength,
            )
            zones.append(assess_zone_lifecycle(frame, zone, start_index=next_idx))

        if next_high < prev_low and (prev_low - next_high) >= min_gap:
            strength = min(1.0, ((prev_low - next_high) / max(avg_range, min_gap, 1e-9)) * 0.65 + (body / max(avg_range, 1e-9)) * 0.35)
            zone = PriceZone(
                kind="relaxed_fvg",
                direction="bearish",
                lower=next_high,
                upper=prev_low,
                created_at=frame.index[next_idx].to_pydatetime(),
                created_index=next_idx,
                source_index=mid_idx,
                strength=strength,
            )
            zones.append(assess_zone_lifecycle(frame, zone, start_index=next_idx))

    return sorted(zones, key=lambda item: (item.created_index or -1, item.strength), reverse=True)


def _relaxed_fvg_features(
    *,
    pair: str,
    side: str,
    entry: float,
    frame: pd.DataFrame,
    settings: SMCResearchFeatureSettings,
) -> dict[str, Any]:
    strict_zones = detect_fvg_zones(frame, lookback=min(len(frame), settings.relaxed_fvg_lookback_bars))
    relaxed_zones = detect_relaxed_fvg_zones(
        frame,
        pair=pair,
        lookback=min(len(frame), settings.relaxed_fvg_lookback_bars),
        min_gap_pips=settings.relaxed_fvg_min_gap_pips,
    )
    direction = _side_direction(side)
    aligned = [zone for zone in relaxed_zones if zone.direction == direction]
    reference_extra_count = max(0, len(relaxed_zones) - len(strict_zones))
    if not aligned:
        return {
            "relaxed_fvg_score": 0.0,
            "relaxed_fvg_count": len(relaxed_zones),
            "relaxed_fvg_aligned_count": 0,
            "relaxed_fvg_reference_extra_count": reference_extra_count,
            "relaxed_fvg_nearest_distance_pips": None,
            "relaxed_fvg_nearest_width_pips": None,
            "relaxed_fvg_nearest_age_bars": None,
            "relaxed_fvg_nearest_is_fresh": False,
        }

    def candidate_score(zone: PriceZone) -> float:
        age = max(0, len(frame) - 1 - int(zone.created_index or 0))
        width = zone.width / pip_size(pair)
        distance = _zone_distance_pips(pair, zone, entry)
        strength = max(0.0, min(100.0, zone.strength * 100.0))
        age_score = _score_linear(age, good=0.0, bad=settings.relaxed_fvg_lookback_bars, invert=True)
        distance_score = _score_linear(distance, good=0.0, bad=settings.relaxed_fvg_max_distance_pips, invert=True)
        freshness_score = 100.0 if zone.is_fresh else max(0.0, 100.0 * (1.0 - zone.fill_ratio))
        width_score = min(100.0, width * 20.0)
        return strength * 0.25 + distance_score * 0.30 + freshness_score * 0.20 + age_score * 0.15 + width_score * 0.10

    best = max(aligned, key=candidate_score)
    age = max(0, len(frame) - 1 - int(best.created_index or 0))
    width = best.width / pip_size(pair)
    distance = _zone_distance_pips(pair, best, entry)
    aligned_widths = [zone.width / pip_size(pair) for zone in aligned if zone.width > 0]
    return {
        "relaxed_fvg_score": round(candidate_score(best), 2),
        "relaxed_fvg_count": len(relaxed_zones),
        "relaxed_fvg_aligned_count": len(aligned),
        "relaxed_fvg_reference_extra_count": reference_extra_count,
        "relaxed_fvg_nearest_distance_pips": round(distance, 4),
        "relaxed_fvg_nearest_width_pips": round(width, 4),
        "relaxed_fvg_nearest_age_bars": int(age),
        "relaxed_fvg_nearest_is_fresh": bool(best.is_fresh),
        "relaxed_fvg_avg_aligned_width_pips": round(mean(aligned_widths), 4) if aligned_widths else None,
    }


def extract_smc_research_features(
    *,
    pair: str,
    side: str,
    entry: float,
    frame: pd.DataFrame,
    structure_event: str,
    swing_window: int = 3,
    settings: SMCResearchFeatureSettings | None = None,
) -> dict[str, Any]:
    cfg = (settings or SMCResearchFeatureSettings()).sanitized()
    if not cfg.enabled or frame.empty:
        return {}

    clean = frame[["open", "high", "low", "close", "volume"]].copy() if "volume" in frame.columns else frame.copy()
    output: dict[str, Any] = {"smc_research_features_enabled": True}
    output.update(
        _structure_strict_features(
            pair=pair,
            side=side,
            frame=clean,
            structure_event=structure_event,
            swing_window=max(2, swing_window),
            settings=cfg,
        )
    )
    output.update(_fresh_ob_features(pair=pair, side=side, entry=entry, frame=clean, settings=cfg))
    output.update(_relaxed_fvg_features(pair=pair, side=side, entry=entry, frame=clean, settings=cfg))
    return output
