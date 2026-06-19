from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from smc.structure import detect_bos_choch, identify_swings, infer_trend


def pip_size(pair: str) -> float:
    clean = pair.upper().replace("/", "")
    return 0.01 if clean.endswith("JPY") else 0.0001


def _clean_pair(pair: str) -> str:
    return pair.upper().replace("/", "").strip()


def _clean_regime(regime: str | None) -> str:
    return (regime or "UNKNOWN").upper().strip()


def _clean_items(values: tuple[str, ...] | list[str] | None, *, pair_mode: bool = False) -> tuple[str, ...]:
    if not values:
        return ()
    cleaned: list[str] = []
    for value in values:
        text = _clean_pair(str(value)) if pair_mode else str(value).upper().strip()
        if text:
            cleaned.append(text)
    return tuple(dict.fromkeys(cleaned))


@dataclass(frozen=True)
class StructureQualitySettings:
    enabled: bool = False
    scan_bars: int = 300
    min_break_pips: float = 2.0
    level_bucket_pips: float = 2.0
    min_score_for_bonus: float = 60.0
    max_bonus: int = 8
    backtest_only: bool = True
    allow_live: bool = False
    allowed_regimes: tuple[str, ...] = ()
    allowed_pairs: tuple[str, ...] = ()
    excluded_pairs: tuple[str, ...] = ()

    def sanitized(self) -> "StructureQualitySettings":
        return StructureQualitySettings(
            enabled=bool(self.enabled),
            scan_bars=max(80, int(self.scan_bars)),
            min_break_pips=max(0.0, float(self.min_break_pips)),
            level_bucket_pips=max(0.1, float(self.level_bucket_pips)),
            min_score_for_bonus=max(0.0, min(100.0, float(self.min_score_for_bonus))),
            max_bonus=max(0, min(20, int(self.max_bonus))),
            backtest_only=bool(self.backtest_only),
            allow_live=bool(self.allow_live),
            allowed_regimes=_clean_items(self.allowed_regimes),
            allowed_pairs=_clean_items(self.allowed_pairs, pair_mode=True),
            excluded_pairs=_clean_items(self.excluded_pairs, pair_mode=True),
        )

    def is_active(self, runtime_mode: str) -> bool:
        if not self.enabled:
            return False
        if not self.backtest_only:
            return True
        if runtime_mode.lower().strip() == "backtest":
            return True
        return self.allow_live

    def bonus_allowed(self, *, pair: str, regime_label: str | None) -> tuple[bool, str]:
        clean_pair = _clean_pair(pair)
        clean_regime = _clean_regime(regime_label)
        if self.excluded_pairs and clean_pair in self.excluded_pairs:
            return False, f"pair {clean_pair} excluded"
        if self.allowed_pairs and clean_pair not in self.allowed_pairs:
            return False, f"pair {clean_pair} not in allowed pairs"
        if self.allowed_regimes and clean_regime not in self.allowed_regimes:
            return False, f"regime {clean_regime} not in allowed regimes"
        return True, "allowed"


@dataclass(frozen=True)
class StructureQualityResult:
    enabled: bool
    score: float
    bonus: int
    event_current: str
    direction_current: str
    event_aligns_side: bool
    break_distance_pips: float
    duplicate_level: bool
    recent_event_count: int
    repeated_level_count: int
    summary: str
    bonus_allowed: bool = True
    bonus_condition: str = "allowed"

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "score": round(float(self.score), 4),
            "bonus": int(self.bonus),
            "event_current": self.event_current,
            "direction_current": self.direction_current,
            "event_aligns_side": bool(self.event_aligns_side),
            "break_distance_pips": round(float(self.break_distance_pips), 4),
            "duplicate_level": bool(self.duplicate_level),
            "recent_event_count": int(self.recent_event_count),
            "repeated_level_count": int(self.repeated_level_count),
            "summary": self.summary,
            "bonus_allowed": bool(self.bonus_allowed),
            "bonus_condition": self.bonus_condition,
        }


_DISABLED_RESULT = StructureQualityResult(
    enabled=False,
    score=0.0,
    bonus=0,
    event_current="none",
    direction_current="neutral",
    event_aligns_side=False,
    break_distance_pips=0.0,
    duplicate_level=False,
    recent_event_count=0,
    repeated_level_count=0,
    summary="structure quality disabled",
    bonus_allowed=False,
    bonus_condition="inactive",
)


def side_direction(side: str) -> str:
    return "bullish" if side.upper() == "BUY" else "bearish"


def level_bucket(pair: str, level: float | None, bucket_pips: float) -> int | None:
    if level is None:
        return None
    return round(float(level) / max(pip_size(pair) * bucket_pips, 1e-9))


def distance_pips(pair: str, left: float, right: float) -> float:
    return abs(float(left) - float(right)) / pip_size(pair)


def score_linear(value: float, *, good: float, bad: float, invert: bool = False) -> float:
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


def structure_events(
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
    swings = identify_swings(scoped, window=swing_window)
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    repeated: set[tuple[str, str, int | None]] = set()

    for local_idx in range(max(20, swing_window * 4), len(scoped)):
        history = swings.iloc[: local_idx + 1]
        recent_highs = history.loc[history["swing_high"], "high"]
        recent_lows = history.loc[history["swing_low"], "low"]
        if recent_highs.empty or recent_lows.empty:
            continue

        last_high = float(recent_highs.iloc[-1])
        last_low = float(recent_lows.iloc[-1])
        close = float(scoped["close"].iloc[local_idx])
        if close > last_high:
            direction = "bullish"
            level = last_high
        elif close < last_low:
            direction = "bearish"
            level = last_low
        else:
            continue

        trend = infer_trend(history)
        event = "BOS" if trend in {direction, "neutral"} else "CHoCH"
        bucket = level_bucket(pair, level, bucket_pips)
        key = (event, direction, bucket)
        if key in seen:
            repeated.add(key)
        seen.add(key)
        distance = distance_pips(pair, close, level)
        events.append(
            {
                "event": event,
                "direction": direction,
                "index": offset + local_idx,
                "level": level,
                "level_bucket": bucket,
                "break_distance_pips": distance,
                "repeated": key in repeated,
            }
        )

    return events, repeated


def bonus_from_score(score: float, settings: StructureQualitySettings) -> int:
    cfg = settings.sanitized()
    if cfg.max_bonus <= 0:
        return 0
    threshold = cfg.min_score_for_bonus
    if score < threshold:
        return 0
    if threshold >= 100.0:
        return cfg.max_bonus if score >= 100.0 else 0
    ratio = (score - threshold) / (100.0 - threshold)
    return max(0, min(cfg.max_bonus, int(round(cfg.max_bonus * ratio))))


def evaluate_structure_quality(
    *,
    pair: str,
    side: str,
    frame: pd.DataFrame,
    structure_event: str,
    swing_window: int = 3,
    settings: StructureQualitySettings | None = None,
    runtime_mode: str = "backtest",
    regime_label: str | None = None,
) -> StructureQualityResult:
    cfg = (settings or StructureQualitySettings()).sanitized()
    if not cfg.is_active(runtime_mode) or frame.empty:
        return _DISABLED_RESULT

    clean = frame[["open", "high", "low", "close", "volume"]].copy() if "volume" in frame.columns else frame.copy()
    current = detect_bos_choch(clean, window=max(2, swing_window))
    direction = side_direction(side)
    level = current.last_swing_high if direction == "bullish" else current.last_swing_low
    close = float(clean["close"].iloc[-1])
    break_distance = distance_pips(pair, close, float(level)) if level is not None else 0.0
    events, repeated = structure_events(
        clean,
        pair=pair,
        scan_bars=cfg.scan_bars,
        swing_window=max(2, swing_window),
        bucket_pips=cfg.level_bucket_pips,
    )
    bucket = level_bucket(pair, level, cfg.level_bucket_pips)
    key = (current.event or structure_event or "", direction, bucket)
    duplicate = key in repeated
    event_aligns = current.event is not None and current.direction == direction
    distance_score = score_linear(
        break_distance,
        good=cfg.min_break_pips * 3.0,
        bad=0.0,
    )
    score = (
        (25.0 if event_aligns else 0.0)
        + min(35.0, distance_score * 0.35)
        + (30.0 if not duplicate else 0.0)
        + (10.0 if current.trend in {direction, "neutral"} else 0.0)
    )
    score = max(0.0, min(100.0, score))
    bonus_allowed, bonus_condition = cfg.bonus_allowed(pair=pair, regime_label=regime_label)
    bonus = bonus_from_score(score, cfg) if bonus_allowed else 0
    event_current = current.event or "none"
    direction_current = current.direction or "neutral"
    summary = (
        f"structure_quality score={score:.1f} bonus={bonus} event={event_current.upper()} "
        f"direction={direction_current.upper()} break={break_distance:.2f}p duplicate={duplicate} "
        f"condition={bonus_condition}"
    )
    return StructureQualityResult(
        enabled=True,
        score=round(score, 2),
        bonus=bonus,
        event_current=event_current,
        direction_current=direction_current,
        event_aligns_side=bool(event_aligns),
        break_distance_pips=round(break_distance, 4),
        duplicate_level=bool(duplicate),
        recent_event_count=len(events),
        repeated_level_count=len(repeated),
        summary=summary,
        bonus_allowed=bonus_allowed,
        bonus_condition=bonus_condition,
    )
