from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from research.smc_parity.event_schema import SMCEvent, json_safe


@dataclass(frozen=True)
class LookaheadRule:
    feature: str
    classification: str
    risk_score: int
    reason: str
    live_safe_alternative: str
    suggested_usage: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def default_lookahead_rules() -> list[LookaheadRule]:
    return [
        LookaheadRule(
            feature="swing_high_low",
            classification="DELAYED_LIVE_SAFE",
            risk_score=65,
            reason="Confirmed swings require right-side candles; the pivot candle cannot be known at the pivot timestamp.",
            live_safe_alternative="Emit the swing only after the confirmation window has closed.",
            suggested_usage="Allowed in live only with confirmation_index; safe for research labels.",
        ),
        LookaheadRule(
            feature="bos_choch_from_confirmed_swings",
            classification="DELAYED_LIVE_SAFE",
            risk_score=55,
            reason="Structure breaks depend on previously confirmed swings; event timestamp must be the break candle, not the source swing.",
            live_safe_alternative="Use close/high-low break of already confirmed swing levels.",
            suggested_usage="Live-safe after the break candle closes.",
        ),
        LookaheadRule(
            feature="fvg_detection",
            classification="DELAYED_LIVE_SAFE",
            risk_score=35,
            reason="Classic three-candle FVG is known only after the third candle closes.",
            live_safe_alternative="Create FVG at the confirmation candle close, not on the middle candle.",
            suggested_usage="Safe for live if created_at equals the third candle close.",
        ),
        LookaheadRule(
            feature="fvg_mitigated_index",
            classification="RESEARCH_ONLY",
            risk_score=90,
            reason="Future mitigation index requires candles that are unavailable at entry time.",
            live_safe_alternative="Track active FVG zones and update mitigation state candle by candle.",
            suggested_usage="Use for diagnostics, labels, and validation, not direct live scoring.",
        ),
        LookaheadRule(
            feature="order_block_mitigated_index",
            classification="RESEARCH_ONLY",
            risk_score=90,
            reason="Future OB mitigation/invalidation can leak outcome information.",
            live_safe_alternative="Track active OB lifecycle incrementally in live.",
            suggested_usage="Research-only unless converted into live state.",
        ),
        LookaheadRule(
            feature="liquidity_swept_index",
            classification="RESEARCH_ONLY",
            risk_score=85,
            reason="A future swept index is only known after price reaches the pool.",
            live_safe_alternative="Maintain active liquidity pools and mark swept only on the sweep candle.",
            suggested_usage="Research labels first; live feature only as current sweep state.",
        ),
        LookaheadRule(
            feature="previous_high_low_broken",
            classification="LIVE_SAFE",
            risk_score=15,
            reason="Previous period high/low is known after that period closes; current break is observable.",
            live_safe_alternative="Use closed previous day/week/session levels only.",
            suggested_usage="Good candidate for live scoring and liquidity context.",
        ),
        LookaheadRule(
            feature="session_high_low_context",
            classification="LIVE_SAFE",
            risk_score=20,
            reason="Completed session high/low is known; active session extreme is safe if updated incrementally.",
            live_safe_alternative="Separate completed-session levels from active-session running levels.",
            suggested_usage="Good candidate for Forex session-aware scoring.",
        ),
        LookaheadRule(
            feature="deepest_retracement_percent",
            classification="RESEARCH_ONLY",
            risk_score=80,
            reason="Deepest retracement is known only after the swing leg completes.",
            live_safe_alternative="Use current retracement from confirmed swing range.",
            suggested_usage="Research labels only; live should use current_retracement_pct.",
        ),
        LookaheadRule(
            feature="ob_strength_percentage",
            classification="DELAYED_LIVE_SAFE",
            risk_score=45,
            reason="OB strength can be computed after the block is confirmed, but FX volume is usually proxy volume.",
            live_safe_alternative="Use strength as a soft score, never a hard veto until validated.",
            suggested_usage="Candidate for shadow score after ablation.",
        ),
    ]


def audit_events(events: list[SMCEvent]) -> dict[str, Any]:
    by_safety = Counter(event.live_safety for event in events)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    research_only_with_future_markers = 0
    events_with_future_markers = 0
    future_markers_not_research_only = 0
    delayed_without_confirmation = 0

    for event in events:
        by_type[event.event_type][event.live_safety] += 1
        has_future_marker = event.mitigated_index is not None or event.swept_index is not None
        if has_future_marker:
            events_with_future_markers += 1
            if event.live_safety == "RESEARCH_ONLY":
                research_only_with_future_markers += 1
            else:
                future_markers_not_research_only += 1
        if event.live_safety == "DELAYED_LIVE_SAFE" and event.confirmation_index is None:
            delayed_without_confirmation += 1

    return {
        "event_count": len(events),
        "live_safety_counts": dict(by_safety),
        "by_event_type": {event_type: dict(counter) for event_type, counter in sorted(by_type.items())},
        "events_with_future_markers": events_with_future_markers,
        "research_only_with_future_markers": research_only_with_future_markers,
        "future_markers_not_research_only": future_markers_not_research_only,
        "delayed_without_confirmation": delayed_without_confirmation,
        "audit_pass": delayed_without_confirmation == 0,
        "live_promotion_safe": delayed_without_confirmation == 0 and future_markers_not_research_only == 0,
    }


def build_lookahead_report(events_by_source: dict[str, list[SMCEvent]]) -> dict[str, Any]:
    return {
        "rules": [rule.to_dict() for rule in default_lookahead_rules()],
        "event_audit": {source: audit_events(events) for source, events in sorted(events_by_source.items())},
    }
