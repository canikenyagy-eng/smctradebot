from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from research.smc_parity.event_schema import json_safe


@dataclass(frozen=True)
class FeatureCandidate:
    name: str
    group: str
    live_safety: str
    intended_use: str
    signal_effect: str
    validation_mode: str
    overfit_risk: str
    implementation_priority: int
    status: str
    metrics_to_watch: list[str]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def default_feature_candidates() -> list[FeatureCandidate]:
    return [
        FeatureCandidate(
            name="previous_high_low_broken_flags",
            group="external_liquidity",
            live_safety="LIVE_SAFE",
            intended_use="Detect previous day/week/session liquidity breaks and sweeps.",
            signal_effect="Boost sweep-plus-displacement setups; penalize entries far from meaningful liquidity.",
            validation_mode="logging_only -> soft_score -> regime_specific_soft_score",
            overfit_risk="low",
            implementation_priority=1,
            status="planned",
            metrics_to_watch=["PF", "AvgR", "payoff_ratio", "trade_count", "OOS_avg_r"],
        ),
        FeatureCandidate(
            name="session_high_low_context",
            group="session_liquidity",
            live_safety="LIVE_SAFE",
            intended_use="Track Asian/London/NY highs and lows as liquidity magnets.",
            signal_effect="Improve London/NY sweep quality and reduce mid-range session noise.",
            validation_mode="logging_only -> session_score_component",
            overfit_risk="low_medium",
            implementation_priority=2,
            status="planned",
            metrics_to_watch=["PF_by_session", "AvgR_by_session", "DD", "acceptance_rate"],
        ),
        FeatureCandidate(
            name="fvg_live_mitigation_state",
            group="imbalance",
            live_safety="DELAYED_LIVE_SAFE",
            intended_use="Track active FVG age, width, distance, and mitigation depth incrementally.",
            signal_effect="Reward fresh efficient imbalance; penalize stale or deeply mitigated zones.",
            validation_mode="logging_only -> shadow_score",
            overfit_risk="medium",
            implementation_priority=3,
            status="planned",
            metrics_to_watch=["PF", "AvgR", "timeout_exit_rate", "payoff_ratio", "OOS_degradation"],
        ),
        FeatureCandidate(
            name="liquidity_pool_live_state",
            group="liquidity",
            live_safety="DELAYED_LIVE_SAFE",
            intended_use="Track equal highs/lows and mark sweeps only when the current candle confirms them.",
            signal_effect="Separate true liquidity sweep setups from random displacement.",
            validation_mode="logging_only -> liquidity_score_component",
            overfit_risk="medium",
            implementation_priority=4,
            status="planned",
            metrics_to_watch=["PF", "AvgR", "win_loss_contribution", "trade_count", "regime_interaction"],
        ),
        FeatureCandidate(
            name="current_retracement_depth",
            group="premium_discount",
            live_safety="DELAYED_LIVE_SAFE",
            intended_use="Measure current retracement from confirmed swing range.",
            signal_effect="Reduce bad entries too close to impulse extremes or after structural failure.",
            validation_mode="logging_only -> soft_score",
            overfit_risk="medium",
            implementation_priority=5,
            status="planned",
            metrics_to_watch=["AvgR", "avg_loss_r", "payoff_ratio", "DD"],
        ),
        FeatureCandidate(
            name="order_block_strength_percentage",
            group="order_block",
            live_safety="DELAYED_LIVE_SAFE",
            intended_use="Convert OB from binary confirmation into quality score.",
            signal_effect="Reward strong aligned OB zones, avoid weak/wide OB noise.",
            validation_mode="logging_only -> shadow_score_only",
            overfit_risk="medium_high",
            implementation_priority=6,
            status="planned",
            metrics_to_watch=["PF", "AvgR", "GBPUSD_delta", "false_positive_rate"],
        ),
    ]


def feature_candidate_report() -> dict[str, Any]:
    candidates = sorted(default_feature_candidates(), key=lambda item: item.implementation_priority)
    return {
        "candidate_count": len(candidates),
        "promotion_rule": (
            "Promote only if walk-forward and Monte Carlo improve PF/AvgR/OOS stability without excessive trade collapse."
        ),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
