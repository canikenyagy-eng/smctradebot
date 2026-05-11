from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalCandidate:
    pair: str
    signal: object
    frame: pd.DataFrame


@dataclass(frozen=True)
class CorrelationDrop:
    pair: str
    kept_pair: str
    correlation: float
    reason: str


class CorrelationCap:
    def __init__(self, threshold: float = 0.82, lookback: int = 120) -> None:
        self.threshold = abs(threshold)
        self.lookback = max(30, lookback)

    @staticmethod
    def _returns(frame: pd.DataFrame, lookback: int) -> pd.Series:
        series = frame["close"].astype(float).pct_change().dropna().tail(lookback)
        series.name = "returns"
        return series

    def _pair_corr(self, left: pd.DataFrame, right: pd.DataFrame) -> float:
        lhs = self._returns(left, self.lookback)
        rhs = self._returns(right, self.lookback)
        joined = pd.concat([lhs, rhs], axis=1, join="inner").dropna()
        if len(joined) < 20:
            return 0.0
        corr = joined.corr().iloc[0, 1]
        if pd.isna(corr):
            return 0.0
        return float(corr)

    @staticmethod
    def _better_candidate(left: SignalCandidate, right: SignalCandidate) -> SignalCandidate:
        left_key = (
            left.signal.score,
            left.signal.score_breakdown.shadow_bonus,
            left.signal.score_breakdown.regime_alignment,
            left.signal.score_breakdown.trigger_confirmation,
            left.signal.score_breakdown.htf_alignment,
        )
        right_key = (
            right.signal.score,
            right.signal.score_breakdown.shadow_bonus,
            right.signal.score_breakdown.regime_alignment,
            right.signal.score_breakdown.trigger_confirmation,
            right.signal.score_breakdown.htf_alignment,
        )
        return left if left_key >= right_key else right

    def filter(
        self,
        candidates: list[SignalCandidate],
    ) -> tuple[list[SignalCandidate], list[CorrelationDrop]]:
        if len(candidates) <= 1:
            return candidates, []

        parent = list(range(len(candidates)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        correlations: dict[tuple[int, int], float] = {}
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                corr = self._pair_corr(candidates[i].frame, candidates[j].frame)
                correlations[(i, j)] = corr
                if abs(corr) >= self.threshold:
                    union(i, j)

        clusters: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(candidates)):
            clusters[find(idx)].append(idx)

        kept: list[SignalCandidate] = []
        dropped: list[CorrelationDrop] = []

        for cluster_indices in clusters.values():
            if len(cluster_indices) == 1:
                kept.append(candidates[cluster_indices[0]])
                continue

            cluster_candidates = [candidates[idx] for idx in cluster_indices]
            best = cluster_candidates[0]
            for candidate in cluster_candidates[1:]:
                best = self._better_candidate(best, candidate)

            kept.append(best)

            for idx in cluster_indices:
                candidate = candidates[idx]
                if candidate is best:
                    continue

                corr = 0.0
                for other_idx in cluster_indices:
                    if other_idx == idx:
                        continue
                    key = (min(idx, other_idx), max(idx, other_idx))
                    corr = max(corr, abs(correlations.get(key, 0.0)))

                dropped.append(
                    CorrelationDrop(
                        pair=candidate.pair,
                        kept_pair=best.pair,
                        correlation=round(corr, 4),
                        reason="correlated exposure cap",
                    )
                )

        kept.sort(key=lambda item: item.signal.score, reverse=True)
        return kept, dropped
