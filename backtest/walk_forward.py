from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable
import json

import pandas as pd

from backtest.engine import BacktestEngine, BacktestPairReport, BacktestRunResult
from backtest.execution import build_rng


@dataclass(frozen=True)
class WalkForwardWindowResult:
    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train: BacktestRunResult
    test: BacktestRunResult

    def to_dict(self) -> dict[str, object]:
        return {
            "window_index": self.window_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_metrics": self.train.overall_metrics(),
            "test_metrics": self.test.overall_metrics(),
            "train_pairs": self.train.pair_rows(),
            "test_pairs": self.test.pair_rows(),
        }


@dataclass(frozen=True)
class WalkForwardResult:
    windows: list[WalkForwardWindowResult]
    summary: dict[str, object]
    parameters: dict[str, object]
    started_at: datetime
    finished_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "parameters": self.parameters,
            "summary": self.summary,
            "windows": [window.to_dict() for window in self.windows],
        }

    def export(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return target


class WalkForwardRunner:
    def __init__(
        self,
        *,
        engine: BacktestEngine,
        pairs: Iterable[str],
        train_months: int = 6,
        test_months: int = 1,
        step_months: int = 1,
        timeframe_config: dict[str, str] | None = None,
    ) -> None:
        self.engine = engine
        self.pairs = [pair.upper().replace("/", "") for pair in pairs]
        self.train_months = max(1, int(train_months))
        self.test_months = max(1, int(test_months))
        self.step_months = max(1, int(step_months))
        self.timeframe_config = timeframe_config or {
            "ltf": self.engine.signal_engine.ltf_timeframe,
            "htf": self.engine.signal_engine.htf_timeframe,
            "trigger": self.engine.signal_engine.trigger_timeframe,
        }

    def _reset_release_state(self) -> None:
        signal_engine = self.engine.signal_engine
        if hasattr(signal_engine, "reset_release_state"):
            signal_engine.reset_release_state()

    @staticmethod
    def _slice_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return frame[(frame.index >= start) & (frame.index < end)].copy()

    def _load_frames(self) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], dict[str, str]]:
        pair_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        errors: dict[str, str] = {}
        for pair in self.pairs:
            try:
                pair_frames[pair] = self.engine.load_pair_frames(pair)
            except Exception as exc:
                errors[pair] = str(exc)
        return pair_frames, errors

    @staticmethod
    def _window_bounds(
        pair_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []

        for ltf, htf, trigger in pair_frames.values():
            if ltf.empty or htf.empty or trigger.empty:
                continue
            starts.append(max(ltf.index.min(), htf.index.min(), trigger.index.min()))
            ends.append(min(ltf.index.max(), htf.index.max(), trigger.index.max()))

        if not starts or not ends:
            return None

        start = max(starts)
        end = min(ends)
        if start >= end:
            return None
        return start, end

    def _run_segment(
        self,
        pair_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        mode: str,
    ) -> BacktestRunResult:
        self._reset_release_state()
        reports: list[BacktestPairReport] = []
        segment_pairs = sorted(pair_frames.keys())
        universe = set(segment_pairs)

        started_at = datetime.now(timezone.utc)
        for pair in segment_pairs:
            ltf, htf, trigger = pair_frames[pair]
            ltf_slice = self._slice_frame(ltf, start, end)
            htf_slice = self._slice_frame(htf, start, end)
            trigger_slice = self._slice_frame(trigger, start, end)

            reference_pair = self.engine.signal_engine._resolve_smt_reference_pair(pair, universe)
            reference_trigger = None
            if reference_pair is not None and reference_pair in pair_frames:
                reference_trigger = self._slice_frame(pair_frames[reference_pair][2], start, end)

            report = self.engine.run_pair_from_frames(
                pair,
                ltf_slice,
                htf_slice,
                trigger_slice,
                reference_pair=reference_pair,
                reference_trigger=reference_trigger,
            )
            reports.append(report)
        finished_at = datetime.now(timezone.utc)

        return BacktestRunResult(
            pair_reports=reports,
            parameters={
                "mode": mode,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "ltf_timeframe": self.timeframe_config.get("ltf"),
                "htf_timeframe": self.timeframe_config.get("htf"),
                "trigger_timeframe": self.timeframe_config.get("trigger"),
                "train_months": self.train_months,
                "test_months": self.test_months,
                "step_months": self.step_months,
            },
            started_at=started_at,
            finished_at=finished_at,
            news_mode=self.engine.news_feed.__class__.__name__,
        )

    @staticmethod
    def _aggregate_summary(windows: list[WalkForwardWindowResult]) -> dict[str, object]:
        if not windows:
            return {
                "window_count": 0,
                "train_avg_win_rate": 0.0,
                "test_avg_win_rate": 0.0,
                "train_avg_profit_factor": 0.0,
                "test_avg_profit_factor": 0.0,
                "train_avg_r": 0.0,
                "test_avg_r": 0.0,
                "train_avg_drawdown_r": 0.0,
                "test_avg_drawdown_r": 0.0,
                "train_total_trades": 0,
                "test_total_trades": 0,
            }

        train_metrics = [window.train.overall_metrics() for window in windows]
        test_metrics = [window.test.overall_metrics() for window in windows]

        def avg(rows: list[dict[str, object]], key: str) -> float:
            values = [float(row.get(key, 0.0)) for row in rows]
            return round(mean(values), 6) if values else 0.0

        def total(rows: list[dict[str, object]], key: str) -> int:
            return int(sum(int(row.get(key, 0)) for row in rows))

        return {
            "window_count": len(windows),
            "train_avg_win_rate": avg(train_metrics, "win_rate"),
            "test_avg_win_rate": avg(test_metrics, "win_rate"),
            "train_avg_profit_factor": avg(train_metrics, "profit_factor"),
            "test_avg_profit_factor": avg(test_metrics, "profit_factor"),
            "train_avg_r": avg(train_metrics, "avg_r"),
            "test_avg_r": avg(test_metrics, "avg_r"),
            "train_avg_drawdown_r": avg(train_metrics, "max_drawdown_r"),
            "test_avg_drawdown_r": avg(test_metrics, "max_drawdown_r"),
            "train_total_trades": total(train_metrics, "trades"),
            "test_total_trades": total(test_metrics, "trades"),
        }

    def run(self) -> WalkForwardResult:
        started_at = datetime.now(timezone.utc)
        if hasattr(self.engine, "execution_settings") and hasattr(self.engine, "_rng"):
            self.engine._rng = build_rng(self.engine.execution_settings.random_seed)
        pair_frames, _ = self._load_frames()
        bounds = self._window_bounds(pair_frames)
        if bounds is None:
            finished_at = datetime.now(timezone.utc)
            return WalkForwardResult(
                windows=[],
                summary=self._aggregate_summary([]),
                parameters={
                    "pairs": self.pairs,
                    "train_months": self.train_months,
                    "test_months": self.test_months,
                    "step_months": self.step_months,
                    "ltf_timeframe": self.timeframe_config.get("ltf"),
                    "htf_timeframe": self.timeframe_config.get("htf"),
                    "trigger_timeframe": self.timeframe_config.get("trigger"),
                },
                started_at=started_at,
                finished_at=finished_at,
            )

        global_start, global_end = bounds
        windows: list[WalkForwardWindowResult] = []
        cursor = global_start
        window_index = 1

        while True:
            train_end = cursor + pd.DateOffset(months=self.train_months)
            test_end = train_end + pd.DateOffset(months=self.test_months)
            if test_end > global_end:
                break

            train_result = self._run_segment(
                pair_frames,
                start=cursor,
                end=train_end,
                mode="train",
            )
            test_result = self._run_segment(
                pair_frames,
                start=train_end,
                end=test_end,
                mode="test",
            )
            windows.append(
                WalkForwardWindowResult(
                    window_index=window_index,
                    train_start=cursor.to_pydatetime(),
                    train_end=train_end.to_pydatetime(),
                    test_start=train_end.to_pydatetime(),
                    test_end=test_end.to_pydatetime(),
                    train=train_result,
                    test=test_result,
                )
            )

            window_index += 1
            cursor = cursor + pd.DateOffset(months=self.step_months)
            if cursor >= global_end:
                break

        finished_at = datetime.now(timezone.utc)
        return WalkForwardResult(
            windows=windows,
            summary=self._aggregate_summary(windows),
            parameters={
                "pairs": self.pairs,
                "train_months": self.train_months,
                "test_months": self.test_months,
                "step_months": self.step_months,
                "ltf_timeframe": self.timeframe_config.get("ltf"),
                "htf_timeframe": self.timeframe_config.get("htf"),
                "trigger_timeframe": self.timeframe_config.get("trigger"),
                "global_start": global_start.isoformat(),
                "global_end": global_end.isoformat(),
            },
            started_at=started_at,
            finished_at=finished_at,
        )
