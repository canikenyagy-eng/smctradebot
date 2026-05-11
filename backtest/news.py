from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from execution.news import NewsAssessment


def _pair_currencies(pair: str) -> tuple[str, str]:
    clean = pair.upper().replace("/", "")
    if len(clean) < 6:
        return "", ""
    return clean[:3], clean[3:6]


def _parse_numeric(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None

    cleaned = str(value).strip().replace(",", "")
    if not cleaned or cleaned in {"-", "N/A", "n/a"}:
        return None

    multiplier = 1.0
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    if cleaned.endswith("K"):
        multiplier = 1_000.0
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1]
    elif cleaned.endswith("B"):
        multiplier = 1_000_000_000.0
        cleaned = cleaned[:-1]

    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


@dataclass(frozen=True)
class NeutralNewsFeed:
    def evaluate(self, pair: str, as_of: datetime) -> NewsAssessment:
        return NewsAssessment(
            allow_trading=True,
            score=15,
            uncertainty="neutral",
            summary="Backtest neutral news assumption",
            high_impact_events=0,
        )


class HistoricalNewsFeed:
    def __init__(
        self,
        events: pd.DataFrame,
        blackout_before_minutes: int = 30,
        blackout_after_minutes: int = 30,
        surprise_threshold: float = 0.5,
    ) -> None:
        frame = events.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["datetime", "currency"])
        frame["currency"] = frame["currency"].astype(str).str.upper().str.strip()
        frame["impact"] = frame["impact"].astype(str).str.lower().str.strip()
        frame["title"] = frame["title"].fillna("").astype(str)
        frame["actual"] = frame["actual"].apply(_parse_numeric)
        frame["forecast"] = frame["forecast"].apply(_parse_numeric)

        self.events = frame.sort_values("datetime").reset_index(drop=True)
        self.blackout_before = timedelta(minutes=blackout_before_minutes)
        self.blackout_after = timedelta(minutes=blackout_after_minutes)
        self.surprise_threshold = surprise_threshold

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        blackout_before_minutes: int = 30,
        blackout_after_minutes: int = 30,
        surprise_threshold: float = 0.5,
    ) -> "HistoricalNewsFeed":
        frame = pd.read_csv(path)
        columns = {column.lower().strip(): column for column in frame.columns}

        required = {"datetime", "currency", "impact"}
        missing = required - set(columns)
        if missing:
            raise ValueError(f"News CSV missing columns: {', '.join(sorted(missing))}")

        normalized = pd.DataFrame(
            {
                "datetime": frame[columns["datetime"]],
                "currency": frame[columns["currency"]],
                "impact": frame[columns["impact"]],
                "title": frame[columns["title"]] if "title" in columns else "",
                "actual": frame[columns["actual"]] if "actual" in columns else None,
                "forecast": frame[columns["forecast"]] if "forecast" in columns else None,
            }
        )
        return cls(
            normalized,
            blackout_before_minutes=blackout_before_minutes,
            blackout_after_minutes=blackout_after_minutes,
            surprise_threshold=surprise_threshold,
        )

    @staticmethod
    def _ensure_utc(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        return ts

    def evaluate(self, pair: str, as_of: datetime) -> NewsAssessment:
        base, quote = _pair_currencies(pair)
        if not base or not quote:
            return NewsAssessment(
                allow_trading=True,
                score=10,
                uncertainty="medium",
                summary="Invalid pair format; reduced backtest confidence",
                high_impact_events=0,
            )

        as_of_ts = self._ensure_utc(as_of)
        relevant = self.events[
            self.events["currency"].isin({base, quote}) & self.events["impact"].str.contains("high", na=False)
        ]

        high_impact_count = int(relevant.shape[0])
        high_uncertainty = False
        medium_uncertainty = False
        details: list[str] = []

        for row in relevant.itertuples(index=False):
            event_time = pd.Timestamp(row.datetime)
            if event_time.tzinfo is None:
                event_time = event_time.tz_localize(timezone.utc)
            else:
                event_time = event_time.tz_convert(timezone.utc)

            delta = event_time.to_pydatetime() - as_of_ts.to_pydatetime()
            near_event = timedelta(0) <= delta <= self.blackout_before or timedelta(0) >= delta >= -self.blackout_after
            if not near_event:
                continue

            actual = row.actual
            forecast = row.forecast
            if actual is None and forecast is None:
                high_uncertainty = True
                details.append(f"{row.currency} {row.title}: no actual/forecast")
                continue

            if actual is None:
                high_uncertainty = True
                details.append(f"{row.currency} {row.title}: actual pending")
                continue

            if forecast is None:
                medium_uncertainty = True
                details.append(f"{row.currency} {row.title}: forecast missing")
                continue

            base_value = max(abs(forecast), 1e-9)
            surprise = abs(actual - forecast) / base_value
            if surprise >= self.surprise_threshold * 1.5:
                high_uncertainty = True
                details.append(f"{row.currency} {row.title}: high surprise")
            elif surprise >= self.surprise_threshold:
                medium_uncertainty = True
                details.append(f"{row.currency} {row.title}: medium surprise")

        if high_uncertainty:
            return NewsAssessment(
                allow_trading=False,
                score=0,
                uncertainty="high",
                summary="High-impact uncertainty: " + "; ".join(details[:3]),
                high_impact_events=high_impact_count,
            )

        if medium_uncertainty:
            return NewsAssessment(
                allow_trading=True,
                score=8,
                uncertainty="medium",
                summary="Medium-impact uncertainty: " + "; ".join(details[:3]),
                high_impact_events=high_impact_count,
            )

        if high_impact_count > 0:
            return NewsAssessment(
                allow_trading=True,
                score=12,
                uncertainty="low",
                summary="High-impact events tracked; no uncertainty spikes",
                high_impact_events=high_impact_count,
            )

        return NewsAssessment(
            allow_trading=True,
            score=15,
            uncertainty="low",
            summary="No relevant high-impact events",
            high_impact_events=0,
        )
