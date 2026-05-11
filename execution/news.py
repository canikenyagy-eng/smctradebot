from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from xml.etree import ElementTree

import requests


@dataclass(frozen=True)
class NewsAssessment:
    allow_trading: bool
    score: int
    uncertainty: str
    summary: str
    high_impact_events: int


class NewsFilter:
    CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

    def __init__(
        self,
        blackout_before_minutes: int = 30,
        blackout_after_minutes: int = 30,
        surprise_threshold: float = 0.5,
        timeout_seconds: int = 8,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.blackout_before = timedelta(minutes=blackout_before_minutes)
        self.blackout_after = timedelta(minutes=blackout_after_minutes)
        self.surprise_threshold = surprise_threshold
        self.timeout_seconds = timeout_seconds
        self.cache_ttl = timedelta(seconds=max(30, cache_ttl_seconds))
        self._cached_events: List[Dict[str, Any]] | None = None
        self._cached_at: datetime | None = None

    @staticmethod
    def _parse_numeric(value: str | None) -> float | None:
        if not value:
            return None

        cleaned = value.strip().replace(",", "")
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

        cleaned = re.sub(r"[^0-9+\-.]", "", cleaned)
        if cleaned in {"", "+", "-", "."}:
            return None

        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None

    @staticmethod
    def _parse_datetime(date_text: str | None, time_text: str | None) -> datetime | None:
        if not date_text or not time_text:
            return None

        date_text = date_text.strip()
        time_text = time_text.strip().lower().replace(" ", "")

        if time_text in {"alladay", "alladay", "tentative"}:
            return None

        combined = f"{date_text} {time_text}"
        formats = [
            "%m-%d-%Y %I:%M%p",
            "%m-%d-%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %I:%M%p",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(combined, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return None

    def _fetch_events(self) -> List[Dict[str, Any]]:
        response = requests.get(self.CALENDAR_URL, timeout=self.timeout_seconds)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        events: List[Dict[str, Any]] = []

        for event in root.findall("event"):
            currency = (event.findtext("country") or "").strip().upper()
            impact = (event.findtext("impact") or "").strip().lower()
            date_text = event.findtext("date")
            time_text = event.findtext("time")

            dt = self._parse_datetime(date_text, time_text)
            if dt is None:
                continue

            events.append(
                {
                    "currency": currency,
                    "impact": impact,
                    "title": (event.findtext("title") or "").strip(),
                    "datetime": dt,
                    "actual": self._parse_numeric(event.findtext("actual")),
                    "forecast": self._parse_numeric(event.findtext("forecast")),
                }
            )

        return events

    def _get_events(self) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        if self._cached_events is not None and self._cached_at is not None:
            if now - self._cached_at <= self.cache_ttl:
                return self._cached_events

        events = self._fetch_events()
        self._cached_events = events
        self._cached_at = now
        return events

    @staticmethod
    def _pair_currencies(pair: str) -> tuple[str, str]:
        clean = pair.upper().replace("/", "")
        if len(clean) < 6:
            return "", ""
        return clean[:3], clean[3:6]

    def evaluate_pair(self, pair: str) -> NewsAssessment:
        base, quote = self._pair_currencies(pair)
        if not base or not quote:
            return NewsAssessment(
                allow_trading=True,
                score=10,
                uncertainty="medium",
                summary="Invalid pair format; reduced news confidence",
                high_impact_events=0,
            )

        now = datetime.now(timezone.utc)
        try:
            events = self._get_events()
        except Exception:
            return NewsAssessment(
                allow_trading=True,
                score=7,
                uncertainty="medium",
                summary="News feed unavailable; confidence reduced",
                high_impact_events=0,
            )

        relevant = [
            event
            for event in events
            if event["currency"] in {base, quote} and "high" in event["impact"]
        ]

        high_impact_count = 0
        high_uncertainty = False
        medium_uncertainty = False
        details: List[str] = []

        for event in relevant:
            delta = event["datetime"] - now
            high_impact_count += 1

            in_pre_blackout = timedelta(0) <= delta <= self.blackout_before
            in_post_blackout = timedelta(0) >= delta >= -self.blackout_after
            near_event = in_pre_blackout or in_post_blackout
            if not near_event:
                continue

            actual = event["actual"]
            forecast = event["forecast"]

            if actual is None and forecast is None:
                high_uncertainty = True
                details.append(f"{event['currency']} {event['title']}: no actual/forecast")
                continue

            if actual is None:
                high_uncertainty = True
                details.append(f"{event['currency']} {event['title']}: actual pending")
                continue

            if forecast is None:
                medium_uncertainty = True
                details.append(f"{event['currency']} {event['title']}: forecast missing")
                continue

            base_value = max(abs(forecast), 1e-9)
            surprise = abs(actual - forecast) / base_value
            if surprise >= self.surprise_threshold * 1.5:
                high_uncertainty = True
                details.append(f"{event['currency']} {event['title']}: high surprise")
            elif surprise >= self.surprise_threshold:
                medium_uncertainty = True
                details.append(f"{event['currency']} {event['title']}: medium surprise")

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
