from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class CurrencyExposureRecord:
    currency: str
    delta: int
    expires_at: datetime


def signal_currency_deltas(symbol: str, side: str) -> dict[str, int]:
    normalized = symbol.upper().replace("/", "")
    base = normalized[:3]
    quote = normalized[3:6]
    if side.upper() == "BUY":
        return {base: 1, quote: -1}
    return {base: -1, quote: 1}


def timeframe_to_minutes(timeframe: str) -> int:
    tf = timeframe.strip().upper()
    if len(tf) < 2:
        return 0
    suffix = tf[0]
    try:
        value = int(tf[1:])
    except ValueError:
        return 0

    if value <= 0:
        return 0
    if suffix == "M":
        return value
    if suffix == "H":
        return value * 60
    if suffix == "D":
        return value * 1440
    return 0


def exposure_expiry(now: datetime, timeframe: str, bars: int, fallback_minutes: int) -> datetime:
    tf_minutes = timeframe_to_minutes(timeframe)
    if bars > 0 and tf_minutes > 0:
        minutes = bars * tf_minutes
    else:
        minutes = max(1, fallback_minutes)
    return now + timedelta(minutes=minutes)
