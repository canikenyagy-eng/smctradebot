from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class TimeframeConfig:
    interval: str
    period: str
    resample_rule: str | None = None


TIMEFRAME_MAP: Dict[str, TimeframeConfig] = {
    "M1": TimeframeConfig(interval="1m", period="7d"),
    "M5": TimeframeConfig(interval="5m", period="60d"),
    "M15": TimeframeConfig(interval="15m", period="60d"),
    "M30": TimeframeConfig(interval="30m", period="60d"),
    "H1": TimeframeConfig(interval="60m", period="730d"),
    "H4": TimeframeConfig(interval="60m", period="730d", resample_rule="4h"),
    "D1": TimeframeConfig(interval="1d", period="10y"),
}


class MarketDataClient:
    def __init__(self, history_limit: int = 500) -> None:
        self.history_limit = history_limit

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        clean_pair = pair.upper().replace("/", "")
        if len(clean_pair) != 6:
            raise ValueError(f"Unsupported forex symbol: {pair}")
        return f"{clean_pair}=X"

    @staticmethod
    def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        frame = frame.rename(columns=rename_map)
        expected = ["open", "high", "low", "close"]
        for col in expected:
            if col not in frame.columns:
                raise ValueError(f"Missing column '{col}' in downloaded data")

        if "volume" not in frame.columns:
            frame["volume"] = 0.0

        frame = frame[["open", "high", "low", "close", "volume"]].dropna()

        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")

        return frame.sort_index()

    @staticmethod
    def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        if frame.empty:
            return frame

        return (
            frame.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    def fetch_ohlcv(self, pair: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        tf_key = timeframe.upper()
        if tf_key not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        tf_cfg = TIMEFRAME_MAP[tf_key]
        ticker = self._normalize_pair(pair)
        raw = yf.download(
            tickers=ticker,
            interval=tf_cfg.interval,
            period=tf_cfg.period,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        frame = self._standardize_frame(raw)
        if tf_cfg.resample_rule:
            frame = self._resample(frame, tf_cfg.resample_rule)

        if frame.empty:
            raise ValueError(f"No market data for {pair} {timeframe}")

        max_rows = limit or self.history_limit
        return frame.tail(max_rows).copy()
