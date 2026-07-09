from __future__ import annotations

import argparse
import logging

from config import Settings
from data.market_data import MarketDataCacheConfig, MarketDataClient, MarketDataDiagnosticsConfig
from main import _itick_config_from_settings, _live_bar_config_from_settings


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-check the live_bars market data provider.")
    parser.add_argument("--pairs", default="EURUSD,EURJPY,CADJPY")
    parser.add_argument("--timeframes", default="M5,M15,H1")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--include-current", action="store_true", help="Temporarily include current in-progress live bars")
    parser.add_argument("--require-live-overlay", action="store_true", help="Require a live bar overlay for every fetch")
    return parser


def parse_csv(raw: str) -> list[str]:
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    live_bar_config = _live_bar_config_from_settings(settings)
    if args.include_current:
        live_bar_config["include_current_bar"] = True
    if args.require_live_overlay:
        live_bar_config["require_live_overlay"] = True

    client = MarketDataClient(
        history_limit=max(args.limit, settings.history_limit),
        data_source="live_bars",
        itick_config=_itick_config_from_settings(settings),
        live_bar_config=live_bar_config,
        cache_config=MarketDataCacheConfig(enabled=False, mode="disabled"),
        diagnostics_config=MarketDataDiagnosticsConfig(enabled=False),
    )
    try:
        failures = 0
        for pair in parse_csv(args.pairs):
            for timeframe in parse_csv(args.timeframes):
                try:
                    frame = client.fetch_ohlcv(pair, timeframe, limit=args.limit)
                    last_time = frame.index[-1].isoformat() if not frame.empty else "-"
                    last_close = float(frame["close"].iloc[-1]) if not frame.empty else 0.0
                    print(
                        f"{pair:<8} {timeframe:<4} rows={len(frame):<4} "
                        f"last={last_time:<30} close={last_close}"
                    )
                except Exception as exc:
                    failures += 1
                    print(f"{pair:<8} {timeframe:<4} ERROR {exc}")
        if failures:
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
