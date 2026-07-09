from __future__ import annotations

import argparse
import asyncio
import logging

from config import Settings
from services.itick_websocket_shadow import ItickWebSocketShadowClient, ItickWebSocketShadowSettings
from services.live_bar_builder import LiveBarBuilder, LiveBarBuilderSettings


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short iTick WebSocket live-bar builder probe.")
    parser.add_argument("--pairs", default=None, help="Comma-separated symbols. Defaults to PAIRS/settings.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Probe duration in seconds")
    parser.add_argument("--bars-dir", default=None, help="Override generated bars directory")
    parser.add_argument("--log", default=None, help="Override live-bar JSONL path")
    return parser


def parse_pairs(raw: str | None, fallback: list[str]) -> list[str]:
    if not raw:
        return fallback
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


async def run_probe() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    pairs = parse_pairs(args.pairs, settings.pairs)
    builder = LiveBarBuilder(
        LiveBarBuilderSettings(
            enabled=True,
            source="itick_websocket_probe",
            timeframes=tuple(settings.live_bar_builder_timeframes),
            bars_dir=args.bars_dir or settings.live_bar_builder_dir,
            log_path=args.log or settings.live_bar_builder_log_path,
            max_bars_per_timeframe=settings.live_bar_builder_max_bars,
            flush_interval_seconds=settings.live_bar_builder_flush_seconds,
            max_quote_age_seconds=settings.live_bar_builder_max_quote_age_seconds,
        )
    )
    client = ItickWebSocketShadowClient(
        ItickWebSocketShadowSettings(
            enabled=True,
            api_key=settings.itick_api_key,
            url=settings.itick_websocket_url,
            api_key_header=settings.itick_api_key_header,
            auth_scheme=settings.itick_auth_scheme,
            symbol_format=settings.itick_symbol_format,
            region=settings.itick_websocket_region,
            subscription_types=settings.itick_websocket_types,
            log_path=settings.itick_websocket_log_path,
            heartbeat_seconds=settings.itick_websocket_heartbeat_seconds,
            reconnect_seconds=settings.itick_websocket_reconnect_seconds,
            stale_seconds=settings.itick_websocket_stale_seconds,
            max_latency_seconds=settings.itick_websocket_max_latency_seconds,
        ),
        quote_consumers=[builder.on_quote],
    )
    try:
        await client.run(pairs, run_seconds=max(1.0, float(args.seconds)))
    finally:
        builder.flush()


def main() -> None:
    configure_logging()
    asyncio.run(run_probe())


if __name__ == "__main__":
    main()
