from __future__ import annotations

import argparse
import asyncio
import logging

from config import Settings
from services.itick_websocket_shadow import ItickWebSocketShadowClient, ItickWebSocketShadowSettings


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short shadow-only iTick WebSocket quote probe.")
    parser.add_argument("--pairs", default=None, help="Comma-separated symbols. Defaults to live PAIRS/settings.")
    parser.add_argument("--seconds", type=float, default=20.0, help="Probe duration in seconds")
    parser.add_argument("--log", default=None, help="Override WebSocket shadow JSONL path")
    return parser


def parse_pairs(raw: str | None, fallback: list[str]) -> list[str]:
    if not raw:
        return fallback
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


async def run_probe() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    pairs = parse_pairs(args.pairs, settings.pairs)
    shadow = ItickWebSocketShadowClient(
        ItickWebSocketShadowSettings(
            enabled=True,
            api_key=settings.itick_api_key,
            url=settings.itick_websocket_url,
            api_key_header=settings.itick_api_key_header,
            auth_scheme=settings.itick_auth_scheme,
            symbol_format=settings.itick_symbol_format,
            region=settings.itick_websocket_region,
            subscription_types=settings.itick_websocket_types,
            log_path=args.log or settings.itick_websocket_log_path,
            heartbeat_seconds=settings.itick_websocket_heartbeat_seconds,
            reconnect_seconds=settings.itick_websocket_reconnect_seconds,
            stale_seconds=settings.itick_websocket_stale_seconds,
            max_latency_seconds=settings.itick_websocket_max_latency_seconds,
        )
    )
    await shadow.run(pairs, run_seconds=max(1.0, float(args.seconds)))


def main() -> None:
    configure_logging()
    asyncio.run(run_probe())


if __name__ == "__main__":
    main()
