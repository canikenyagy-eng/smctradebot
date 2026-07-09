from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from config import Settings
from services.live_health import (
    HealthAlertSettings,
    HealthAlertState,
    HealthCheckSettings,
    LiveHealthChecker,
    combine_health_components,
    format_health_message,
)
from services.feed_health import build_feed_health_components
from services.telegram import TelegramSignalService


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check live bot heartbeat and optionally alert Telegram.")
    parser.add_argument("--heartbeat", default=None, help="Heartbeat JSON path")
    parser.add_argument("--max-age-minutes", type=int, default=None, help="Maximum allowed heartbeat age")
    parser.add_argument("--alert", action="store_true", help="Send Telegram alert when unhealthy/recovered")
    parser.add_argument("--no-alert", action="store_true", help="Disable Telegram alert even if env enables it")
    parser.add_argument("--state", default=None, help="Alert state JSON path")
    parser.add_argument("--cooldown-minutes", type=int, default=None, help="Alert cooldown")
    parser.add_argument("--output", default=None, help="Optional JSON result output path")
    parser.add_argument("--fail-on-unhealthy", action="store_true", help="Exit with code 2 when unhealthy")
    return parser


def build_check_settings(settings: Settings, args: argparse.Namespace) -> HealthCheckSettings:
    return HealthCheckSettings(
        heartbeat_path=args.heartbeat or settings.live_heartbeat_path,
        max_scan_age_minutes=args.max_age_minutes or settings.health_max_scan_age_minutes,
    )


def build_alert_settings(settings: Settings, args: argparse.Namespace) -> HealthAlertSettings:
    enabled = settings.enable_health_alerts
    if args.alert:
        enabled = True
    if args.no_alert:
        enabled = False
    return HealthAlertSettings(
        enabled=enabled,
        state_path=args.state or settings.health_alert_state_path,
        cooldown_minutes=args.cooldown_minutes or settings.health_alert_cooldown_minutes,
    )


async def maybe_send_alert(settings: Settings, alert_state: HealthAlertState, result) -> tuple[bool, str]:
    should_send, reason = alert_state.should_send(result)
    if not should_send:
        alert_state.update(result, alert_sent=False, alert_reason=reason)
        return False, reason

    telegram = TelegramSignalService(
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        send_retries=settings.telegram_send_retries,
        retry_base_delay_seconds=settings.telegram_retry_base_delay_seconds,
    )
    try:
        delivered = await telegram.send_text(format_health_message(result), label="live_health")
    finally:
        await telegram.close()
    alert_state.update(result, alert_sent=delivered, alert_reason=reason)
    return delivered, reason


def write_output(path: str | None, payload: dict[str, object]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def print_result(payload: dict[str, object]) -> None:
    status = "OK" if payload.get("ok") else "UNHEALTHY"
    print()
    print(f"LIVE HEALTH: {status}")
    print(f"Status: {payload.get('status')}")
    print(f"Reason: {payload.get('reason')}")
    print(f"Age: {payload.get('age_seconds')} / {payload.get('max_age_seconds')} seconds")
    print(f"Heartbeat: {payload.get('heartbeat_path')}")
    print(f"Alert: sent={payload.get('alert_sent')} reason={payload.get('alert_reason')}")
    components = payload.get("components")
    if isinstance(components, list) and components:
        print("Components:")
        for component in components:
            if isinstance(component, dict):
                marker = "OK" if component.get("ok") else "ALERT"
                print(f"  {marker} {component.get('name')}: {component.get('reason')}")


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    checker = LiveHealthChecker(build_check_settings(settings, args))
    result = checker.check()
    result = combine_health_components(result, build_feed_health_components(settings))
    alert_state = HealthAlertState(build_alert_settings(settings, args))
    alert_sent, alert_reason = asyncio.run(maybe_send_alert(settings, alert_state, result))
    payload = result.to_dict()
    payload["alert_sent"] = alert_sent
    payload["alert_reason"] = alert_reason
    write_output(args.output, payload)
    print_result(payload)
    if args.fail_on_unhealthy and not result.ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
