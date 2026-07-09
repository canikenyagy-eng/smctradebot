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
from services.itick_websocket_shadow import ItickWebSocketShadowReporter
from services.live_bar_builder import LiveBarBuilderReporter
from services.telegram import TelegramSignalService
from research.market_data_redundancy_report import build_report as build_redundancy_report


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


def _component(name: str, ok: bool, reason: str, details: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "name": name,
        "ok": bool(ok),
        "reason": reason,
        "details": details or {},
    }


def _itick_component(settings: Settings) -> dict[str, object]:
    reporter = ItickWebSocketShadowReporter(
        log_path=settings.itick_websocket_log_path,
        summary_path=settings.itick_websocket_summary_path,
        recent_minutes=settings.feed_health_recent_minutes,
        stale_seconds=settings.itick_websocket_stale_seconds,
        max_latency_seconds=settings.itick_websocket_max_latency_seconds,
        max_stale_rate=settings.itick_websocket_max_stale_rate,
        max_slow_rate=settings.itick_websocket_max_slow_rate,
        max_connection_errors=settings.itick_websocket_max_connection_errors,
        max_latest_quote_age_seconds=settings.itick_websocket_max_latest_quote_age_seconds,
    )
    report = reporter.build_report()
    reporter.write_report(report)
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        return _component("itick_websocket", False, "summary missing")
    ok = overall.get("alert") is not True
    reason = (
        "healthy"
        if ok
        else (
            f"quotes={overall.get('quotes', 0)} stale={overall.get('stale', 0)} "
            f"slow_rate={float(overall.get('slow_rate', 0.0)):.3%} "
            f"errors={overall.get('connection_errors', 0)} "
            f"latest_age={overall.get('latest_quote_age_seconds')}"
        )
    )
    return _component(
        "itick_websocket",
        ok,
        reason,
        {
            "quotes": overall.get("quotes", 0),
            "stale": overall.get("stale", 0),
            "slow": overall.get("slow", 0),
            "slow_rate": overall.get("slow_rate", 0.0),
            "connection_errors": overall.get("connection_errors", 0),
            "latest_quote_age_seconds": overall.get("latest_quote_age_seconds"),
        },
    )


def _live_bar_component(settings: Settings) -> dict[str, object]:
    reporter = LiveBarBuilderReporter(
        log_path=settings.live_bar_builder_log_path,
        recent_minutes=settings.feed_health_recent_minutes,
        max_bar_age_seconds=settings.feed_health_live_bar_max_age_seconds,
    )
    report = reporter.build_report()
    output = Path(settings.live_bar_builder_summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        return _component("live_bar_builder", False, "summary missing")
    ok = overall.get("alert") is not True
    reason = (
        "healthy"
        if ok
        else (
            f"updates={overall.get('updates', 0)} stale_updates={overall.get('stale_updates', 0)} "
            f"max_age={overall.get('max_bar_age_seconds')}"
        )
    )
    return _component(
        "live_bar_builder",
        ok,
        reason,
        {
            "updates": overall.get("updates", 0),
            "closed": overall.get("closed", 0),
            "stale_updates": overall.get("stale_updates", 0),
            "max_bar_age_seconds": overall.get("max_bar_age_seconds", 0),
        },
    )


def _redundancy_component(settings: Settings) -> dict[str, object]:
    report = build_redundancy_report(Path(settings.market_data_redundancy_log_path), settings.feed_health_recent_minutes)
    failed = int(report.get("failed", 0) or 0)
    stale_attempts = report.get("stale_attempts", {})
    stale_count = sum(int(value) for value in stale_attempts.values()) if isinstance(stale_attempts, dict) else 0
    ok = failed == 0 and stale_count == 0
    reason = "healthy" if ok else f"failed={failed} stale_attempts={stale_count}"
    return _component(
        "market_data_redundancy",
        ok,
        reason,
        {
            "requests": report.get("requests", 0),
            "ok": report.get("ok", 0),
            "failed": failed,
            "selected_source": report.get("selected_source", {}),
            "stale_attempts": stale_attempts,
        },
    )


def build_feed_components(settings: Settings) -> list[dict[str, object]]:
    if not settings.enable_feed_health_checks:
        return []
    components: list[dict[str, object]] = []
    if settings.feed_health_check_itick_websocket and (
        settings.enable_itick_websocket_shadow or settings.enable_live_bar_builder
    ):
        components.append(_itick_component(settings))
    if settings.feed_health_check_live_bars and settings.enable_live_bar_builder:
        components.append(_live_bar_component(settings))
    should_check_redundancy = settings.feed_health_check_redundancy or settings.data_source.strip().lower() == "redundant"
    if should_check_redundancy:
        components.append(_redundancy_component(settings))
    return components


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    checker = LiveHealthChecker(build_check_settings(settings, args))
    result = checker.check()
    result = combine_health_components(result, build_feed_components(settings))
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
