# Phase 4 Live Telemetry

Phase 4 starts with a live observability layer. It is disabled by default and does not change signal generation or Telegram delivery.

## Enable Telemetry

Add to `.env`:

```env
ENABLE_LIVE_TELEMETRY=1
LIVE_TELEMETRY_LOG_PATH=logs/live_telemetry.jsonl
LIVE_TELEMETRY_INCLUDE_SIGNAL_DETAILS=1
```

Optional RC shadow observation can run alongside telemetry:

```env
ENABLE_PRE_TRADE_FILTER=0
ENABLE_PRE_TRADE_FILTER_SHADOW=1
PRE_TRADE_BLOCK_EXPANSION_CONTINUATION=1
PRE_TRADE_BLOCK_EXPANSION_CONTINUATION_FALLBACK=0
```

## Events

The logger writes JSONL rows with these event types:

- `live_engine_started`
- `live_scan_started`
- `live_signals_found`
- `live_pre_trade_shadow_summary`
- `live_telegram_delivery`
- `live_scan_completed`
- `live_scan_failed`

## Watch Logs

```bash
tail -f logs/live_telemetry.jsonl
```

Use telemetry for forward validation and operational monitoring. Keep it separate from trading logic: telemetry should observe, not decide.
