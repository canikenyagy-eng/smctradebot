# Phase 5 iTick WebSocket Shadow

This layer monitors iTick live WebSocket quotes without changing signal generation.

The goal is to verify whether iTick is fresher than Yahoo/cache before switching live candles.

## Enable

Keep `DATA_SOURCE=yahoo` until the shadow feed proves stable.

```env
ENABLE_ITICK_WEBSOCKET_SHADOW=1
ITICK_WEBSOCKET_URL=wss://api.itick.org/forex
ITICK_WEBSOCKET_REGION=GB
ITICK_WEBSOCKET_TYPES=quote
ITICK_WEBSOCKET_LOG_PATH=logs/itick_websocket_shadow.jsonl
ITICK_WEBSOCKET_SUMMARY_PATH=reports/itick_websocket_shadow_summary.json
ITICK_WEBSOCKET_HEARTBEAT_SECONDS=30
ITICK_WEBSOCKET_RECONNECT_SECONDS=5
ITICK_WEBSOCKET_MAX_RECONNECT_SECONDS=60
ITICK_WEBSOCKET_RECONNECT_BACKOFF_FACTOR=2
ITICK_WEBSOCKET_RECONNECT_JITTER_SECONDS=1
ITICK_WEBSOCKET_STALE_SECONDS=5
ITICK_WEBSOCKET_MAX_LATENCY_SECONDS=2
ITICK_WEBSOCKET_MAX_STALE_RATE=0
ITICK_WEBSOCKET_MAX_SLOW_RATE=0.005
ITICK_WEBSOCKET_MAX_CONNECTION_ERRORS=3
ITICK_WEBSOCKET_MAX_LATEST_QUOTE_AGE_SECONDS=180
```

`ITICK_API_KEY`, `ITICK_API_KEY_HEADER=token`, `ITICK_AUTH_SCHEME=`, and `ITICK_SYMBOL_FORMAT={base}{quote}` are reused from the existing iTick REST config.

## Short Probe

Run a controlled probe without restarting the live bot:

```bash
python -m research.itick_websocket_shadow_probe \
  --pairs EURUSD,EURJPY,CADJPY \
  --seconds 20
```

Then summarize:

```bash
python -m research.itick_websocket_shadow_report --recent-minutes 60
```

## Live Shadow Mode

When enabled, `main.py` starts a background WebSocket task and writes quote events to:

```text
logs/itick_websocket_shadow.jsonl
```

The task is shadow-only:

- no signal candles are replaced
- no score changes
- no Telegram signal changes
- no execution/autotrading changes

## Comparison Workflow

Run both reports:

```bash
python -m research.market_data_diagnostics_report --recent-minutes 60
python -m research.itick_websocket_shadow_report --recent-minutes 60
```

Use Yahoo/cache diagnostics to measure candle age and iTick WebSocket diagnostics to measure live quote latency.

## Pass Criteria

Use iTick as live-candle candidate only after several live sessions where:

- WebSocket quote count is non-zero for all live pairs
- p95 quote latency is below `2s`
- stale quote rate is at or below `ITICK_WEBSOCKET_MAX_STALE_RATE`
- slow quote rate is at or below `ITICK_WEBSOCKET_MAX_SLOW_RATE`
- reconnects/errors are rare
- connection-error count stays below `ITICK_WEBSOCKET_MAX_CONNECTION_ERRORS` inside the report window
- latest quote age stays below `ITICK_WEBSOCKET_MAX_LATEST_QUOTE_AGE_SECONDS`
- price behavior is consistent with another reference source

If iTick passes, the next step is a separate `LiveBarBuilder` that builds M5/M15/H1 candles from WebSocket quotes.

## LiveBarBuilder Shadow

After WebSocket quotes are stable, enable local OHLCV construction:

```env
ENABLE_LIVE_BAR_BUILDER=1
LIVE_BAR_BUILDER_TIMEFRAMES=M5,M15,H1
LIVE_BAR_BUILDER_DIR=data/live_bars/itick
LIVE_BAR_BUILDER_LOG_PATH=logs/live_bars_itick.jsonl
LIVE_BAR_BUILDER_SUMMARY_PATH=reports/live_bar_builder_summary.json
LIVE_BAR_BUILDER_MAX_BARS=1000
LIVE_BAR_BUILDER_FLUSH_SECONDS=2
LIVE_BAR_BUILDER_MAX_QUOTE_AGE_SECONDS=5
```

This still does not change live signal generation. It only builds local bars from iTick quotes.

Run a controlled probe:

```bash
python -m research.live_bar_builder_probe \
  --pairs EURUSD,EURJPY,CADJPY \
  --seconds 60
```

Summarize generated bars:

```bash
python -m research.live_bar_builder_report --recent-minutes 60
```

Generated CSV bars are written to:

```text
data/live_bars/itick/
```

The next migration step is a dedicated live-bar market data provider that can read these files and replace Yahoo only after bar freshness is stable.

## LiveBarMarketDataProvider

`DATA_SOURCE=live_bars` enables a hybrid provider:

- historical candles come from `LIVE_BAR_PROVIDER_FALLBACK_SOURCE`
- fresh tail candles come from `data/live_bars/itick/*.csv`
- overlapping timestamps are replaced by the iTick-built bars
- the normal OHLCV cache is bypassed for `live_bars` so fresh local bars are not hidden by cached reads

Recommended candidate config:

```env
DATA_SOURCE=live_bars
LIVE_BAR_PROVIDER_DIR=data/live_bars/itick
LIVE_BAR_PROVIDER_FALLBACK_SOURCE=yahoo
LIVE_BAR_PROVIDER_INCLUDE_CURRENT_BAR=0
LIVE_BAR_PROVIDER_REQUIRE_LIVE_OVERLAY=1
LIVE_BAR_PROVIDER_MAX_LIVE_BAR_AGE_SECONDS=7200
```

Use completed bars first (`LIVE_BAR_PROVIDER_INCLUDE_CURRENT_BAR=0`) to avoid intrabar repaint risk. Current in-progress bars can be tested later, but should not be the first live default.

Smoke-check the provider before switching live:

```bash
python -m research.live_bar_provider_check \
  --pairs EURUSD,EURJPY,CADJPY \
  --timeframes M5,M15,H1 \
  --limit 10 \
  --require-live-overlay
```

Only switch the live bot after:

- WebSocket shadow report has `Alert: False`
- LiveBarBuilder report has `Alert: False`
- `live_bar_provider_check` returns rows for M5, M15, and H1
- at least one full H1 bar has been built locally

## Troubleshooting

`server rejected WebSocket connection: HTTP 401` means the WebSocket cluster rejected authentication. Check that:

- `ITICK_API_KEY` is not expired
- the subscription includes WebSocket forex access
- `ITICK_API_KEY_HEADER=token`
- `ITICK_AUTH_SCHEME=` is empty for iTick header-token auth
