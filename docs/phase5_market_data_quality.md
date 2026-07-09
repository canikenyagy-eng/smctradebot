# Phase 5 Market Data Freshness And Latency Diagnostics

This layer adds observability around live OHLCV fetches and optionally blocks stale live signals using the existing freshness gate.

It does not change strategy logic, scoring, exits, risk, Telegram signal formatting, or auto-trading behavior.

## Components

- `MarketDataClient` can write one diagnostics row per OHLCV fetch.
- `ENABLE_MARKET_DATA_FRESHNESS_GATE` blocks live signals if the trigger candle is stale.
- `research.market_data_diagnostics_report` summarizes recent fetch quality.
- The daily forward report helper now also writes market data diagnostics summary.

## Config

Add or keep these values in `.env`:

```env
ENABLE_MARKET_DATA_FRESHNESS_GATE=1
MAX_LIVE_CANDLE_AGE_SECONDS=1800
ENABLE_MARKET_DATA_DIAGNOSTICS=1
MARKET_DATA_DIAGNOSTICS_LOG_PATH=logs/market_data_diagnostics.jsonl
MARKET_DATA_DIAGNOSTICS_MAX_LATENCY_SECONDS=5.0
MARKET_DATA_DIAGNOSTICS_MAX_CANDLE_AGE_SECONDS=1800
MARKET_DATA_DIAGNOSTICS_LOG_CACHE_HITS=1
MARKET_DATA_DIAGNOSTICS_SUMMARY_PATH=reports/market_data_diagnostics_summary.json
```

`MAX_LIVE_CANDLE_AGE_SECONDS=1800` allows up to 30 minutes of candle age. That is intentionally tolerant for Yahoo-style delayed data while still blocking very stale data.

## Diagnostics Log

Each live OHLCV fetch can log:

- pair
- timeframe
- data source
- cache mode
- served source: provider, cache, stale cache fallback, etc.
- latency seconds
- last candle timestamp
- candle age seconds
- stale flag
- slow flag
- provider/cache error details when present

Log path:

```bash
logs/market_data_diagnostics.jsonl
```

## Manual Report

```bash
cd "/Users/kanannagiev/Documents/New project/project"
source .venv/bin/activate
python -m research.market_data_diagnostics_report
```

Analyze a shorter window:

```bash
python -m research.market_data_diagnostics_report --recent-minutes 180
```

Fail with exit code `2` if errors, stale rows, or slow rows exist:

```bash
python -m research.market_data_diagnostics_report --fail-on-alert
```

## Daily Report Integration

The daily helper now runs:

```bash
python -m research.forward_outcome_tracker
python -m research.forward_performance_report --no-rows
python -m research.market_data_diagnostics_report
```

Helper command:

```bash
scripts/run_forward_reports.sh
```

Summary output:

```bash
reports/market_data_diagnostics_summary.json
```

## Live Startup Log

On startup, the engine log shows:

```text
data_freshness_gate=True/1800s data_diagnostics=True
```

## Interpretation

`errors > 0` means provider/cache fetches failed.

`stale > 0` means returned candles were older than the configured max age.

`slow > 0` means fetch latency exceeded `MARKET_DATA_DIAGNOSTICS_MAX_LATENCY_SECONDS`.

`served_from` shows whether live scans used provider data, fresh cache, cache-only data, or stale cache fallback.

## Notes

The freshness gate only affects live signal acceptance when `ENABLE_MARKET_DATA_FRESHNESS_GATE=1` and the signal engine runtime is `live`.

For true live quote freshness validation, use the iTick WebSocket shadow workflow in `docs/phase5_itick_websocket_shadow.md`.

Diagnostics are passive. They only write JSONL rows and summaries.
