# Phase 6 Live Reliability

Phase 6 adds live reliability controls around the signal engine without changing SMC logic, scoring, exits, or Telegram dispatch.

## Provider Redundancy

Use the redundant data provider only after testing it in a controlled window:

```env
DATA_SOURCE=redundant
MARKET_DATA_REDUNDANCY_PRIMARY_SOURCE=live_bars
MARKET_DATA_REDUNDANCY_BACKUP_SOURCES=
MARKET_DATA_REDUNDANCY_REQUIRE_FRESH=1
MARKET_DATA_REDUNDANCY_MAX_CANDLE_AGE_SECONDS=1800
MARKET_DATA_REDUNDANCY_FAIL_CLOSED=1
MARKET_DATA_REDUNDANCY_LOG_PATH=logs/market_data_redundancy.jsonl
```

Behavior:

- `primary_source` is tried first.
- backups are tried in order only if the primary fails.
- each returned frame must pass freshness checks.
- when no provider is fresh, the system fails closed and blocks signals.
- the provider writes decision logs to `MARKET_DATA_REDUNDANCY_LOG_PATH`.

Do not use delayed providers as blind backups. If `yahoo` is listed as a backup, it still must pass freshness checks; otherwise it is rejected.

## Smoke Test

Run a one-off check without changing `.env`:

```bash
python - <<'PY'
from config import Settings
from main import _itick_config_from_settings, _live_bar_config_from_settings, _redundant_config_from_settings
from data.market_data import MarketDataCacheConfig, MarketDataClient

settings = Settings.from_env()
config = _redundant_config_from_settings(settings)
config["backup_sources"] = []
client = MarketDataClient(
    history_limit=20,
    data_source="redundant",
    mt5_login=settings.mt5_login,
    mt5_password=settings.mt5_password,
    mt5_server=settings.mt5_server,
    mt5_path=settings.mt5_path,
    itick_config=_itick_config_from_settings(settings),
    live_bar_config=_live_bar_config_from_settings(settings),
    redundant_config=config,
    cache_config=MarketDataCacheConfig(enabled=False, mode="disabled"),
)
for pair in ("EURUSD", "EURJPY", "CADJPY"):
    frame = client.fetch_ohlcv(pair, "M5", 10)
    print(pair, len(frame), frame.index[-1], frame["close"].iloc[-1])
client.close()
PY
```

## Report

Summarize provider decisions:

```bash
python -m research.market_data_redundancy_report --recent-minutes 60
```

Pass criteria:

- `Failed` is `0`.
- selected source is normally `live_bars`.
- backup selection only appears during real primary feed outages.
- stale attempts are rejected, not silently used.

## Telegram Health Alerts

The live health checker can include feed diagnostics in the same cooldown-protected Telegram alert stream:

```env
ENABLE_HEALTH_ALERTS=1
ENABLE_FEED_HEALTH_CHECKS=1
FEED_HEALTH_RECENT_MINUTES=60
FEED_HEALTH_CHECK_ITICK_WEBSOCKET=1
FEED_HEALTH_CHECK_LIVE_BARS=1
FEED_HEALTH_CHECK_REDUNDANCY=0
FEED_HEALTH_LIVE_BAR_MAX_AGE_SECONDS=180
FEED_HEALTH_LIVE_BAR_MAX_STALE_RATE=0.005
```

Manual dry run without Telegram:

```bash
python -m research.live_health_check --no-alert --output logs/live_health_status.json
```

Manual run with Telegram alerting:

```bash
python -m research.live_health_check --alert --output logs/live_health_status.json
```

The checker marks the bot unhealthy when:

- heartbeat is missing/stale/failed
- iTick WebSocket summary is in alert state
- LiveBarBuilder summary is in alert state
- redundancy check is enabled and reports failed/stale provider attempts

Alerts are sent only when the health state changes, cooldown expires, or the bot recovers.

## Feed Safe Mode

Feed safe mode is an automatic pre-scan circuit breaker. It uses the same feed diagnostics as health alerts, but instead of only notifying Telegram, it can block new signal generation when feed quality degrades.

```env
ENABLE_FEED_SAFE_MODE=1
FEED_SAFE_MODE_BLOCK_SIGNALS=1
FEED_SAFE_MODE_RECENT_MINUTES=60
FEED_SAFE_MODE_CHECK_ITICK_WEBSOCKET=1
FEED_SAFE_MODE_CHECK_LIVE_BARS=1
FEED_SAFE_MODE_CHECK_REDUNDANCY=0
FEED_SAFE_MODE_LIVE_BAR_MAX_AGE_SECONDS=180
FEED_SAFE_MODE_LIVE_BAR_MAX_STALE_RATE=0.005
FEED_SAFE_MODE_LOG_PATH=logs/feed_safe_mode.jsonl
```

Behavior:

- safe mode runs at the start of each live scan cycle.
- when every checked feed component is healthy, scanning continues normally.
- when any checked feed component is unhealthy and `FEED_SAFE_MODE_BLOCK_SIGNALS=1`, the bot skips `engine.scan_pairs`.
- LiveBarBuilder is considered degraded when stale updates exceed `FEED_SAFE_MODE_LIVE_BAR_MAX_STALE_RATE`.
- skipped cycles write `found=0` and `sent=0`; no Telegram trade signal is generated from degraded data.
- every decision is logged to `FEED_SAFE_MODE_LOG_PATH`.

Dry-run the decision without restarting the bot:

```bash
ENABLE_FEED_SAFE_MODE=1 python - <<'PY'
from config import Settings
from services.feed_safe_mode import FeedSafeModeGuard

decision = FeedSafeModeGuard(Settings.from_env()).evaluate()
print(decision.to_dict())
PY
```
