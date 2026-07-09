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
