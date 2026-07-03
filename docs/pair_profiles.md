# Pair Profiles

Pair profiles let each Forex pair use its own signal admission rules while keeping one shared portfolio/risk engine.

The layer can override, per pair:

- `min_score`
- `evaluation_step` for backtests
- `session_windows_utc`
- `regime_blocklist`

It does not change SMC feature calculations, exit engine internals, Telegram delivery, or auto-trade anything.

## Backtest First

```env
ENABLE_PAIR_PROFILES=1
PAIR_PROFILES_BACKTEST_ONLY=1
ALLOW_LIVE_PAIR_PROFILES=0
PAIR_PROFILES_JSON={"EURUSD":{"min_score":80,"evaluation_step":2,"session_windows_utc":"07-16","regime_blocklist":"TREND"},"EURJPY":{"min_score":78,"evaluation_step":3,"session_windows_utc":"07-16","regime_blocklist":"TREND"},"CADJPY":{"min_score":80,"evaluation_step":3,"session_windows_utc":"07-16","regime_blocklist":"TREND"}}
```

With this setup, `backtest_runner.py` applies pair profiles, but live ignores custom profiles.

## Enable In Live

Only after validation:

```env
ENABLE_PAIR_PROFILES=1
PAIR_PROFILES_BACKTEST_ONLY=1
ALLOW_LIVE_PAIR_PROFILES=1
PAIR_PROFILES_JSON={"EURUSD":{"min_score":80,"evaluation_step":2,"session_windows_utc":"07-16","regime_blocklist":"TREND"},"EURJPY":{"min_score":78,"evaluation_step":3,"session_windows_utc":"07-16","regime_blocklist":"TREND"},"CADJPY":{"min_score":80,"evaluation_step":3,"session_windows_utc":"07-16","regime_blocklist":"TREND"}}
```

When live pair profiles are allowed, the profile keys become the scanned pair universe.
`evaluation_step` is currently a backtest-only cadence override. Live scanning still uses `SCAN_INTERVAL_MINUTES`.

## Example Profiles

Validated expansion portfolio:

```json
{
  "EURUSD": {
    "min_score": 80,
    "evaluation_step": 2,
    "session_windows_utc": "07-16",
    "regime_blocklist": "TREND"
  },
  "EURJPY": {
    "min_score": 78,
    "evaluation_step": 3,
    "session_windows_utc": "07-16",
    "regime_blocklist": "TREND"
  },
  "CADJPY": {
    "min_score": 80,
    "evaluation_step": 3,
    "session_windows_utc": "07-16",
    "regime_blocklist": "TREND"
  }
}
```

`USDJPY` is intentionally excluded from the validated expansion portfolio after harsh portfolio stress showed it reduced AvgR, PF, Monte Carlo p05, and drawdown stability. Re-test `USDJPY` separately with stricter thresholds/session constraints before adding it back.

Conservative:

```json
{
  "EURUSD": {
    "min_score": 80,
    "session_windows_utc": "12-16",
    "regime_blocklist": "TREND"
  }
}
```

## Notes

- `session_windows_utc` supports `"07-16"`, `"07-16,12-16"`, `["07-16"]`, or `[[7, 16]]`.
- `regime_blocklist` supports `"TREND"` or `["TREND"]`.
- `evaluation_step` supports positive integers and affects backtests only.
- Invalid pair keys or disabled pair profiles are ignored.
- If `ALLOW_LIVE_PAIR_PROFILES=0`, custom pair profiles are ignored in live even when `ENABLE_PAIR_PROFILES=1`.
