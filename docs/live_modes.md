# Live Modes

Live modes are optional overlays for the Telegram signal engine. They do not change backtest logic and remain disabled unless `ENABLE_LIVE_MODE=1`.

## Disabled / Legacy

```env
ENABLE_LIVE_MODE=0
```

The engine uses normal `.env` settings.

## Balanced

Research profile:

```text
Pairs: EURUSD, EURJPY, CADJPY
Session: 07-16 UTC / 10-19 MSK
Min score: pair-specific
Blocked regime: TREND
Exit profile: m15_vol_liq_v1
```

Enable:

```env
ENABLE_LIVE_MODE=1
LIVE_MODE=balanced
```

## Aggressive

Research profile:

```text
Pairs: EURUSD, EURJPY, CADJPY
Session: 07-16 UTC / 10-19 MSK
Min score: pair-specific
Blocked regime: TREND
Exit profile: m15_vol_liq_v1
```

Enable:

```env
ENABLE_LIVE_MODE=1
LIVE_MODE=aggressive
```

Notes:

- The validated expansion set is `EURUSD + EURJPY + CADJPY`.
- `USDJPY` is excluded from the aggressive expansion set after harsh stress testing showed it dragged portfolio AvgR, PF, Monte Carlo p05, and drawdown stability.
- Do not add `USDJPY` back unless a separate stricter profile passes stress and portfolio validation.

## Conservative

Research profile:

```text
Pairs: EURUSD
Session: 12-16 UTC / 15-19 MSK
Min score: 80
Blocked regime: TREND
Exit profile: m15_vol_liq_v1
```

Enable:

```env
ENABLE_LIVE_MODE=1
LIVE_MODE=conservative
```

## Notes

- Live modes only send Telegram signals; they do not auto-trade.
- `balanced` is designed for higher signal frequency.
- `aggressive` is experimental and should be used only when more signal flow is worth lower quality.
- `conservative` is designed for lower noise and prop-style caution.
- If `LIVE_MODE` is invalid, the engine falls back to legacy settings.
- Use `docs/pair_profiles.md` when each pair needs its own custom threshold/session/regime rules.
