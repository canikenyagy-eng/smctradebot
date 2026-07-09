# Phase 4 Forward Outcome Tracker

The forward outcome tracker reads `forward_signal_candidate` rows from the forward-test journal and marks whether each candidate later reached TP, SL, or time-stop using OHLCV data.

It is research/monitoring only. It does not change live signals, Telegram delivery, risk, exits, or filters.

## Enable Config Defaults

Add or keep these values in `.env`:

```env
ENABLE_FORWARD_OUTCOME_TRACKER=0
FORWARD_OUTCOME_LOG_PATH=logs/forward_outcomes.jsonl
FORWARD_OUTCOME_SUMMARY_PATH=reports/forward_outcomes_summary.json
FORWARD_OUTCOME_TIMEFRAME=M15
FORWARD_OUTCOME_HISTORY_LIMIT=1500
FORWARD_OUTCOME_SENT_ONLY=0
FORWARD_OUTCOME_MAX_HOLD_BARS=48
FORWARD_OUTCOME_ENTRY_EXPIRY_BARS=0
FORWARD_OUTCOME_AMBIGUOUS_POLICY=ambiguous
```

`ENABLE_FORWARD_OUTCOME_TRACKER` is a config marker only. The tracker is run manually or by automation with the CLI below.

## Run

```bash
cd "/Users/kanannagiev/Documents/New project/project"
source .venv/bin/activate
python -m research.forward_outcome_tracker
```

## Useful Modes

Track only Telegram-delivered signals:

```bash
python -m research.forward_outcome_tracker --sent-only
```

Use cache only:

```bash
python -m research.forward_outcome_tracker --cache-only
```

Refresh market data cache first:

```bash
python -m research.forward_outcome_tracker --refresh-cache
```

Analyze without writing outcome rows:

```bash
python -m research.forward_outcome_tracker --no-write
```

Use M5 outcome bars for more granular marking:

```bash
python -m research.forward_outcome_tracker --timeframe M5 --history-limit 3000
```

## Output Files

Outcome events:

```bash
logs/forward_outcomes.jsonl
```

Latest summary:

```bash
reports/forward_outcomes_summary.json
```

## Outcome Statuses

`closed` means the tracker reached a terminal outcome.

`open` means entry was filled but not enough future bars have passed to hit TP, SL, or time-stop.

`pending_entry` means a limit-style entry has not touched yet and has not expired.

`entry_not_filled` means a limit-style entry expired before fill.

`waiting_for_data` means no future candles are available yet.

`insufficient_data` means the provider/cache did not return usable OHLCV.

## Exit Reasons

`take_profit` means the candidate reached TP first.

`stop_loss` means the candidate reached SL first.

`time_stop` means neither TP nor SL hit before max hold/time-stop bars; exit is marked at close.

`ambiguous_tp_sl` means TP and SL were both inside the same candle. Default policy keeps `r_multiple=null` and records `r_min`/`r_max` instead of inventing intrabar order.

## Notes

The tracker uses `static_tp_sl_time_stop_v1`. It intentionally does not model partial exits, trailing stops, or break-even movement yet. That keeps the first forward labels conservative and easy to audit.

## Performance Report

After outcomes have been written, generate grouped forward analytics:

```bash
python -m research.forward_performance_report
```

See `docs/phase4_forward_performance_report.md` for grouped metrics and options.
