# Phase 4 Forward Performance Reporter

The forward performance reporter joins live signal candidates from the forward journal with the latest outcome rows and produces forward-test analytics.

It is read-only. It does not fetch market data, send Telegram messages, change live signals, or modify outcomes.

## Config Defaults

Add or keep these values in `.env`:

```env
ENABLE_FORWARD_PERFORMANCE_REPORT=0
FORWARD_PERFORMANCE_REPORT_PATH=reports/forward_performance_report.json
FORWARD_PERFORMANCE_SENT_ONLY=0
FORWARD_PERFORMANCE_SCORE_BUCKET_SIZE=5
FORWARD_PERFORMANCE_MIN_CLOSED_TRADES=0
```

`ENABLE_FORWARD_PERFORMANCE_REPORT` is a config marker only. The report is generated manually or by automation with the CLI below.

## Run

```bash
cd "/Users/kanannagiev/Documents/New project/project"
source .venv/bin/activate
python -m research.forward_performance_report
```

## Daily Helper

Run the current daily forward validation helper:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
scripts/run_forward_reports.sh
```

The helper writes a log to:

```bash
logs/forward_reports_daily.out.log
```

The helper now delegates to:

```bash
python -m research.daily_live_forward_report
```

## Daily Launchd Schedule

Install the daily report job:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
bash launchd/install_forward_reports.sh
```

The launch agent runs once per day at `21:30` local Mac time. On this Mac timezone, that is `16:30 UTC`, about 30 minutes after the live session window `07-16 UTC` ends.

Check status:

```bash
launchctl list | grep com.smc.forwardreports
```

Uninstall:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
bash launchd/uninstall_forward_reports.sh
```

Launchd stdout/stderr logs:

```bash
tail -f "$HOME/Library/Logs/SMCSignalEngine/forward-reports.out.log"
tail -f "$HOME/Library/Logs/SMCSignalEngine/forward-reports.err.log"
```

## Recommended Workflow

1. Run the live bot with forward journal enabled.
2. After candidates have had enough candles to play out, run the outcome tracker.
3. Run this reporter to aggregate forward performance.

```bash
python -m research.forward_outcome_tracker
python -m research.forward_performance_report
```

## Useful Modes

Only include Telegram-delivered candidates:

```bash
python -m research.forward_performance_report --sent-only
```

Export without per-candidate rows:

```bash
python -m research.forward_performance_report --no-rows
```

Use wider score buckets:

```bash
python -m research.forward_performance_report --score-bucket-size 10
```

Mark groups with fewer than 5 closed-with-R trades as insufficient sample:

```bash
python -m research.forward_performance_report --min-closed-trades 5
```

## Output

Default report path:

```bash
reports/forward_performance_report.json
```

Console summary includes:

- overall forward performance
- by pair
- by regime
- by session
- by score bucket
- by pre-trade shadow verdict
- by pre-trade shadow reason

## Metrics

Each group includes:

- candidates
- delivered and delivery rate
- closed and closed-with-R
- wins, losses, breakeven
- win rate
- AvgR and TotalR
- profit factor
- max drawdown in R
- best/worst R
- open, pending, no-outcome, and ambiguous counts

## Session Buckets

The reporter uses UTC session buckets:

- `asia_00_07_utc`
- `london_07_12_utc`
- `london_ny_overlap_12_16_utc`
- `new_york_late_16_21_utc`
- `rollover_21_24_utc`

## Notes

Groups with `sample_ok=false` should not be used for decisions yet. They need more closed forward trades before we treat the metric as reliable.
