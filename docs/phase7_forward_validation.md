# Phase 7 Forward Validation

Phase 7 turns live Telegram candidates into measurable forward evidence without adding auto-trading or changing strategy logic.

## Daily Report

Run this after the active session:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
source .venv/bin/activate

python -m research.daily_live_forward_report
```

The helper:

- refreshes theoretical TP/SL/time-stop outcomes from the configured market data source.
- builds the rolling forward performance report.
- summarizes pair, regime, session, score bucket, and pre-trade shadow performance.
- summarizes feed safe-mode, live health, and feed-quality components.
- writes `reports/daily_live_forward_report.json`.

## Telegram Summary

Enable the compact Telegram summary after the daily report:

```env
DAILY_FORWARD_REPORT_SEND_TELEGRAM=1
DAILY_FORWARD_REPORT_TELEGRAM_ON_NO_SIGNALS=1
```

Manual one-off Telegram send:

```bash
python -m research.daily_live_forward_report --telegram
```

Suppress Telegram for a manual diagnostic run:

```bash
python -m research.daily_live_forward_report --no-telegram
```

## Launchd Automation

Install the post-session daily report job:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
bash launchd/install_forward_reports.sh
```

The job runs once per day at `21:30` local Mac time, about 30 minutes after the `07-16 UTC` live session ends on this machine.

Check status:

```bash
launchctl list | grep com.smc.forwardreports
```

Inspect logs:

```bash
tail -f "$HOME/Library/Logs/SMCSignalEngine/forward-reports.out.log"
tail -f logs/forward_reports_daily.out.log
```

## Useful Options

```bash
python -m research.daily_live_forward_report --recent-minutes 480
python -m research.daily_live_forward_report --skip-outcome-update
python -m research.daily_live_forward_report --sent-only
python -m research.daily_live_forward_report --include-rows
```

## Pass Criteria

- `Feed: OK`
- `Safe-mode blocks: 0`
- closed outcomes gradually increase over multiple sessions.
- `AvgR` and `PF` stay positive after enough samples.
- losing clusters are explainable by pair, regime, session, or feed conditions.

## Recommendation Labels

- `COLLECT_MORE_FORWARD_DATA`: not enough closed forward outcomes yet.
- `FEED_REVIEW`: feed health or safe mode degraded during the window.
- `KEEP_PROFILE`: forward PF and AvgR meet target thresholds.
- `HOLD_PROFILE`: positive but not strong enough to expand.
- `TIGHTEN_OR_PAUSE_PROFILE`: forward expectancy is negative or PF is below 1.
- `REVIEW_BLOCKED_WINDOWS`: safe mode blocked scans and the blocked window needs inspection.
