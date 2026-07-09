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
