# Phase 4 Forward-Test Journal

The forward-test journal stores every live signal as a trade candidate for later outcome tracking. It is append-only JSONL and does not change signal generation, Telegram delivery, risk logic, or pre-trade filters.

## Enable

Add these values to `.env`:

```env
ENABLE_FORWARD_JOURNAL=1
FORWARD_JOURNAL_LOG_PATH=logs/forward_journal.jsonl
FORWARD_JOURNAL_INCLUDE_SCORE_BREAKDOWN=1
```

Then run the bot normally:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
source .venv/bin/activate
python main.py
```

## Watch The Journal

```bash
tail -f logs/forward_journal.jsonl
```

## Event Types

`forward_signal_candidate` is written when the live engine creates a signal candidate.

`forward_signal_delivery` is written after Telegram delivery is attempted.

## Candidate Fields

Each candidate includes:

- `journal_id`
- `cycle_id`
- signal fingerprint
- symbol and side
- entry, stop loss, take profit, planned RR
- score and score breakdown
- HTF bias, regime, zone, trigger, structure
- entry mode and entry source
- trade management plan
- signal metadata
- optional pre-trade shadow verdict

## Delivery Fields

Each delivery event includes:

- `journal_id`
- signal fingerprint
- symbol and side
- delivered status
- Telegram latency

## Notes

This is a raw forward-test journal. Repeated signal fingerprints may appear across cycles if the live engine keeps seeing the same setup. Outcome tracking should use `journal_id` for individual candidates and `fingerprint` for duplicate/group analysis.

## Outcome Tracking

After signals have had enough time to play out, run:

```bash
python -m research.forward_outcome_tracker
```

See `docs/phase4_outcome_tracker.md` for outcome statuses and tracker options.
