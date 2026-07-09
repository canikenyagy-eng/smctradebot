# Phase 3 Stress Validation

`research.phase3_stress_validation` is a research-only runner for fast live-candidate validation.

It does not change live signals, Telegram delivery, SMC logic, exits, or risk rules. It wraps the existing `BacktestEngine` with:

- execution stress presets
- persisted trade cache
- Monte Carlo from completed trades
- pair/regime breakdown
- one JSON report updated after each scenario

## Strict LTF Candidate

Run the current strict LTF candidate with the step-3 pair profile:

```bash
.venv/bin/python -m research.phase3_stress_validation \
  --pairs EURUSD,EURJPY,CADJPY \
  --history-limit 3000 \
  --evaluation-step 3 \
  --stress-presets ideal,moderate,harsh \
  --signal-profile strict_ltf_only \
  --pair-profile-preset strict_ltf_step3_v1 \
  --cache-only \
  --trade-cache \
  --mc-iterations 5000 \
  --export-run-artifacts \
  --output reports/phase3_strict_ltf_stress.json
```

For a quick control run:

```bash
.venv/bin/python -m research.phase3_stress_validation \
  --pairs EURUSD,EURJPY,CADJPY \
  --history-limit 3000 \
  --evaluation-step 3 \
  --stress-presets moderate \
  --signal-profile strict_ltf_only \
  --pair-profile-preset strict_ltf_step3_v1 \
  --cache-only \
  --trade-cache \
  --mc-iterations 5000 \
  --output reports/phase3_strict_ltf_moderate_control.json
```

## Presets

- `ideal`: no execution friction.
- `mild`: light spread/slippage.
- `moderate`: live-candidate stress.
- `harsh`: robustness stress with wider spread, volatility slippage, delay, and lower fill probability.

Use `--stress-presets all` to run every preset.

## Profiles

Signal profiles:

- `current`: use `.env` / current config as-is.
- `strict_ltf_only`: enables only `ENABLE_STRICT_LTF_DIRECTION_GATE` and keeps the other Phase 2 hardening flags off.

Pair profile presets:

- `current`: use `.env` / current config as-is.
- `validated_expansion_v1`: `EURUSD` step 2, `EURJPY` step 3, `CADJPY` step 3.
- `strict_ltf_step3_v1`: same portfolio, but all pairs use step 3 for strict LTF validation.

## Output

The JSON report contains:

- `runs[].metrics`
- `runs[].monte_carlo`
- `runs[].pairs`
- `runs[].regimes`
- `runs[].execution_settings`
- `runs[].trade_cache`
- `ranking`

If `--export-run-artifacts` is enabled, each scenario also exports:

- `summary.json`
- `trades.csv`
- `pair_summary.csv`

## Friction Diagnostics

After exporting run artifacts, diagnose why edge compresses under execution friction:

```bash
.venv/bin/python -m research.phase3_friction_diagnostics \
  --suite-dir reports/phase3_strict_ltf_step3_full_suite \
  --baseline ideal \
  --scenarios moderate,harsh \
  --top-losses 10 \
  --output reports/phase3_friction_diagnostics.json
```

The diagnostics report breaks down:

- losing trades by pair, regime, entry mode, entry source, exit reason, and sleeve
- total spread/slippage/delay cost
- worst trades by R
- matched-trade deltas versus the ideal baseline
- flags for regime drag, fallback-entry drag, timeout drag, and execution-friction drag

## Phase 3.3 Pre-Trade Filter

The pre-trade filter is disabled by default and affects backtests only when explicitly wired into a runner/config.

Environment flags:

- `ENABLE_PRE_TRADE_FILTER=0`
- `PRE_TRADE_BLOCK_EXPANSION_CONTINUATION=0`
- `PRE_TRADE_BLOCK_EXPANSION_CONTINUATION_FALLBACK=0`

Current release-candidate hypothesis:

- `timeout_fast_soft_fallback_no_expansion_continuation`
- `timeout_fast` exits
- soft MARKET fallback floor at trigger strength `8`
- pre-trade veto for `EXPANSION + continuation`, independent of entry source

Run the release-candidate validation pack:

```bash
.venv/bin/python -m research.phase3_hypothesis_calibration \
  --pairs EURUSD,EURJPY,CADJPY \
  --history-limit 3000 \
  --evaluation-step 3 \
  --scenarios timeout_fast,timeout_fast_soft_fallback_no_expansion_continuation \
  --stress-presets mild,moderate,harsh \
  --cache-only \
  --trade-cache \
  --mc-iterations 5000 \
  --output reports/phase3_release_candidate_step3_full_stress.json
```

For cadence validation, repeat with `--evaluation-step 2` and compare against step 3.

## Interpretation Rules

Treat a profile as live-candidate only if:

- PF stays above `1.3` in moderate stress.
- AvgR remains positive after spread/slippage/delay.
- Monte Carlo p05 is acceptable for the account/risk plan.
- Max drawdown remains compatible with prop/account rules.
- Harsh stress does not fully collapse the edge.

If `moderate` is positive but MC p05 is negative, keep the profile in shadow/live-candidate mode before enabling it as a default.

## Shadow-Live RC Veto

To observe the current RC veto in live without changing Telegram signals, copy the values from:

```bash
docs/phase3_rc_shadow_live.env.example
```

into `.env`, then run the bot normally. Keep `ENABLE_PRE_TRADE_FILTER=0`; only `ENABLE_PRE_TRADE_FILTER_SHADOW=1` should be enabled for observation.

The shadow logger writes JSONL rows to:

```bash
logs/pre_trade_filter_shadow.jsonl
```

Watch the log with:

```bash
tail -f logs/pre_trade_filter_shadow.jsonl
```

A row with `"would_block": true` means the RC veto would have blocked that live signal, but the signal is still sent to Telegram because this is shadow-only.
