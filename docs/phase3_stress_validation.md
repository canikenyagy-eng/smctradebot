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

## Interpretation Rules

Treat a profile as live-candidate only if:

- PF stays above `1.3` in moderate stress.
- AvgR remains positive after spread/slippage/delay.
- Monte Carlo p05 is acceptable for the account/risk plan.
- Max drawdown remains compatible with prop/account rules.
- Harsh stress does not fully collapse the edge.

If `moderate` is positive but MC p05 is negative, keep the profile in shadow/live-candidate mode before enabling it as a default.
