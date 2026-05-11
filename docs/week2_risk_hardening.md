## Week 2: Risk Hardening + Realistic Backtesting

### Enable Realistic Execution
```bash
ENABLE_REALISTIC_EXECUTION=1
SPREAD_DEFAULT_PIPS=1.2
SPREAD_BY_PAIR=EURUSD:1.0,GBPJPY:1.8
SLIPPAGE_MODE=random
MAX_SLIPPAGE_PIPS=1.5
EXECUTION_DELAY_BARS=1
PARTIAL_FILL_PROBABILITY=0.85
PARTIAL_FILL_MIN_RATIO=0.50
LIMIT_TOUCH_TOLERANCE_PIPS=0.1
APPLY_SPREAD_TO_LIMIT=0
RANDOM_SEED=42
```

### Enable ATR-based Backtest Risk
```bash
ENABLE_ATR_RISK=1
ATR_PERIOD=14
ATR_MULTIPLIER=1.5
```

### Enable Equity Protection
```bash
ENABLE_EQUITY_PROTECTION=1
MAX_DRAWDOWN_LIMIT=10.0
DRAWDOWN_RISK_REDUCTION_FACTOR=0.5
MAX_CONSECUTIVE_LOSSES=4
MIN_RISK_MULTIPLIER=0.25
```

### Run
```bash
.venv/bin/python backtest_runner.py --pairs EURUSD,GBPUSD --history-limit 3000
```

### Exports
- `backtests/<timestamp>/summary.json`
- `reports/realistic_comparison.json` (when realism features are enabled)
- `reports/score_distribution.json` (with `--analyze-scores`)
- `reports/feature_analysis.json` (with `--analyze-scores`)

