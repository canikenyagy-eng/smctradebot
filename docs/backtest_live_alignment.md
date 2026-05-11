# Backtest-Live Alignment Policy

## Core Principle

**Same signal generation pipeline for both backtest and live.**

No separate logic branches. No simulation bias.

---

## Unified Pipeline

```
┌─────────────────────────────────────────────┐
│           CORE (Shared)                     │
├─────────────────────────────────────────────┤
│  Regime Detection   → Same logic            │
│  SMC Features     → Same extraction       │
│  Scoring         → Same calculation     │
│  Trade Gate      → Same permission       │
│  Risk Engine    → Same sizing          │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
   BACKTEST                LIVE
   Execution          Execution
   Quality              Live
   Adjustment         Broker
```

---

## Alignment Checks

### 1. Score Normalization
- **Before:** backtest_only = True (different logic)
- **After:** backtest_only = False (same for both)

### 2. Dynamic Threshold  
- **Before:** backtest_only = True (different logic)
- **After:** backtest_only = False (same for both)

### 3. Execution Quality
- Backtest: Applies slippage/spread model
- Live: Real broker fills
- This is intentional difference (realism vs reality)

### 4. Trade Gate
- Same checks for both modes
- No backtest bypass

---

## Forbidden Patterns

### ❌ Separate Logic
```python
if backtest:
    # Different scoring  ❌
    score = normalize_backtest(score)
else:
    # Different scoring ❌
    score = normalize_live(score)
```

### ❌ Backtest Bypass
```python
if backtest:
    skip_risk_check()  ❌
```

---

## Execution Quality (Acceptable Difference)

Only execution quality adjustment differs:
- Backtest: Applies realistic slippage/spread
- Live: Uses actual broker fills

This is correct - backtest should be realistic.