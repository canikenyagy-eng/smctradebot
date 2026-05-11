# SMC Role Redefinition - Institutional System

## Executive Summary

SMC transforms from **Decision System** → **Feature Extraction Subsystem**

## Pipeline Position

```
Regime → Expectancy → Risk Budget → Trade Gate → SMC Features → Execution → Portfolio
                    ↑
            SMC provides FEATURES here
```

## Module Roles

| Module | Institutional Role | Regime Weight |
|--------|-------------------|--------------|
| structure | Feature: BOS detection | trend_strong=1.0 |
| liquidity | Feature: sweep events | expansion=1.0 |
| fvg | Feature: gap presence | trend_strong=1.0 |
| order_block | Feature: support proximity | range_tight=0.8 |
| mtf | Feature: bias alignment | trend_strong=1.0 |
| trigger | Feature: composite signal | trend_strong=1.0 |
| zones | Feature: price levels | range_tight=0.5 |
| mitigation | Feature: return level | all=0.3 |
| smt | Feature: institution flow | trend_strong=1.0 |
| regime | CLASSIFICATION | ALWAYS (first) |

## Key Principles

1. **NO SMC component makes decisions** - all are feature extractors
2. **Regime controls activation** - weights vary by regime
3. **Features are probabilistic** - includes strength/confidence
4. **Composite input to expectancy** - multiple features combined

## Transformation Summary

| Attribute | Before | After |
|-----------|--------|-------|
| Role | Signal system | Feature system |
| Decision | SMC decides | Expectancy decides |
| Output | Trade decision | Features |
| Weighting | Static | Regime-dependent |
| Filter | Score threshold | Expectancy filter |
| Activation | Always active | Conditional |