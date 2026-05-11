# Profitability Bottleneck Analysis

## Primary Causes of Edge Failure

### P0 - Critical
1. **Score Threshold Arbitrary** - min_score=70 is historical artifact without statistical basis
2. **Regime Weights Static** - Not adjusted by live performance
3. **Transition Not Blocked** - Still generates signals in transition with 0 weight

### P1 - Major
1. **Score = Weighted Sum** - Features 0-15 scale, not derived from win rates
2. **Shadow Scoring Duplication** - Creates false confidence via double-counting
3. **Expectancy Not Linked To Scoring** - Score doesn't predict expectancy

### P2 - Moderate
1. **Execution Quality Off By Default** - Backtest unrealistic
2. **No Signal Count Monitoring** - Unbalanced regime exposure

## Structural Weaknesses

### Issues:
1. Scoring not probabilistic - score 40 ≠ 40% win probability
2. AND logic - Over-filters creating small N
3. Binary threshold - score 69 vs 71 has no statistical difference
4. Risk reactive - Budget should control signal, not react to it

## Regime Dependencies

| Regime | Issue |
|--------|-------|
| trend_strong | Over-traded |
| range_wide | High false positives |
| transition | Still generates signals |

## Institutional Fix Direction

1. **Link score to expectancy data** - Score → expected win rate mapping
2. **Live feedback loop** - Track actual win rate by score range
3. **Regime signal limits** - max 3 signals per regime per day
4. **Remove AND logic** - Use ANY 2 of 3 instead
5. **Transition zero-tolerance** - Block all trades