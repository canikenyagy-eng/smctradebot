# AI in Trading System - Role Definition

## Current Policy

**AI (Gemini/LLM) is DESCRIPTIVE ONLY.**

AI must NOT make trading decisions.

---

## Allowed AI Usage

### 1. Signal Formatting (Post-Processing)
```python
# AI can format signal for display
 SIGNAL_SUMMARY = ai.summarize(signal)  # ✅ OK
```

### 2. Trade Context Description
```python
# AI can describe market context
context = ai.describe(regime, liquidity)  # ✅ OK
```

### 3. Report Generation
```python
# AI can generate trade reports
report = ai.generate_report(trades)  # ✅ OK
```

---

## Forbidden AI Usage

### ❌ AI MUST NOT:
- Modify trade direction (BUY ↔ SELL)
- Change signal score
- Override trade gate
- Filter signals
- Adjust risk parameters
- Modify entry/exit levels

---

## System Architecture

```
Deterministic Core:
├── RegimeEngine → regime classification
├── SMC → feature extraction
├── Scoring → score calculation
├── TradeGate → permission
└── RiskEngine → position sizing

AI (Post-Process Only):
├── Signal formatting
├── Report generation
├── Context description
```

---

## Implementation Rule

**If AI fails → System continues with deterministic output only.**