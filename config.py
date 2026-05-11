from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


def _load_env_file(path: Path | None = None) -> None:
    env_path = path or Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_pairs(raw: str) -> List[str]:
    pairs = [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]
    return pairs or [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "GBPJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
        "EURJPY",
        "EURGBP",
    ]


def _parse_pair_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        left, right = item.split(":", 1)
        left_key = left.strip().upper().replace("/", "")
        right_key = right.strip().upper().replace("/", "")
        if len(left_key) == 6 and len(right_key) == 6 and left_key != right_key:
            mapping[left_key] = right_key
    return mapping


def _parse_spread_map(raw: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        left, right = item.split(":", 1)
        pair = left.strip().upper().replace("/", "")
        if len(pair) != 6:
            continue
        try:
            mapping[pair] = max(0.0, float(right.strip()))
        except ValueError:
            continue
    return mapping


def _parse_bool(raw: str, default: bool = False) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_adaptive_weights(raw: str | None) -> dict[str, dict[str, float]] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    parsed: dict[str, dict[str, float]] = {}
    for regime, value in payload.items():
        if not isinstance(regime, str) or not isinstance(value, dict):
            continue
        regime_key = regime.strip().lower()
        regime_map: dict[str, float] = {}
        for feature, weight in value.items():
            if not isinstance(feature, str):
                continue
            try:
                regime_map[feature.strip()] = float(weight)
            except (TypeError, ValueError):
                continue
        if regime_map:
            parsed[regime_key] = regime_map
    return parsed or None


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    pairs: List[str]
    scan_interval_minutes: int
    walk_forward_enabled: bool
    wf_train_months: int
    wf_test_months: int
    wf_step_months: int
    enable_realistic_execution: bool
    spread_default_pips: float
    spread_by_pair: dict[str, float]
    slippage_mode: str
    max_slippage_pips: float
    execution_delay_bars: int
    partial_fill_probability: float
    partial_fill_min_ratio: float
    limit_touch_tolerance_pips: float
    apply_spread_to_limit: bool
    random_seed: int | None
    enable_atr_risk: bool
    atr_period: int
    atr_multiplier: float
    enable_equity_protection: bool
    max_drawdown_limit: float
    drawdown_risk_reduction_factor: float
    max_consecutive_losses: int
    min_risk_multiplier: float
    ltf_timeframe: str
    htf_timeframe: str
    trigger_timeframe: str
    min_score: int
    risk_reward: float
    history_limit: int
    swing_window: int
    regime_short_window: int
    regime_long_window: int
    regime_opposition_confidence: float
    contraction_min_trigger_strength: int
    range_min_trigger_strength: int
    require_displacement_in_contraction: bool
    session_min_score: int
    enable_smt_confirmation: bool
    smt_hard_gate: bool
    smt_min_strength: float
    smt_opposite_block_strength: float
    smt_reference_map: dict[str, str]
    partial_tp_enabled: bool
    partial_tp_r: float
    partial_tp_fraction: float
    break_even_r: float
    trailing_enabled: bool
    trailing_start_r: float
    trailing_lookback_bars: int
    time_stop_bars: int
    pair_correlation_threshold: float
    correlation_lookback: int
    currency_exposure_cap: int
    portfolio_currency_gross_cap: int
    portfolio_currency_net_cap: int
    portfolio_exposure_window_minutes: int
    pair_cooldown_minutes: int
    max_entries_per_bias: int
    bias_window_minutes: int
    news_blackout_before_minutes: int
    news_blackout_after_minutes: int
    news_surprise_threshold: float
    enable_mitigation_entry: bool
    export_reports: bool
    export_regime_report: bool
    enable_adaptive_weights: bool
    adaptive_regime_weights: dict[str, dict[str, float]] | None
    enable_score_normalization: bool
    score_normalization_method: str
    score_normalization_window: int
    score_normalization_scale_factor: float
    score_normalization_backtest_only: bool
    allow_live_score_normalization: bool
    enable_dynamic_threshold: bool
    threshold_percentile: float
    threshold_rolling_window: int
    apply_dynamic_threshold: bool
    dynamic_threshold_backtest_only: bool
    allow_live_dynamic_threshold: bool
    enable_feature_analytics: bool
    export_meta_report: bool
    
    # Prop Risk Engine v2 (Week 5)
    enable_regime_engine_v2: bool
    enable_prop_risk_v2: bool
    enable_portfolio_risk_v2: bool
    enable_trade_gate_v2: bool
    enable_execution_quality_model: bool
    
    # Prop Risk v2 Settings
    prop_base_risk: float
    prop_max_risk: float
    prop_dd_threshold_low: float
    prop_dd_threshold_mid: float
    prop_dd_threshold_high: float
    prop_loss_2_reduction: float
    prop_loss_3_reduction: float
    prop_loss_4_pause: bool
    
    # Portfolio v2 Settings
    portfolio_max_currency_exposure: int
    portfolio_max_currency_gross: int
    portfolio_correlation_threshold: float
    portfolio_max_cluster: int
    portfolio_max_net_direction: int
    
    # Trade Gate Settings
    gate_min_regime_tradability: int
    gate_block_transition: bool
    
    # Execution Quality Settings
    execution_base_slippage: float
    execution_max_multiplier: float

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file()
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is required")

        return cls(
            telegram_bot_token=token,
            telegram_chat_id=chat_id,
            pairs=_parse_pairs(
                os.getenv(
                    "PAIRS",
                    "EURUSD,GBPUSD,USDJPY,GBPJPY,AUDUSD,USDCAD,USDCHF,NZDUSD,EURJPY,EURGBP",
                )
            ),
            scan_interval_minutes=max(1, int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))),
            walk_forward_enabled=_parse_bool(os.getenv("WALK_FORWARD_ENABLED", "0"), default=False),
            wf_train_months=max(1, int(os.getenv("WF_TRAIN_MONTHS", "6"))),
            wf_test_months=max(1, int(os.getenv("WF_TEST_MONTHS", "1"))),
            wf_step_months=max(1, int(os.getenv("WF_STEP_MONTHS", "1"))),
            enable_realistic_execution=_parse_bool(os.getenv("ENABLE_REALISTIC_EXECUTION", "0"), default=False),
            spread_default_pips=max(0.0, float(os.getenv("SPREAD_DEFAULT_PIPS", "0.0"))),
            spread_by_pair=_parse_spread_map(os.getenv("SPREAD_BY_PAIR", "")),
            slippage_mode=os.getenv("SLIPPAGE_MODE", "none").strip().lower(),
            max_slippage_pips=max(0.0, float(os.getenv("MAX_SLIPPAGE_PIPS", "0.0"))),
            execution_delay_bars=max(0, int(os.getenv("EXECUTION_DELAY_BARS", "0"))),
            partial_fill_probability=max(0.0, min(1.0, float(os.getenv("PARTIAL_FILL_PROBABILITY", "1.0")))),
            partial_fill_min_ratio=max(0.01, min(0.99, float(os.getenv("PARTIAL_FILL_MIN_RATIO", "0.5")))),
            limit_touch_tolerance_pips=max(0.0, float(os.getenv("LIMIT_TOUCH_TOLERANCE_PIPS", "0.0"))),
            apply_spread_to_limit=_parse_bool(os.getenv("APPLY_SPREAD_TO_LIMIT", "0"), default=False),
            random_seed=_parse_optional_int(os.getenv("RANDOM_SEED")),
            enable_atr_risk=_parse_bool(os.getenv("ENABLE_ATR_RISK", "0"), default=False),
            atr_period=max(2, int(os.getenv("ATR_PERIOD", "14"))),
            atr_multiplier=max(0.1, float(os.getenv("ATR_MULTIPLIER", "1.5"))),
            enable_equity_protection=_parse_bool(os.getenv("ENABLE_EQUITY_PROTECTION", "0"), default=False),
            max_drawdown_limit=max(0.01, float(os.getenv("MAX_DRAWDOWN_LIMIT", "10.0"))),
            drawdown_risk_reduction_factor=max(0.0, min(1.0, float(os.getenv("DRAWDOWN_RISK_REDUCTION_FACTOR", "0.5")))),
            max_consecutive_losses=max(1, int(os.getenv("MAX_CONSECUTIVE_LOSSES", "4"))),
            min_risk_multiplier=max(0.01, min(1.0, float(os.getenv("MIN_RISK_MULTIPLIER", "0.25")))),
            ltf_timeframe=os.getenv("LTF_TIMEFRAME", "M15").upper(),
            htf_timeframe=os.getenv("HTF_TIMEFRAME", "H1").upper(),
            trigger_timeframe=os.getenv("TRIGGER_TIMEFRAME", "M5").upper(),
            min_score=max(0, min(100, int(os.getenv("MIN_SIGNAL_SCORE", "70")))),
            risk_reward=max(1.0, float(os.getenv("RISK_REWARD", "2.0"))),
            history_limit=max(150, int(os.getenv("HISTORY_LIMIT", "500"))),
            swing_window=max(2, int(os.getenv("SWING_WINDOW", "3"))),
            regime_short_window=max(5, int(os.getenv("REGIME_SHORT_WINDOW", "20"))),
            regime_long_window=max(20, int(os.getenv("REGIME_LONG_WINDOW", "80"))),
            regime_opposition_confidence=max(0.0, min(1.0, float(os.getenv("REGIME_OPPOSITION_CONFIDENCE", "0.70")))),
            contraction_min_trigger_strength=max(0, min(20, int(os.getenv("CONTRACTION_MIN_TRIGGER_STRENGTH", "9")))),
            range_min_trigger_strength=max(0, min(20, int(os.getenv("RANGE_MIN_TRIGGER_STRENGTH", "8")))),
            require_displacement_in_contraction=_parse_bool(
                os.getenv("REQUIRE_DISPLACEMENT_IN_CONTRACTION", "1"),
                default=True,
            ),
            session_min_score=max(0, min(20, int(os.getenv("SESSION_MIN_SCORE", "5")))),
            enable_smt_confirmation=_parse_bool(os.getenv("ENABLE_SMT_CONFIRMATION", "1"), default=True),
            smt_hard_gate=_parse_bool(os.getenv("SMT_HARD_GATE", "0"), default=False),
            smt_min_strength=max(0.0, min(100.0, float(os.getenv("SMT_MIN_STRENGTH", "60")))),
            smt_opposite_block_strength=max(0.0, min(100.0, float(os.getenv("SMT_OPPOSITE_BLOCK_STRENGTH", "80")))),
            smt_reference_map=_parse_pair_map(
                os.getenv(
                    "SMT_REFERENCE_MAP",
                    "EURUSD:GBPUSD,GBPUSD:EURUSD,AUDUSD:NZDUSD,NZDUSD:AUDUSD,USDJPY:EURJPY,EURJPY:USDJPY,GBPJPY:EURJPY,EURGBP:GBPUSD,USDCAD:AUDUSD,USDCHF:EURUSD",
                )
            ),
            partial_tp_enabled=_parse_bool(os.getenv("PARTIAL_TP_ENABLED", "1"), default=True),
            partial_tp_r=max(0.0, float(os.getenv("PARTIAL_TP_R", "1.0"))),
            partial_tp_fraction=max(0.0, min(0.95, float(os.getenv("PARTIAL_TP_FRACTION", "0.50")))),
            break_even_r=max(0.0, float(os.getenv("BREAK_EVEN_R", "1.0"))),
            trailing_enabled=_parse_bool(os.getenv("TRAILING_ENABLED", "1"), default=True),
            trailing_start_r=max(0.0, float(os.getenv("TRAILING_START_R", "1.5"))),
            trailing_lookback_bars=max(1, int(os.getenv("TRAILING_LOOKBACK_BARS", "6"))),
            time_stop_bars=max(0, int(os.getenv("TIME_STOP_BARS", "48"))),
            pair_correlation_threshold=max(0.0, min(0.99, float(os.getenv("PAIR_CORRELATION_THRESHOLD", "0.82")))),
            correlation_lookback=max(30, int(os.getenv("CORRELATION_LOOKBACK", "120"))),
            currency_exposure_cap=max(0, int(os.getenv("CURRENCY_EXPOSURE_CAP", "2"))),
            portfolio_currency_gross_cap=max(0, int(os.getenv("PORTFOLIO_CURRENCY_GROSS_CAP", "4"))),
            portfolio_currency_net_cap=max(0, int(os.getenv("PORTFOLIO_CURRENCY_NET_CAP", "2"))),
            portfolio_exposure_window_minutes=max(1, int(os.getenv("PORTFOLIO_EXPOSURE_WINDOW_MINUTES", "240"))),
            pair_cooldown_minutes=max(0, int(os.getenv("PAIR_COOLDOWN_MINUTES", "30"))),
            max_entries_per_bias=max(0, int(os.getenv("MAX_ENTRIES_PER_BIAS", "2"))),
            bias_window_minutes=max(0, int(os.getenv("BIAS_WINDOW_MINUTES", "240"))),
            news_blackout_before_minutes=max(0, int(os.getenv("NEWS_BLACKOUT_BEFORE_MIN", "30"))),
            news_blackout_after_minutes=max(0, int(os.getenv("NEWS_BLACKOUT_AFTER_MIN", "30"))),
            news_surprise_threshold=max(0.0, float(os.getenv("NEWS_SURPRISE_THRESHOLD", "0.50"))),
            enable_mitigation_entry=_parse_bool(os.getenv("ENABLE_MITIGATION_ENTRY", "1"), default=True),
            export_reports=_parse_bool(os.getenv("EXPORT_REPORTS", "1"), default=True),
            export_regime_report=_parse_bool(os.getenv("EXPORT_REGIME_REPORT", "0"), default=False),
            enable_adaptive_weights=_parse_bool(os.getenv("ENABLE_ADAPTIVE_WEIGHTS", "0"), default=False),
            adaptive_regime_weights=_parse_adaptive_weights(os.getenv("ADAPTIVE_WEIGHTS_JSON")),
            enable_score_normalization=_parse_bool(os.getenv("ENABLE_SCORE_NORMALIZATION", "0"), default=False),
            score_normalization_method=os.getenv("SCORE_NORMALIZATION_METHOD", "minmax").strip().lower(),
            score_normalization_window=max(10, int(os.getenv("SCORE_NORMALIZATION_WINDOW", "200"))),
            score_normalization_scale_factor=max(0.0, float(os.getenv("SCORE_NORMALIZATION_SCALE_FACTOR", "1.0"))),
            score_normalization_backtest_only=_parse_bool(os.getenv("SCORE_NORMALIZATION_BACKTEST_ONLY", "0"), default=False),
            allow_live_score_normalization=_parse_bool(os.getenv("ALLOW_LIVE_SCORE_NORMALIZATION", "1"), default=True),
            enable_dynamic_threshold=_parse_bool(os.getenv("ENABLE_DYNAMIC_THRESHOLD", "0"), default=False),
            threshold_percentile=max(0.0, min(100.0, float(os.getenv("THRESHOLD_PERCENTILE", "80")))),
            threshold_rolling_window=max(10, int(os.getenv("THRESHOLD_ROLLING_WINDOW", "200"))),
            apply_dynamic_threshold=_parse_bool(os.getenv("APPLY_DYNAMIC_THRESHOLD", "0"), default=False),
            dynamic_threshold_backtest_only=_parse_bool(os.getenv("DYNAMIC_THRESHOLD_BACKTEST_ONLY", "0"), default=False),
            allow_live_dynamic_threshold=_parse_bool(os.getenv("ALLOW_LIVE_DYNAMIC_THRESHOLD", "1"), default=True),
            enable_feature_analytics=_parse_bool(os.getenv("ENABLE_FEATURE_ANALYTICS", "0"), default=False),
            export_meta_report=_parse_bool(os.getenv("EXPORT_META_REPORT", "0"), default=False),
            
            # Prop Risk Engine v2 flags
            enable_regime_engine_v2=_parse_bool(os.getenv("ENABLE_REGIME_ENGINE_V2", "0"), default=False),
            enable_prop_risk_v2=_parse_bool(os.getenv("ENABLE_PROP_RISK_V2", "0"), default=False),
            enable_portfolio_risk_v2=_parse_bool(os.getenv("ENABLE_PORTFOLIO_RISK_V2", "0"), default=False),
            enable_trade_gate_v2=_parse_bool(os.getenv("ENABLE_TRADE_GATE_V2", "0"), default=False),
            enable_execution_quality_model=_parse_bool(os.getenv("ENABLE_EXECUTION_QUALITY_MODEL", "0"), default=False),
            
            # Prop Risk v2 Settings
            prop_base_risk=max(0.1, float(os.getenv("PROP_BASE_RISK", "1.0"))),
            prop_max_risk=max(0.1, float(os.getenv("PROP_MAX_RISK", "2.0"))),
            prop_dd_threshold_low=max(0.0, float(os.getenv("PROP_DD_THRESHOLD_LOW", "3.0"))),
            prop_dd_threshold_mid=max(0.0, float(os.getenv("PROP_DD_THRESHOLD_MID", "6.0"))),
            prop_dd_threshold_high=max(0.0, float(os.getenv("PROP_DD_THRESHOLD_HIGH", "10.0"))),
            prop_loss_2_reduction=max(0.0, min(1.0, float(os.getenv("PROP_LOSS_2_REDUCTION", "0.8")))),
            prop_loss_3_reduction=max(0.0, min(1.0, float(os.getenv("PROP_LOSS_3_REDUCTION", "0.6")))),
            prop_loss_4_pause=_parse_bool(os.getenv("PROP_LOSS_4_PAUSE", "1"), default=True),
            
            # Portfolio v2 Settings
            portfolio_max_currency_exposure=max(0, int(os.getenv("PORTFOLIO_MAX_CURRENCY_EXPOSURE", "2"))),
            portfolio_max_currency_gross=max(0, int(os.getenv("PORTFOLIO_MAX_CURRENCY_GROSS", "4"))),
            portfolio_correlation_threshold=max(0.0, min(0.99, float(os.getenv("PORTFOLIO_CORRELATION_THRESHOLD", "0.82")))),
            portfolio_max_cluster=max(0, int(os.getenv("PORTFOLIO_MAX_CLUSTER", "3"))),
            portfolio_max_net_direction=max(0, int(os.getenv("PORTFOLIO_MAX_NET_DIRECTION", "4"))),
            
            # Trade Gate Settings
            gate_min_regime_tradability=max(0, min(100, int(os.getenv("GATE_MIN_REGIME_TRADABILITY", "30")))),
            gate_block_transition=_parse_bool(os.getenv("GATE_BLOCK_TRANSITION", "1"), default=True),
            
            # Execution Quality Settings
            execution_base_slippage=max(0.0, float(os.getenv("EXECUTION_BASE_SLIPPAGE", "0.5"))),
            execution_max_multiplier=max(1.0, float(os.getenv("EXECUTION_MAX_MULTIPLIER", "2.0"))),
        )
