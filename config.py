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
    return pairs or ["EURUSD", "USDJPY"]


def _parse_csv_upper(raw: str) -> List[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


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


def _parse_session_windows(raw: str) -> List[tuple[int, int]]:
    windows: List[tuple[int, int]] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        separator = "-" if "-" in text else ":"
        if separator not in text:
            continue
        left, right = text.split(separator, 1)
        try:
            start = int(left.strip())
            end = int(right.strip())
        except ValueError:
            continue
        start = max(0, min(23, start))
        end = max(0, min(24, end))
        if start == end:
            continue
        item = (start, end)
        if item not in windows:
            windows.append(item)
    return windows


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


def _parse_object_map(raw: str | None) -> dict[str, dict[str, object]] | None:
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

    parsed: dict[str, dict[str, object]] = {}
    for regime, value in payload.items():
        if not isinstance(regime, str) or not isinstance(value, dict):
            continue
        key = regime.strip().lower()
        if not key:
            continue
        parsed[key] = dict(value)
    return parsed or None


def _parse_json_dict(raw: str | None) -> dict[str, object]:
    if raw is None:
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_send_retries: int
    telegram_retry_base_delay_seconds: float
    pairs: List[str]
    data_source: str
    mt5_login: int
    mt5_password: str
    mt5_server: str
    mt5_path: str
    itick_api_key: str
    itick_base_url: str
    itick_ohlcv_path_template: str
    itick_ticks_path_template: str
    itick_api_key_header: str
    itick_api_key_query_param: str
    itick_auth_scheme: str
    itick_symbol_format: str
    itick_timeout_seconds: float
    itick_timeframe_map: dict[str, object]
    itick_extra_headers: dict[str, object]
    enable_market_data_shadow: bool
    market_data_shadow_candidate_source: str
    market_data_shadow_timeframes: List[str]
    market_data_shadow_log_path: str
    market_data_shadow_cache_dir: str
    market_data_shadow_ttl_hours: float
    market_data_shadow_compare_signals: bool
    market_data_shadow_max_close_diff_pips: float
    market_data_shadow_max_staleness_seconds: int
    enable_market_data_freshness_gate: bool
    max_live_candle_age_seconds: int
    scan_interval_minutes: int
    enable_live_mode: bool
    live_mode: str
    enable_pair_profiles: bool
    pair_profiles: dict[str, object]
    pair_profiles_backtest_only: bool
    allow_live_pair_profiles: bool
    market_data_cache_enabled: bool
    market_data_cache_dir: str
    market_data_cache_ttl_hours: float
    market_data_cache_mode: str
    backtest_evaluation_step: int
    enable_backtest_snapshot_cache: bool
    backtest_snapshot_cache_max_entries: int
    backtest_end_time: str
    enable_backtest_trade_cache: bool
    backtest_trade_cache_dir: str
    backtest_trade_cache_version: str
    backtest_account_enabled: bool
    backtest_starting_balance: float
    backtest_risk_per_trade: float
    backtest_account_currency: str
    walk_forward_enabled: bool
    wf_train_months: int
    wf_test_months: int
    wf_step_months: int
    enable_realistic_execution: bool
    skip_realistic_comparison: bool
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
    enable_strict_ltf_direction_gate: bool
    enable_market_fallback_entry: bool
    market_fallback_min_trigger_strength: int
    market_fallback_require_displacement: bool
    enable_pip_aware_liquidity: bool
    liquidity_equal_level_tolerance_pips: float
    liquidity_atr_tolerance_factor: float
    session_min_score: int
    enable_session_gate: bool
    session_gate_windows_utc: List[tuple[int, int]]
    session_gate_backtest_only: bool
    allow_live_session_gate: bool
    enable_regime_label_gate: bool
    regime_label_blocklist: List[str]
    regime_gate_backtest_only: bool
    allow_live_regime_gate: bool
    enable_smt_confirmation: bool
    smt_backtest_only: bool
    allow_live_smt_confirmation: bool
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
    enable_order_block_shadow: bool
    order_block_shadow_backtest_only: bool
    allow_live_order_block_shadow: bool
    export_reports: bool
    export_regime_report: bool
    enable_adaptive_weights: bool
    adaptive_weights_preset: str
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
    enable_regime_engine_v2: bool
    enable_prop_risk_v2: bool
    enable_portfolio_risk_v2: bool
    enable_trade_gate_v2: bool
    enable_execution_quality_model: bool
    enable_pre_trade_filter: bool
    enable_pre_trade_filter_shadow: bool
    pre_trade_filter_shadow_log_path: str
    enable_live_telemetry: bool
    live_telemetry_log_path: str
    live_telemetry_include_signal_details: bool
    enable_forward_journal: bool
    forward_journal_log_path: str
    forward_journal_include_score_breakdown: bool
    enable_forward_outcome_tracker: bool
    forward_outcome_log_path: str
    forward_outcome_summary_path: str
    forward_outcome_timeframe: str
    forward_outcome_history_limit: int
    forward_outcome_sent_only: bool
    forward_outcome_max_hold_bars: int
    forward_outcome_entry_expiry_bars: int
    forward_outcome_ambiguous_policy: str
    enable_forward_performance_report: bool
    forward_performance_report_path: str
    forward_performance_sent_only: bool
    forward_performance_score_bucket_size: int
    forward_performance_min_closed_trades: int
    enable_live_heartbeat: bool
    live_heartbeat_path: str
    health_max_scan_age_minutes: int
    enable_health_alerts: bool
    health_alert_state_path: str
    health_alert_cooldown_minutes: int
    pre_trade_block_expansion_continuation: bool
    pre_trade_block_expansion_continuation_fallback: bool
    prop_base_risk: float
    prop_max_risk: float
    prop_dd_threshold_low: float
    prop_dd_threshold_mid: float
    prop_dd_threshold_high: float
    prop_loss_2_reduction: float
    prop_loss_3_reduction: float
    prop_loss_4_pause: bool
    portfolio_max_currency_exposure: int
    portfolio_max_currency_gross: int
    portfolio_correlation_threshold: float
    portfolio_max_cluster: int
    portfolio_max_net_direction: int
    gate_min_regime_tradability: int
    gate_block_transition: bool
    execution_base_slippage: float
    execution_max_multiplier: float
    enable_tick_execution: bool
    enable_realistic_slippage: bool
    enable_partial_fills: bool
    execution_latency_ticks: int
    execution_latency_ms: int
    max_slippage_pips: float
    enable_adaptive_sizing: bool
    sizing_min_multiplier: float
    sizing_max_multiplier: float
    sizing_confidence_floor_score: int
    sizing_confidence_ceiling_score: int
    enable_meta_label: bool
    meta_label_mode: str
    meta_label_probability_threshold: float
    meta_label_enable_size_adjustment: bool
    meta_label_low_probability_multiplier: float
    meta_label_high_probability_multiplier: float
    meta_label_high_probability_threshold: float
    enable_portfolio_layer: bool
    portfolio_layer_mode: str
    portfolio_layer_min_multiplier: float
    portfolio_layer_max_multiplier: float
    portfolio_layer_learning_window: int
    portfolio_layer_min_trades_per_sleeve: int
    portfolio_layer_max_sleeve_concentration: float
    enable_smc_research_features: bool
    smc_structure_scan_bars: int
    smc_structure_min_break_pips: float
    smc_structure_level_bucket_pips: float
    smc_ob_lookback_bars: int
    smc_ob_max_age_bars: int
    smc_ob_max_width_pips: float
    smc_ob_max_distance_pips: float
    smc_relaxed_fvg_lookback_bars: int
    smc_relaxed_fvg_min_gap_pips: float
    smc_relaxed_fvg_max_distance_pips: float
    enable_structure_quality_scoring: bool
    structure_quality_replaces_raw_structure_score: bool
    structure_quality_min_score_for_bonus: float
    structure_quality_max_bonus: int
    structure_quality_backtest_only: bool
    allow_live_structure_quality_scoring: bool
    structure_quality_allowed_regimes: List[str]
    structure_quality_allowed_pairs: List[str]
    structure_quality_excluded_pairs: List[str]
    enable_exit_engine: bool
    exit_profile_preset: str
    exit_use_regime_profiles: bool
    exit_profile_overrides: dict[str, dict[str, object]] | None
    exit_atr_trailing_enabled: bool
    exit_atr_trailing_period: int
    exit_atr_trailing_multiplier: float
    exit_liquidity_trailing_enabled: bool
    exit_liquidity_lookback_bars: int
    exit_liquidity_buffer_pips: float
    exit_volatility_rr_enabled: bool
    exit_volatility_rr_floor: float
    exit_volatility_rr_cap: float

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
            telegram_send_retries=max(1, int(os.getenv("TELEGRAM_SEND_RETRIES", "3"))),
            telegram_retry_base_delay_seconds=max(
                0.1,
                float(os.getenv("TELEGRAM_RETRY_BASE_DELAY_SECONDS", "1.0")),
            ),
            pairs=_parse_pairs(
                os.getenv(
                    "PAIRS",
                    "EURUSD,USDJPY",
                )
            ),
            data_source=os.getenv("DATA_SOURCE", "yahoo").strip().lower(),
            mt5_login=_parse_optional_int(os.getenv("MT5_LOGIN")),
            mt5_password=os.getenv("MT5_PASSWORD", "").strip(),
            mt5_server=os.getenv("MT5_SERVER", "").strip(),
            mt5_path=os.getenv("MT5_PATH", "C:/Program Files/MetaTrader 5/terminal64.exe").strip(),
            itick_api_key=os.getenv("ITICK_API_KEY", "").strip(),
            itick_base_url=os.getenv("ITICK_BASE_URL", "").strip(),
            itick_ohlcv_path_template=os.getenv("ITICK_OHLCV_PATH_TEMPLATE", "").strip(),
            itick_ticks_path_template=os.getenv("ITICK_TICKS_PATH_TEMPLATE", "").strip(),
            itick_api_key_header=os.getenv("ITICK_API_KEY_HEADER", "Authorization").strip(),
            itick_api_key_query_param=os.getenv("ITICK_API_KEY_QUERY_PARAM", "").strip(),
            itick_auth_scheme=os.getenv("ITICK_AUTH_SCHEME", "Bearer").strip(),
            itick_symbol_format=os.getenv("ITICK_SYMBOL_FORMAT", "{base}{quote}").strip() or "{base}{quote}",
            itick_timeout_seconds=max(1.0, float(os.getenv("ITICK_TIMEOUT_SECONDS", "10"))),
            itick_timeframe_map=_parse_json_dict(os.getenv("ITICK_TIMEFRAME_MAP_JSON")),
            itick_extra_headers=_parse_json_dict(os.getenv("ITICK_EXTRA_HEADERS_JSON")),
            enable_market_data_shadow=_parse_bool(os.getenv("ENABLE_MARKET_DATA_SHADOW", "0"), default=False),
            market_data_shadow_candidate_source=os.getenv("MARKET_DATA_SHADOW_CANDIDATE_SOURCE", "itick").strip().lower(),
            market_data_shadow_timeframes=_parse_csv_upper(os.getenv("MARKET_DATA_SHADOW_TIMEFRAMES", "M5,M15,H1")),
            market_data_shadow_log_path=os.getenv("MARKET_DATA_SHADOW_LOG_PATH", "logs/market_data_shadow.jsonl").strip(),
            market_data_shadow_cache_dir=os.getenv("MARKET_DATA_SHADOW_CACHE_DIR", "data/cache/ohlcv_shadow").strip(),
            market_data_shadow_ttl_hours=max(0.0, float(os.getenv("MARKET_DATA_SHADOW_TTL_HOURS", "0.01"))),
            market_data_shadow_compare_signals=_parse_bool(
                os.getenv("MARKET_DATA_SHADOW_COMPARE_SIGNALS", "1"),
                default=True,
            ),
            market_data_shadow_max_close_diff_pips=max(
                0.0,
                float(os.getenv("MARKET_DATA_SHADOW_MAX_CLOSE_DIFF_PIPS", "2.0")),
            ),
            market_data_shadow_max_staleness_seconds=max(
                1,
                int(os.getenv("MARKET_DATA_SHADOW_MAX_STALENESS_SECONDS", "120")),
            ),
            enable_market_data_freshness_gate=_parse_bool(
                os.getenv("ENABLE_MARKET_DATA_FRESHNESS_GATE", "0"),
                default=False,
            ),
            max_live_candle_age_seconds=max(
                60,
                int(os.getenv("MAX_LIVE_CANDLE_AGE_SECONDS", "1800")),
            ),
            scan_interval_minutes=max(1, int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))),
            enable_live_mode=_parse_bool(os.getenv("ENABLE_LIVE_MODE", "0"), default=False),
            live_mode=os.getenv("LIVE_MODE", "balanced").strip().lower(),
            enable_pair_profiles=_parse_bool(os.getenv("ENABLE_PAIR_PROFILES", "0"), default=False),
            pair_profiles=_parse_json_dict(os.getenv("PAIR_PROFILES_JSON")),
            pair_profiles_backtest_only=_parse_bool(os.getenv("PAIR_PROFILES_BACKTEST_ONLY", "1"), default=True),
            allow_live_pair_profiles=_parse_bool(os.getenv("ALLOW_LIVE_PAIR_PROFILES", "0"), default=False),
            market_data_cache_enabled=_parse_bool(os.getenv("MARKET_DATA_CACHE_ENABLED", "1"), default=True),
            market_data_cache_dir=os.getenv("MARKET_DATA_CACHE_DIR", "data/cache/ohlcv").strip(),
            market_data_cache_ttl_hours=max(0.0, float(os.getenv("MARKET_DATA_CACHE_TTL_HOURS", "12"))),
            market_data_cache_mode=os.getenv("MARKET_DATA_CACHE_MODE", "read_through").strip().lower(),
            backtest_evaluation_step=max(1, int(os.getenv("BACKTEST_EVALUATION_STEP", "1"))),
            enable_backtest_snapshot_cache=_parse_bool(os.getenv("ENABLE_BACKTEST_SNAPSHOT_CACHE", "1"), default=True),
            backtest_snapshot_cache_max_entries=max(1000, int(os.getenv("BACKTEST_SNAPSHOT_CACHE_MAX_ENTRIES", "50000"))),
            backtest_end_time=os.getenv("BACKTEST_END_TIME", "").strip(),
            enable_backtest_trade_cache=_parse_bool(os.getenv("ENABLE_BACKTEST_TRADE_CACHE", "0"), default=False),
            backtest_trade_cache_dir=os.getenv("BACKTEST_TRADE_CACHE_DIR", "data/cache/backtests").strip(),
            backtest_trade_cache_version=os.getenv("BACKTEST_TRADE_CACHE_VERSION", "trade_cache_v1").strip(),
            backtest_account_enabled=_parse_bool(os.getenv("BACKTEST_ACCOUNT_ENABLED", "1"), default=True),
            backtest_starting_balance=max(0.0, float(os.getenv("BACKTEST_STARTING_BALANCE", "1000"))),
            backtest_risk_per_trade=max(0.0, float(os.getenv("BACKTEST_RISK_PER_TRADE", "50"))),
            backtest_account_currency=os.getenv("BACKTEST_ACCOUNT_CURRENCY", "USD").strip().upper(),
            walk_forward_enabled=_parse_bool(os.getenv("WALK_FORWARD_ENABLED", "0"), default=False),
            wf_train_months=max(1, int(os.getenv("WF_TRAIN_MONTHS", "6"))),
            wf_test_months=max(1, int(os.getenv("WF_TEST_MONTHS", "1"))),
            wf_step_months=max(1, int(os.getenv("WF_STEP_MONTHS", "1"))),
            enable_realistic_execution=_parse_bool(os.getenv("ENABLE_REALISTIC_EXECUTION", "0"), default=False),
            skip_realistic_comparison=_parse_bool(os.getenv("SKIP_REALISTIC_COMPARISON", "0"), default=False),
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
            enable_strict_ltf_direction_gate=_parse_bool(
                os.getenv("ENABLE_STRICT_LTF_DIRECTION_GATE", "0"),
                default=False,
            ),
            enable_market_fallback_entry=_parse_bool(
                os.getenv("ENABLE_MARKET_FALLBACK_ENTRY", "1"),
                default=True,
            ),
            market_fallback_min_trigger_strength=max(
                0,
                min(20, int(os.getenv("MARKET_FALLBACK_MIN_TRIGGER_STRENGTH", "0"))),
            ),
            market_fallback_require_displacement=_parse_bool(
                os.getenv("MARKET_FALLBACK_REQUIRE_DISPLACEMENT", "0"),
                default=False,
            ),
            enable_pip_aware_liquidity=_parse_bool(os.getenv("ENABLE_PIP_AWARE_LIQUIDITY", "0"), default=False),
            liquidity_equal_level_tolerance_pips=max(
                0.0,
                float(os.getenv("LIQUIDITY_EQUAL_LEVEL_TOLERANCE_PIPS", "3.0")),
            ),
            liquidity_atr_tolerance_factor=max(
                0.0,
                float(os.getenv("LIQUIDITY_ATR_TOLERANCE_FACTOR", "0.0")),
            ),
            session_min_score=max(0, min(20, int(os.getenv("SESSION_MIN_SCORE", "5")))),
            enable_session_gate=_parse_bool(os.getenv("ENABLE_SESSION_GATE", "0"), default=False),
            session_gate_windows_utc=_parse_session_windows(os.getenv("SESSION_GATE_WINDOWS_UTC", "")),
            session_gate_backtest_only=_parse_bool(os.getenv("SESSION_GATE_BACKTEST_ONLY", "1"), default=True),
            allow_live_session_gate=_parse_bool(os.getenv("ALLOW_LIVE_SESSION_GATE", "0"), default=False),
            enable_regime_label_gate=_parse_bool(os.getenv("ENABLE_REGIME_LABEL_GATE", "0"), default=False),
            regime_label_blocklist=_parse_csv_upper(os.getenv("REGIME_LABEL_BLOCKLIST", "")),
            regime_gate_backtest_only=_parse_bool(os.getenv("REGIME_GATE_BACKTEST_ONLY", "1"), default=True),
            allow_live_regime_gate=_parse_bool(os.getenv("ALLOW_LIVE_REGIME_GATE", "0"), default=False),
            enable_smt_confirmation=_parse_bool(os.getenv("ENABLE_SMT_CONFIRMATION", "1"), default=True),
            smt_backtest_only=_parse_bool(os.getenv("SMT_BACKTEST_ONLY", "0"), default=False),
            allow_live_smt_confirmation=_parse_bool(os.getenv("ALLOW_LIVE_SMT_CONFIRMATION", "1"), default=True),
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
            enable_order_block_shadow=_parse_bool(os.getenv("ENABLE_ORDER_BLOCK_SHADOW", "1"), default=True),
            order_block_shadow_backtest_only=_parse_bool(
                os.getenv("ORDER_BLOCK_SHADOW_BACKTEST_ONLY", "0"),
                default=False,
            ),
            allow_live_order_block_shadow=_parse_bool(
                os.getenv("ALLOW_LIVE_ORDER_BLOCK_SHADOW", "1"),
                default=True,
            ),
            export_reports=_parse_bool(os.getenv("EXPORT_REPORTS", "1"), default=True),
            export_regime_report=_parse_bool(os.getenv("EXPORT_REGIME_REPORT", "0"), default=False),
            enable_adaptive_weights=_parse_bool(os.getenv("ENABLE_ADAPTIVE_WEIGHTS", "0"), default=False),
            adaptive_weights_preset=os.getenv("ADAPTIVE_WEIGHTS_PRESET", "default").strip().lower(),
            adaptive_regime_weights=_parse_adaptive_weights(os.getenv("ADAPTIVE_WEIGHTS_JSON")),
            enable_score_normalization=_parse_bool(os.getenv("ENABLE_SCORE_NORMALIZATION", "0"), default=False),
            score_normalization_method=os.getenv("SCORE_NORMALIZATION_METHOD", "minmax").strip().lower(),
            score_normalization_window=max(10, int(os.getenv("SCORE_NORMALIZATION_WINDOW", "200"))),
            score_normalization_scale_factor=max(0.0, float(os.getenv("SCORE_NORMALIZATION_SCALE_FACTOR", "1.0"))),
            score_normalization_backtest_only=_parse_bool(os.getenv("SCORE_NORMALIZATION_BACKTEST_ONLY", "1"), default=True),
            allow_live_score_normalization=_parse_bool(os.getenv("ALLOW_LIVE_SCORE_NORMALIZATION", "0"), default=False),
            enable_dynamic_threshold=_parse_bool(os.getenv("ENABLE_DYNAMIC_THRESHOLD", "0"), default=False),
            threshold_percentile=max(0.0, min(100.0, float(os.getenv("THRESHOLD_PERCENTILE", "80")))),
            threshold_rolling_window=max(10, int(os.getenv("THRESHOLD_ROLLING_WINDOW", "200"))),
            apply_dynamic_threshold=_parse_bool(os.getenv("APPLY_DYNAMIC_THRESHOLD", "0"), default=False),
            dynamic_threshold_backtest_only=_parse_bool(os.getenv("DYNAMIC_THRESHOLD_BACKTEST_ONLY", "1"), default=True),
            allow_live_dynamic_threshold=_parse_bool(os.getenv("ALLOW_LIVE_DYNAMIC_THRESHOLD", "0"), default=False),
            enable_feature_analytics=_parse_bool(os.getenv("ENABLE_FEATURE_ANALYTICS", "0"), default=False),
            export_meta_report=_parse_bool(os.getenv("EXPORT_META_REPORT", "0"), default=False),
            enable_regime_engine_v2=_parse_bool(os.getenv("ENABLE_REGIME_ENGINE_V2", "0"), default=False),
            enable_prop_risk_v2=_parse_bool(os.getenv("ENABLE_PROP_RISK_V2", "0"), default=False),
            enable_portfolio_risk_v2=_parse_bool(os.getenv("ENABLE_PORTFOLIO_RISK_V2", "0"), default=False),
            enable_trade_gate_v2=_parse_bool(os.getenv("ENABLE_TRADE_GATE_V2", "0"), default=False),
            enable_execution_quality_model=_parse_bool(os.getenv("ENABLE_EXECUTION_QUALITY_MODEL", "0"), default=False),
            enable_pre_trade_filter=_parse_bool(os.getenv("ENABLE_PRE_TRADE_FILTER", "0"), default=False),
            enable_pre_trade_filter_shadow=_parse_bool(
                os.getenv("ENABLE_PRE_TRADE_FILTER_SHADOW", "0"),
                default=False,
            ),
            pre_trade_filter_shadow_log_path=os.getenv(
                "PRE_TRADE_FILTER_SHADOW_LOG_PATH",
                "logs/pre_trade_filter_shadow.jsonl",
            ).strip(),
            enable_live_telemetry=_parse_bool(os.getenv("ENABLE_LIVE_TELEMETRY", "0"), default=False),
            live_telemetry_log_path=os.getenv("LIVE_TELEMETRY_LOG_PATH", "logs/live_telemetry.jsonl").strip(),
            live_telemetry_include_signal_details=_parse_bool(
                os.getenv("LIVE_TELEMETRY_INCLUDE_SIGNAL_DETAILS", "1"),
                default=True,
            ),
            enable_forward_journal=_parse_bool(os.getenv("ENABLE_FORWARD_JOURNAL", "0"), default=False),
            forward_journal_log_path=os.getenv("FORWARD_JOURNAL_LOG_PATH", "logs/forward_journal.jsonl").strip(),
            forward_journal_include_score_breakdown=_parse_bool(
                os.getenv("FORWARD_JOURNAL_INCLUDE_SCORE_BREAKDOWN", "1"),
                default=True,
            ),
            enable_forward_outcome_tracker=_parse_bool(
                os.getenv("ENABLE_FORWARD_OUTCOME_TRACKER", "0"),
                default=False,
            ),
            forward_outcome_log_path=os.getenv("FORWARD_OUTCOME_LOG_PATH", "logs/forward_outcomes.jsonl").strip(),
            forward_outcome_summary_path=os.getenv(
                "FORWARD_OUTCOME_SUMMARY_PATH",
                "reports/forward_outcomes_summary.json",
            ).strip(),
            forward_outcome_timeframe=os.getenv("FORWARD_OUTCOME_TIMEFRAME", "M15").strip().upper(),
            forward_outcome_history_limit=max(50, int(os.getenv("FORWARD_OUTCOME_HISTORY_LIMIT", "1500"))),
            forward_outcome_sent_only=_parse_bool(os.getenv("FORWARD_OUTCOME_SENT_ONLY", "0"), default=False),
            forward_outcome_max_hold_bars=max(1, int(os.getenv("FORWARD_OUTCOME_MAX_HOLD_BARS", "48"))),
            forward_outcome_entry_expiry_bars=max(0, int(os.getenv("FORWARD_OUTCOME_ENTRY_EXPIRY_BARS", "0"))),
            forward_outcome_ambiguous_policy=os.getenv(
                "FORWARD_OUTCOME_AMBIGUOUS_POLICY",
                "ambiguous",
            ).strip().lower(),
            enable_forward_performance_report=_parse_bool(
                os.getenv("ENABLE_FORWARD_PERFORMANCE_REPORT", "0"),
                default=False,
            ),
            forward_performance_report_path=os.getenv(
                "FORWARD_PERFORMANCE_REPORT_PATH",
                "reports/forward_performance_report.json",
            ).strip(),
            forward_performance_sent_only=_parse_bool(
                os.getenv("FORWARD_PERFORMANCE_SENT_ONLY", "0"),
                default=False,
            ),
            forward_performance_score_bucket_size=max(
                1,
                int(os.getenv("FORWARD_PERFORMANCE_SCORE_BUCKET_SIZE", "5")),
            ),
            forward_performance_min_closed_trades=max(
                0,
                int(os.getenv("FORWARD_PERFORMANCE_MIN_CLOSED_TRADES", "0")),
            ),
            enable_live_heartbeat=_parse_bool(os.getenv("ENABLE_LIVE_HEARTBEAT", "1"), default=True),
            live_heartbeat_path=os.getenv("LIVE_HEARTBEAT_PATH", "logs/live_heartbeat.json").strip(),
            health_max_scan_age_minutes=max(1, int(os.getenv("HEALTH_MAX_SCAN_AGE_MINUTES", "15"))),
            enable_health_alerts=_parse_bool(os.getenv("ENABLE_HEALTH_ALERTS", "0"), default=False),
            health_alert_state_path=os.getenv(
                "HEALTH_ALERT_STATE_PATH",
                "logs/live_health_alert_state.json",
            ).strip(),
            health_alert_cooldown_minutes=max(1, int(os.getenv("HEALTH_ALERT_COOLDOWN_MINUTES", "60"))),
            pre_trade_block_expansion_continuation=_parse_bool(
                os.getenv("PRE_TRADE_BLOCK_EXPANSION_CONTINUATION", "0"),
                default=False,
            ),
            pre_trade_block_expansion_continuation_fallback=_parse_bool(
                os.getenv("PRE_TRADE_BLOCK_EXPANSION_CONTINUATION_FALLBACK", "0"),
                default=False,
            ),
            prop_base_risk=max(0.1, float(os.getenv("PROP_BASE_RISK", "1.0"))),
            prop_max_risk=max(0.1, float(os.getenv("PROP_MAX_RISK", "2.0"))),
            prop_dd_threshold_low=max(0.0, float(os.getenv("PROP_DD_THRESHOLD_LOW", "3.0"))),
            prop_dd_threshold_mid=max(0.0, float(os.getenv("PROP_DD_THRESHOLD_MID", "6.0"))),
            prop_dd_threshold_high=max(0.0, float(os.getenv("PROP_DD_THRESHOLD_HIGH", "10.0"))),
            prop_loss_2_reduction=max(0.0, min(1.0, float(os.getenv("PROP_LOSS_2_REDUCTION", "0.8")))),
            prop_loss_3_reduction=max(0.0, min(1.0, float(os.getenv("PROP_LOSS_3_REDUCTION", "0.6")))),
            prop_loss_4_pause=_parse_bool(os.getenv("PROP_LOSS_4_PAUSE", "1"), default=True),
            portfolio_max_currency_exposure=max(0, int(os.getenv("PORTFOLIO_MAX_CURRENCY_EXPOSURE", "2"))),
            portfolio_max_currency_gross=max(0, int(os.getenv("PORTFOLIO_MAX_CURRENCY_GROSS", "4"))),
            portfolio_correlation_threshold=max(0.0, min(0.99, float(os.getenv("PORTFOLIO_CORRELATION_THRESHOLD", "0.82")))),
            portfolio_max_cluster=max(0, int(os.getenv("PORTFOLIO_MAX_CLUSTER", "3"))),
            portfolio_max_net_direction=max(0, int(os.getenv("PORTFOLIO_MAX_NET_DIRECTION", "4"))),
            gate_min_regime_tradability=max(0, min(100, int(os.getenv("GATE_MIN_REGIME_TRADABILITY", "30")))),
            gate_block_transition=_parse_bool(os.getenv("GATE_BLOCK_TRANSITION", "1"), default=True),
            execution_base_slippage=max(0.0, float(os.getenv("EXECUTION_BASE_SLIPPAGE", "0.5"))),
            execution_max_multiplier=max(1.0, float(os.getenv("EXECUTION_MAX_MULTIPLIER", "2.0"))),
            enable_tick_execution=_parse_bool(os.getenv("ENABLE_TICK_EXECUTION", "0"), default=False),
            enable_realistic_slippage=_parse_bool(os.getenv("ENABLE_REALISTIC_SLIPPAGE", "0"), default=False),
            enable_partial_fills=_parse_bool(os.getenv("ENABLE_PARTIAL_FILLS", "0"), default=False),
            execution_latency_ticks=max(0, int(os.getenv("EXECUTION_LATENCY_TICKS", "0"))),
            execution_latency_ms=max(0, int(os.getenv("EXECUTION_LATENCY_MS", "0"))),
            enable_adaptive_sizing=_parse_bool(os.getenv("ENABLE_ADAPTIVE_SIZING", "0"), default=False),
            sizing_min_multiplier=max(0.05, float(os.getenv("SIZING_MIN_MULTIPLIER", "0.40"))),
            sizing_max_multiplier=max(0.05, float(os.getenv("SIZING_MAX_MULTIPLIER", "1.50"))),
            sizing_confidence_floor_score=max(0, min(100, int(os.getenv("SIZING_CONFIDENCE_FLOOR_SCORE", "65")))),
            sizing_confidence_ceiling_score=max(1, min(100, int(os.getenv("SIZING_CONFIDENCE_CEILING_SCORE", "90")))),
            enable_meta_label=_parse_bool(os.getenv("ENABLE_META_LABEL", "0"), default=False),
            meta_label_mode=os.getenv("META_LABEL_MODE", "analysis_only").strip().lower(),
            meta_label_probability_threshold=max(0.0, min(1.0, float(os.getenv("META_LABEL_PROBABILITY_THRESHOLD", "0.55")))),
            meta_label_enable_size_adjustment=_parse_bool(os.getenv("META_LABEL_ENABLE_SIZE_ADJUSTMENT", "0"), default=False),
            meta_label_low_probability_multiplier=max(0.05, float(os.getenv("META_LABEL_LOW_PROBABILITY_MULTIPLIER", "0.75"))),
            meta_label_high_probability_multiplier=max(0.05, float(os.getenv("META_LABEL_HIGH_PROBABILITY_MULTIPLIER", "1.10"))),
            meta_label_high_probability_threshold=max(0.0, min(1.0, float(os.getenv("META_LABEL_HIGH_PROBABILITY_THRESHOLD", "0.72")))),
            enable_portfolio_layer=_parse_bool(os.getenv("ENABLE_PORTFOLIO_LAYER", "0"), default=False),
            portfolio_layer_mode=os.getenv("PORTFOLIO_LAYER_MODE", "analysis_only").strip().lower(),
            portfolio_layer_min_multiplier=max(0.05, float(os.getenv("PORTFOLIO_LAYER_MIN_MULTIPLIER", "0.70"))),
            portfolio_layer_max_multiplier=max(0.05, float(os.getenv("PORTFOLIO_LAYER_MAX_MULTIPLIER", "1.25"))),
            portfolio_layer_learning_window=max(5, int(os.getenv("PORTFOLIO_LAYER_LEARNING_WINDOW", "30"))),
            portfolio_layer_min_trades_per_sleeve=max(1, int(os.getenv("PORTFOLIO_LAYER_MIN_TRADES_PER_SLEEVE", "5"))),
            portfolio_layer_max_sleeve_concentration=max(0.10, min(0.95, float(os.getenv("PORTFOLIO_LAYER_MAX_SLEEVE_CONCENTRATION", "0.55")))),
            enable_smc_research_features=_parse_bool(os.getenv("ENABLE_SMC_RESEARCH_FEATURES", "0"), default=False),
            smc_structure_scan_bars=max(80, int(os.getenv("SMC_STRUCTURE_SCAN_BARS", "300"))),
            smc_structure_min_break_pips=max(0.0, float(os.getenv("SMC_STRUCTURE_MIN_BREAK_PIPS", "2.0"))),
            smc_structure_level_bucket_pips=max(0.1, float(os.getenv("SMC_STRUCTURE_LEVEL_BUCKET_PIPS", "2.0"))),
            smc_ob_lookback_bars=max(50, int(os.getenv("SMC_OB_LOOKBACK_BARS", "300"))),
            smc_ob_max_age_bars=max(1, int(os.getenv("SMC_OB_MAX_AGE_BARS", "300"))),
            smc_ob_max_width_pips=max(0.1, float(os.getenv("SMC_OB_MAX_WIDTH_PIPS", "20.0"))),
            smc_ob_max_distance_pips=max(0.1, float(os.getenv("SMC_OB_MAX_DISTANCE_PIPS", "30.0"))),
            smc_relaxed_fvg_lookback_bars=max(50, int(os.getenv("SMC_RELAXED_FVG_LOOKBACK_BARS", "300"))),
            smc_relaxed_fvg_min_gap_pips=max(0.0, float(os.getenv("SMC_RELAXED_FVG_MIN_GAP_PIPS", "0.1"))),
            smc_relaxed_fvg_max_distance_pips=max(0.1, float(os.getenv("SMC_RELAXED_FVG_MAX_DISTANCE_PIPS", "30.0"))),
            enable_structure_quality_scoring=_parse_bool(os.getenv("ENABLE_STRUCTURE_QUALITY_SCORING", "0"), default=False),
            structure_quality_replaces_raw_structure_score=_parse_bool(
                os.getenv("STRUCTURE_QUALITY_REPLACES_RAW_STRUCTURE_SCORE", "0"),
                default=False,
            ),
            structure_quality_min_score_for_bonus=max(
                0.0,
                min(100.0, float(os.getenv("STRUCTURE_QUALITY_MIN_SCORE_FOR_BONUS", "60.0"))),
            ),
            structure_quality_max_bonus=max(0, min(20, int(os.getenv("STRUCTURE_QUALITY_MAX_BONUS", "8")))),
            structure_quality_backtest_only=_parse_bool(os.getenv("STRUCTURE_QUALITY_BACKTEST_ONLY", "1"), default=True),
            allow_live_structure_quality_scoring=_parse_bool(
                os.getenv("ALLOW_LIVE_STRUCTURE_QUALITY_SCORING", "0"),
                default=False,
            ),
            structure_quality_allowed_regimes=_parse_csv_upper(os.getenv("STRUCTURE_QUALITY_ALLOWED_REGIMES", "")),
            structure_quality_allowed_pairs=_parse_pairs(os.getenv("STRUCTURE_QUALITY_ALLOWED_PAIRS", "")) if os.getenv("STRUCTURE_QUALITY_ALLOWED_PAIRS", "").strip() else [],
            structure_quality_excluded_pairs=_parse_pairs(os.getenv("STRUCTURE_QUALITY_EXCLUDED_PAIRS", "")) if os.getenv("STRUCTURE_QUALITY_EXCLUDED_PAIRS", "").strip() else [],
            enable_exit_engine=_parse_bool(os.getenv("ENABLE_EXIT_ENGINE", "1"), default=True),
            exit_profile_preset=os.getenv("EXIT_PROFILE_PRESET", "m15_vol_liq_v1").strip().lower(),
            exit_use_regime_profiles=_parse_bool(os.getenv("EXIT_USE_REGIME_PROFILES", "1"), default=True),
            exit_profile_overrides=_parse_object_map(os.getenv("EXIT_PROFILE_OVERRIDES_JSON")),
            exit_atr_trailing_enabled=_parse_bool(os.getenv("EXIT_ATR_TRAILING_ENABLED", "0"), default=False),
            exit_atr_trailing_period=max(2, int(os.getenv("EXIT_ATR_TRAILING_PERIOD", "14"))),
            exit_atr_trailing_multiplier=max(0.1, float(os.getenv("EXIT_ATR_TRAILING_MULTIPLIER", "1.5"))),
            exit_liquidity_trailing_enabled=_parse_bool(os.getenv("EXIT_LIQUIDITY_TRAILING_ENABLED", "1"), default=True),
            exit_liquidity_lookback_bars=max(2, int(os.getenv("EXIT_LIQUIDITY_LOOKBACK_BARS", "8"))),
            exit_liquidity_buffer_pips=max(0.0, float(os.getenv("EXIT_LIQUIDITY_BUFFER_PIPS", "1.0"))),
            exit_volatility_rr_enabled=_parse_bool(os.getenv("EXIT_VOLATILITY_RR_ENABLED", "1"), default=True),
            exit_volatility_rr_floor=max(0.1, float(os.getenv("EXIT_VOLATILITY_RR_FLOOR", "0.75"))),
            exit_volatility_rr_cap=max(0.1, float(os.getenv("EXIT_VOLATILITY_RR_CAP", "1.40"))),
        )
