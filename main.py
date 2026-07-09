from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from core.pair_profiles import PairRuntimeProfile, build_pair_runtime_profiles, clean_pair
from core.signal_engine import SignalEngine
from data.market_data import MarketDataCacheConfig, MarketDataClient, MarketDataDiagnosticsConfig
from execution.news import NewsFilter
from services.forward_journal import ForwardJournalSettings, ForwardSignalJournal
from services.itick_websocket_shadow import ItickWebSocketShadowClient, ItickWebSocketShadowSettings
from services.live_bar_builder import LiveBarBuilder, LiveBarBuilderSettings
from services.live_health import LiveHeartbeatSettings, LiveHeartbeatWriter
from services.live_telemetry import LiveTelemetryLogger, LiveTelemetrySettings
from services.market_data_shadow import MarketDataShadowLogger, MarketDataShadowSettings
from services.pretrade_shadow import PreTradeShadowLogger, PreTradeShadowSettings
from services.telegram import TelegramSignalService


LIVE_PROFILE_PAIRS = ("EURUSD", "EURJPY", "CADJPY")
LIVE_MODE_BALANCED_PAIRS = ("EURUSD", "EURJPY", "CADJPY")
LIVE_MODE_AGGRESSIVE_PAIRS = ("EURUSD", "EURJPY", "CADJPY")
LIVE_MODE_CONSERVATIVE_PAIRS = ("EURUSD",)
LIVE_MODE_MIN_SCORE = 80
LIVE_MODE_AGGRESSIVE_MIN_SCORE = 78
LIVE_MODE_REGIME_BLOCKLIST = ("TREND",)
LIVE_MODE_BALANCED_SESSION_UTC = ((7, 16),)
LIVE_MODE_AGGRESSIVE_SESSION_UTC = ((7, 16),)
LIVE_MODE_CONSERVATIVE_SESSION_UTC = ((12, 16),)


@dataclass(frozen=True)
class EffectiveLiveMode:
    enabled: bool
    name: str
    pairs: tuple[str, ...]
    min_score: int
    enable_session_gate: bool
    session_gate_windows_utc: tuple[tuple[int, int], ...]
    allow_live_session_gate: bool
    enable_regime_label_gate: bool
    regime_label_blocklist: tuple[str, ...]
    allow_live_regime_gate: bool
    description: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _itick_config_from_settings(settings: Settings) -> dict[str, object]:
    return {
        "api_key": settings.itick_api_key,
        "base_url": settings.itick_base_url,
        "ohlcv_path_template": settings.itick_ohlcv_path_template,
        "ticks_path_template": settings.itick_ticks_path_template,
        "api_key_header": settings.itick_api_key_header,
        "api_key_query_param": settings.itick_api_key_query_param,
        "auth_scheme": settings.itick_auth_scheme,
        "symbol_format": settings.itick_symbol_format,
        "timeout_seconds": settings.itick_timeout_seconds,
        "timeframe_map": settings.itick_timeframe_map,
        "extra_headers": settings.itick_extra_headers,
    }


def _live_bar_config_from_settings(settings: Settings) -> dict[str, object]:
    return {
        "bars_dir": settings.live_bar_provider_dir,
        "fallback_source": settings.live_bar_provider_fallback_source,
        "include_current_bar": settings.live_bar_provider_include_current_bar,
        "require_live_overlay": settings.live_bar_provider_require_live_overlay,
        "max_live_bar_age_seconds": settings.live_bar_provider_max_live_bar_age_seconds,
        "itick_config": _itick_config_from_settings(settings),
    }


def _redundant_config_from_settings(settings: Settings) -> dict[str, object]:
    return {
        "primary_source": settings.market_data_redundancy_primary_source,
        "backup_sources": settings.market_data_redundancy_backup_sources,
        "require_fresh": settings.market_data_redundancy_require_fresh,
        "max_candle_age_seconds": settings.market_data_redundancy_max_candle_age_seconds,
        "fail_closed": settings.market_data_redundancy_fail_closed,
        "log_path": settings.market_data_redundancy_log_path,
        "itick_config": _itick_config_from_settings(settings),
        "live_bar_config": _live_bar_config_from_settings(settings),
        "mt5_config": {
            "login": settings.mt5_login,
            "password": settings.mt5_password,
            "server": settings.mt5_server,
            "path": settings.mt5_path,
        },
    }


def _build_market_data(
    settings: Settings,
    *,
    data_source: str | None = None,
    cache_dir: str | Path | None = None,
    ttl_hours: float | None = None,
) -> MarketDataClient:
    source = (data_source or settings.data_source).strip().lower()
    cache_enabled = settings.market_data_cache_enabled
    cache_mode = settings.market_data_cache_mode
    if source in {"live_bars", "redundant"}:
        cache_enabled = False
        cache_mode = "disabled"

    return MarketDataClient(
        history_limit=settings.history_limit,
        data_source=source,
        mt5_login=settings.mt5_login,
        mt5_password=settings.mt5_password,
        mt5_server=settings.mt5_server,
        mt5_path=settings.mt5_path,
        itick_config=_itick_config_from_settings(settings),
        live_bar_config=_live_bar_config_from_settings(settings),
        redundant_config=_redundant_config_from_settings(settings),
        cache_config=MarketDataCacheConfig(
            enabled=cache_enabled,
            cache_dir=cache_dir or settings.market_data_cache_dir,
            ttl_hours=settings.market_data_cache_ttl_hours if ttl_hours is None else ttl_hours,
            mode=cache_mode,
        ),
        diagnostics_config=MarketDataDiagnosticsConfig(
            enabled=settings.enable_market_data_diagnostics,
            log_path=settings.market_data_diagnostics_log_path,
            max_latency_seconds=settings.market_data_diagnostics_max_latency_seconds,
            max_candle_age_seconds=settings.market_data_diagnostics_max_candle_age_seconds,
            log_cache_hits=settings.market_data_diagnostics_log_cache_hits,
        ),
    )


def _legacy_live_pairs(settings: Settings) -> tuple[str, ...]:
    if settings.enable_exit_engine and settings.exit_profile_preset == "m15_vol_liq_v1":
        return LIVE_PROFILE_PAIRS
    return tuple(settings.pairs)


def _effective_live_mode(settings: Settings) -> EffectiveLiveMode:
    legacy_pairs = _legacy_live_pairs(settings)
    if not settings.enable_live_mode:
        return EffectiveLiveMode(
            enabled=False,
            name="legacy",
            pairs=legacy_pairs,
            min_score=settings.min_score,
            enable_session_gate=settings.enable_session_gate,
            session_gate_windows_utc=tuple(settings.session_gate_windows_utc),
            allow_live_session_gate=settings.allow_live_session_gate,
            enable_regime_label_gate=settings.enable_regime_label_gate,
            regime_label_blocklist=tuple(settings.regime_label_blocklist),
            allow_live_regime_gate=settings.allow_live_regime_gate,
            description="legacy settings from env",
        )

    mode = settings.live_mode.strip().lower()
    if mode == "balanced":
        return EffectiveLiveMode(
            enabled=True,
            name="balanced",
            pairs=LIVE_MODE_BALANCED_PAIRS,
            min_score=LIVE_MODE_MIN_SCORE,
            enable_session_gate=True,
            session_gate_windows_utc=LIVE_MODE_BALANCED_SESSION_UTC,
            allow_live_session_gate=True,
            enable_regime_label_gate=True,
            regime_label_blocklist=LIVE_MODE_REGIME_BLOCKLIST,
            allow_live_regime_gate=True,
            description="EURUSD+EURJPY+CADJPY 07-16 UTC score>=80 block TREND",
        )
    if mode == "aggressive":
        return EffectiveLiveMode(
            enabled=True,
            name="aggressive",
            pairs=LIVE_MODE_AGGRESSIVE_PAIRS,
            min_score=LIVE_MODE_AGGRESSIVE_MIN_SCORE,
            enable_session_gate=True,
            session_gate_windows_utc=LIVE_MODE_AGGRESSIVE_SESSION_UTC,
            allow_live_session_gate=True,
            enable_regime_label_gate=True,
            regime_label_blocklist=LIVE_MODE_REGIME_BLOCKLIST,
            allow_live_regime_gate=True,
            description="EURUSD+EURJPY+CADJPY 07-16 UTC score>=78 block TREND",
        )
    if mode == "conservative":
        return EffectiveLiveMode(
            enabled=True,
            name="conservative",
            pairs=LIVE_MODE_CONSERVATIVE_PAIRS,
            min_score=LIVE_MODE_MIN_SCORE,
            enable_session_gate=True,
            session_gate_windows_utc=LIVE_MODE_CONSERVATIVE_SESSION_UTC,
            allow_live_session_gate=True,
            enable_regime_label_gate=True,
            regime_label_blocklist=LIVE_MODE_REGIME_BLOCKLIST,
            allow_live_regime_gate=True,
            description="EURUSD 12-16 UTC score>=80 block TREND",
        )

    logging.getLogger("engine").warning(
        "Unknown LIVE_MODE=%s; falling back to legacy settings", settings.live_mode
    )
    return EffectiveLiveMode(
        enabled=False,
        name="legacy",
        pairs=legacy_pairs,
        min_score=settings.min_score,
        enable_session_gate=settings.enable_session_gate,
        session_gate_windows_utc=tuple(settings.session_gate_windows_utc),
        allow_live_session_gate=settings.allow_live_session_gate,
        enable_regime_label_gate=settings.enable_regime_label_gate,
        regime_label_blocklist=tuple(settings.regime_label_blocklist),
        allow_live_regime_gate=settings.allow_live_regime_gate,
        description="legacy fallback after invalid LIVE_MODE",
    )


def _format_windows(windows: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{start:02d}-{end:02d}" for start, end in windows) or "-"


def _live_mode_pair_profile_payload(live_mode: EffectiveLiveMode) -> dict[str, dict[str, object]]:
    if not live_mode.enabled:
        return {}
    return {
        pair: {
            "min_score": live_mode.min_score,
            "session_windows_utc": [f"{start:02d}-{end:02d}" for start, end in live_mode.session_gate_windows_utc],
            "regime_blocklist": list(live_mode.regime_label_blocklist),
            "description": live_mode.description,
        }
        for pair in live_mode.pairs
    }


def _build_live_pair_profiles(settings: Settings, live_mode: EffectiveLiveMode) -> dict[str, PairRuntimeProfile]:
    if settings.enable_pair_profiles and settings.pair_profiles:
        if not settings.allow_live_pair_profiles:
            logging.getLogger("engine").warning(
                "ENABLE_PAIR_PROFILES=1 but ALLOW_LIVE_PAIR_PROFILES=0; custom pair profiles are ignored in live"
            )
            return {}
        return build_pair_runtime_profiles(
            settings.pair_profiles,
            enabled=True,
            session_backtest_only=settings.pair_profiles_backtest_only,
            allow_live_session=True,
            regime_backtest_only=settings.pair_profiles_backtest_only,
            allow_live_regime=True,
        )

    return build_pair_runtime_profiles(
        _live_mode_pair_profile_payload(live_mode),
        enabled=live_mode.enabled,
        session_backtest_only=True,
        allow_live_session=True,
        regime_backtest_only=True,
        allow_live_regime=True,
    )


def _build_signal_engine(
    settings: Settings,
    market_data: MarketDataClient,
    news_filter: NewsFilter,
    *,
    live_mode: EffectiveLiveMode | None = None,
    pair_profiles: dict[str, PairRuntimeProfile] | None = None,
) -> SignalEngine:
    mode = live_mode or _effective_live_mode(settings)
    return SignalEngine(
        market_data=market_data,
        news_filter=news_filter,
        htf_timeframe=settings.htf_timeframe,
        ltf_timeframe=settings.ltf_timeframe,
        trigger_timeframe=settings.trigger_timeframe,
        min_score=mode.min_score,
        risk_reward=settings.risk_reward,
        swing_window=settings.swing_window,
        pair_correlation_threshold=settings.pair_correlation_threshold,
        correlation_lookback=settings.correlation_lookback,
        currency_exposure_cap=settings.currency_exposure_cap,
        portfolio_currency_gross_cap=settings.portfolio_currency_gross_cap,
        portfolio_currency_net_cap=settings.portfolio_currency_net_cap,
        portfolio_exposure_window_minutes=settings.portfolio_exposure_window_minutes,
        pair_cooldown_minutes=settings.pair_cooldown_minutes,
        max_entries_per_bias=settings.max_entries_per_bias,
        bias_window_minutes=settings.bias_window_minutes,
        regime_opposition_confidence=settings.regime_opposition_confidence,
        contraction_min_trigger_strength=settings.contraction_min_trigger_strength,
        range_min_trigger_strength=settings.range_min_trigger_strength,
        require_displacement_in_contraction=settings.require_displacement_in_contraction,
        enable_strict_ltf_direction_gate=settings.enable_strict_ltf_direction_gate,
        enable_market_fallback_entry=settings.enable_market_fallback_entry,
        market_fallback_min_trigger_strength=settings.market_fallback_min_trigger_strength,
        market_fallback_require_displacement=settings.market_fallback_require_displacement,
        enable_pip_aware_liquidity=settings.enable_pip_aware_liquidity,
        liquidity_equal_level_tolerance_pips=settings.liquidity_equal_level_tolerance_pips,
        liquidity_atr_tolerance_factor=settings.liquidity_atr_tolerance_factor,
        session_min_score=settings.session_min_score,
        enable_session_gate=mode.enable_session_gate,
        session_gate_windows_utc=mode.session_gate_windows_utc,
        session_gate_backtest_only=settings.session_gate_backtest_only,
        allow_live_session_gate=mode.allow_live_session_gate,
        enable_regime_label_gate=mode.enable_regime_label_gate,
        regime_label_blocklist=mode.regime_label_blocklist,
        regime_gate_backtest_only=settings.regime_gate_backtest_only,
        allow_live_regime_gate=mode.allow_live_regime_gate,
        enable_smt_confirmation=settings.enable_smt_confirmation,
        smt_backtest_only=settings.smt_backtest_only,
        allow_live_smt_confirmation=settings.allow_live_smt_confirmation,
        smt_hard_gate=settings.smt_hard_gate,
        smt_min_strength=settings.smt_min_strength,
        smt_opposite_block_strength=settings.smt_opposite_block_strength,
        smt_reference_map=settings.smt_reference_map,
        partial_tp_enabled=settings.partial_tp_enabled,
        partial_tp_r=settings.partial_tp_r,
        partial_tp_fraction=settings.partial_tp_fraction,
        break_even_r=settings.break_even_r,
        trailing_enabled=settings.trailing_enabled,
        trailing_start_r=settings.trailing_start_r,
        trailing_lookback_bars=settings.trailing_lookback_bars,
        time_stop_bars=settings.time_stop_bars,
        regime_short_window=settings.regime_short_window,
        regime_long_window=settings.regime_long_window,
        enable_mitigation_entry=settings.enable_mitigation_entry,
        enable_order_block_shadow=settings.enable_order_block_shadow,
        order_block_shadow_backtest_only=settings.order_block_shadow_backtest_only,
        allow_live_order_block_shadow=settings.allow_live_order_block_shadow,
        enable_adaptive_weights=settings.enable_adaptive_weights,
        adaptive_weights_preset=settings.adaptive_weights_preset,
        adaptive_regime_weights=settings.adaptive_regime_weights,
        enable_score_normalization=settings.enable_score_normalization,
        score_normalization_method=settings.score_normalization_method,
        score_normalization_window=settings.score_normalization_window,
        score_normalization_scale_factor=settings.score_normalization_scale_factor,
        score_normalization_backtest_only=settings.score_normalization_backtest_only,
        allow_live_score_normalization=settings.allow_live_score_normalization,
        runtime_mode="live",
        enable_market_data_freshness_gate=settings.enable_market_data_freshness_gate,
        max_live_candle_age_seconds=settings.max_live_candle_age_seconds,
        enable_dynamic_threshold=settings.enable_dynamic_threshold,
        threshold_percentile=settings.threshold_percentile,
        threshold_rolling_window=settings.threshold_rolling_window,
        apply_dynamic_threshold=settings.apply_dynamic_threshold,
        dynamic_threshold_backtest_only=settings.dynamic_threshold_backtest_only,
        allow_live_dynamic_threshold=settings.allow_live_dynamic_threshold,
        enable_structure_quality_scoring=settings.enable_structure_quality_scoring,
        structure_quality_replaces_raw_structure_score=settings.structure_quality_replaces_raw_structure_score,
        structure_quality_scan_bars=settings.smc_structure_scan_bars,
        structure_quality_min_break_pips=settings.smc_structure_min_break_pips,
        structure_quality_level_bucket_pips=settings.smc_structure_level_bucket_pips,
        structure_quality_min_score_for_bonus=settings.structure_quality_min_score_for_bonus,
        structure_quality_max_bonus=settings.structure_quality_max_bonus,
        structure_quality_backtest_only=settings.structure_quality_backtest_only,
        allow_live_structure_quality_scoring=settings.allow_live_structure_quality_scoring,
        structure_quality_allowed_regimes=settings.structure_quality_allowed_regimes,
        structure_quality_allowed_pairs=settings.structure_quality_allowed_pairs,
        structure_quality_excluded_pairs=settings.structure_quality_excluded_pairs,
        live_exit_profile_enabled=settings.enable_exit_engine,
        live_exit_profile_preset=settings.exit_profile_preset,
        live_exit_use_regime_profiles=settings.exit_use_regime_profiles,
        live_exit_volatility_rr_enabled=settings.exit_volatility_rr_enabled,
        live_exit_volatility_rr_floor=settings.exit_volatility_rr_floor,
        live_exit_volatility_rr_cap=settings.exit_volatility_rr_cap,
        live_exit_liquidity_trailing_enabled=settings.exit_liquidity_trailing_enabled,
        live_exit_liquidity_lookback_bars=settings.exit_liquidity_lookback_bars,
        live_exit_liquidity_buffer_pips=settings.exit_liquidity_buffer_pips,
        pair_runtime_profiles=pair_profiles,
    )


async def run_engine() -> None:
    settings = Settings.from_env()
    live_mode = _effective_live_mode(settings)
    pair_profiles = _build_live_pair_profiles(settings, live_mode)
    live_pairs = list(pair_profiles.keys()) if settings.enable_pair_profiles and settings.allow_live_pair_profiles and pair_profiles else list(live_mode.pairs)
    live_pairs = [clean_pair(pair) for pair in live_pairs]

    market_data = _build_market_data(settings)
    news_filter = NewsFilter(
        blackout_before_minutes=settings.news_blackout_before_minutes,
        blackout_after_minutes=settings.news_blackout_after_minutes,
        surprise_threshold=settings.news_surprise_threshold,
    )
    engine = _build_signal_engine(settings, market_data, news_filter, live_mode=live_mode, pair_profiles=pair_profiles)
    shadow_logger: MarketDataShadowLogger | None = None
    shadow_engine: SignalEngine | None = None
    shadow_market_data: MarketDataClient | None = None
    itick_websocket_shadow: ItickWebSocketShadowClient | None = None
    live_bar_builder: LiveBarBuilder | None = None
    telemetry = LiveTelemetryLogger(
        LiveTelemetrySettings(
            enabled=settings.enable_live_telemetry,
            log_path=settings.live_telemetry_log_path,
            include_signal_details=settings.live_telemetry_include_signal_details,
        )
    )
    forward_journal = ForwardSignalJournal(
        ForwardJournalSettings(
            enabled=settings.enable_forward_journal,
            log_path=settings.forward_journal_log_path,
            include_score_breakdown=settings.forward_journal_include_score_breakdown,
        )
    )
    heartbeat = LiveHeartbeatWriter(
        LiveHeartbeatSettings(
            enabled=settings.enable_live_heartbeat,
            path=settings.live_heartbeat_path,
        )
    )
    pre_trade_shadow: PreTradeShadowLogger | None = None
    if settings.enable_pre_trade_filter_shadow:
        pre_trade_shadow = PreTradeShadowLogger(
            PreTradeShadowSettings(
                enabled=True,
                log_path=settings.pre_trade_filter_shadow_log_path,
                block_expansion_continuation=settings.pre_trade_block_expansion_continuation,
                block_expansion_continuation_fallback=settings.pre_trade_block_expansion_continuation_fallback,
            )
        )
    if settings.enable_market_data_shadow:
        shadow_cache_dir = Path(settings.market_data_shadow_cache_dir) / settings.market_data_shadow_candidate_source
        shadow_market_data = _build_market_data(
            settings,
            data_source=settings.market_data_shadow_candidate_source,
            cache_dir=shadow_cache_dir,
            ttl_hours=settings.market_data_shadow_ttl_hours,
        )
        shadow_logger = MarketDataShadowLogger(
            MarketDataShadowSettings(
                enabled=True,
                primary_source=settings.data_source,
                candidate_source=settings.market_data_shadow_candidate_source,
                timeframes=tuple(settings.market_data_shadow_timeframes),
                log_path=settings.market_data_shadow_log_path,
                max_close_diff_pips=settings.market_data_shadow_max_close_diff_pips,
                max_staleness_seconds=settings.market_data_shadow_max_staleness_seconds,
                compare_signals=settings.market_data_shadow_compare_signals,
            ),
            primary_client=market_data,
            candidate_client=shadow_market_data,
        )
        if settings.market_data_shadow_compare_signals:
            shadow_engine = _build_signal_engine(
                settings,
                shadow_market_data,
                news_filter,
                live_mode=live_mode,
                pair_profiles=pair_profiles,
            )
    if settings.enable_live_bar_builder:
        live_bar_builder = LiveBarBuilder(
            LiveBarBuilderSettings(
                enabled=True,
                source="itick_websocket",
                timeframes=tuple(settings.live_bar_builder_timeframes),
                bars_dir=settings.live_bar_builder_dir,
                log_path=settings.live_bar_builder_log_path,
                max_bars_per_timeframe=settings.live_bar_builder_max_bars,
                flush_interval_seconds=settings.live_bar_builder_flush_seconds,
                max_quote_age_seconds=settings.live_bar_builder_max_quote_age_seconds,
            )
        )
    if settings.enable_itick_websocket_shadow or settings.enable_live_bar_builder:
        itick_websocket_shadow = ItickWebSocketShadowClient(
            ItickWebSocketShadowSettings(
                enabled=True,
                api_key=settings.itick_api_key,
                url=settings.itick_websocket_url,
                api_key_header=settings.itick_api_key_header,
                auth_scheme=settings.itick_auth_scheme,
                symbol_format=settings.itick_symbol_format,
                region=settings.itick_websocket_region,
                subscription_types=settings.itick_websocket_types,
                log_path=settings.itick_websocket_log_path,
                heartbeat_seconds=settings.itick_websocket_heartbeat_seconds,
                reconnect_seconds=settings.itick_websocket_reconnect_seconds,
                max_reconnect_seconds=settings.itick_websocket_max_reconnect_seconds,
                reconnect_backoff_factor=settings.itick_websocket_reconnect_backoff_factor,
                reconnect_jitter_seconds=settings.itick_websocket_reconnect_jitter_seconds,
                stale_seconds=settings.itick_websocket_stale_seconds,
                max_latency_seconds=settings.itick_websocket_max_latency_seconds,
            ),
            quote_consumers=[live_bar_builder.on_quote] if live_bar_builder is not None else None,
        )
    telegram = TelegramSignalService(
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        send_retries=settings.telegram_send_retries,
        retry_base_delay_seconds=settings.telegram_retry_base_delay_seconds,
    )

    logger = logging.getLogger("engine")
    logger.info(
        "Started signal engine for pairs: %s | live_profile=%s enabled=%s vol_rr=%s/%s/%s liq_trail=%s | live_mode=%s enabled=%s min_score=%s session=%s regime_block=%s pair_profiles=%s pre_trade_shadow=%s/%s/%s forward_journal=%s heartbeat=%s data_freshness_gate=%s/%ss data_diagnostics=%s itick_ws_shadow=%s live_bar_builder=%s",
        ", ".join(live_pairs),
        settings.exit_profile_preset,
        settings.enable_exit_engine,
        settings.exit_volatility_rr_enabled,
        settings.exit_volatility_rr_floor,
        settings.exit_volatility_rr_cap,
        settings.exit_liquidity_trailing_enabled,
        live_mode.name,
        live_mode.enabled,
        live_mode.min_score,
        _format_windows(live_mode.session_gate_windows_utc) if live_mode.enable_session_gate else "-",
        ",".join(live_mode.regime_label_blocklist) if live_mode.enable_regime_label_gate else "-",
        ",".join(pair_profiles.keys()) if pair_profiles else "-",
        settings.enable_pre_trade_filter_shadow,
        settings.pre_trade_block_expansion_continuation,
        settings.pre_trade_block_expansion_continuation_fallback,
        settings.enable_forward_journal,
        settings.enable_live_heartbeat,
        settings.enable_market_data_freshness_gate,
        settings.max_live_candle_age_seconds,
        settings.enable_market_data_diagnostics,
        settings.enable_itick_websocket_shadow,
        settings.enable_live_bar_builder,
    )
    telemetry.engine_started(
        pairs=live_pairs,
        data_source=settings.data_source,
        live_mode=live_mode.name,
        scan_interval_minutes=settings.scan_interval_minutes,
        exit_profile=settings.exit_profile_preset,
        pre_trade_shadow_enabled=settings.enable_pre_trade_filter_shadow,
    )
    heartbeat.engine_started(
        pairs=live_pairs,
        data_source=settings.data_source,
        live_mode=live_mode.name,
        scan_interval_minutes=settings.scan_interval_minutes,
    )
    if itick_websocket_shadow is not None:
        await itick_websocket_shadow.start(live_pairs)

    try:
        while True:
            cycle_started = time.monotonic()
            cycle_started_at = datetime.now(timezone.utc)
            cycle_id = telemetry.next_cycle_id()
            telemetry.scan_started(cycle_id=cycle_id, pairs=live_pairs)
            heartbeat.scan_started(
                cycle_id=cycle_id,
                pairs=live_pairs,
                scan_started_at=cycle_started_at,
                scan_interval_minutes=settings.scan_interval_minutes,
            )
            try:
                signals = await asyncio.to_thread(engine.scan_pairs, live_pairs)
            except Exception as exc:
                duration = time.monotonic() - cycle_started
                telemetry.scan_failed(
                    cycle_id=cycle_id,
                    duration_seconds=duration,
                    error=exc,
                )
                heartbeat.scan_failed(
                    cycle_id=cycle_id,
                    pairs=live_pairs,
                    scan_started_at=cycle_started_at,
                    duration_seconds=duration,
                    error=exc,
                    scan_interval_minutes=settings.scan_interval_minutes,
                )
                raise

            telemetry.signals_found(cycle_id=cycle_id, signals=signals)
            pre_trade_shadow_rows: list[dict[str, object]] = []
            if pre_trade_shadow is not None:
                pre_trade_shadow_rows = await asyncio.to_thread(pre_trade_shadow.evaluate_signals, signals)
                telemetry.pre_trade_shadow_summary(cycle_id=cycle_id, rows=pre_trade_shadow_rows)

            shadow_by_fingerprint = {
                str(row.get("fingerprint")): row
                for row in pre_trade_shadow_rows
                if row.get("fingerprint") is not None
            }
            journal_ids: dict[str, str] = {}
            for signal in signals:
                fingerprint = signal.fingerprint()
                journal_ids[fingerprint] = forward_journal.record_candidate(
                    cycle_id=cycle_id,
                    signal=signal,
                    pre_trade_shadow=shadow_by_fingerprint.get(fingerprint),
                )

            sent_count = 0
            for signal in signals:
                send_started = time.monotonic()
                delivered = await telegram.send_signal(signal)
                delivery_latency = time.monotonic() - send_started
                telemetry.telegram_delivery(
                    cycle_id=cycle_id,
                    signal=signal,
                    delivered=delivered,
                    latency_seconds=delivery_latency,
                )
                forward_journal.record_delivery(
                    cycle_id=cycle_id,
                    journal_id=journal_ids.get(signal.fingerprint())
                    or forward_journal.build_journal_id(cycle_id=cycle_id, signal=signal),
                    signal=signal,
                    delivered=delivered,
                    latency_seconds=delivery_latency,
                )
                if delivered:
                    sent_count += 1

            logger.info(
                "Scan completed | found=%s sent=%s pairs=%s",
                len(signals),
                sent_count,
                len(live_pairs),
            )
            telemetry.scan_completed(
                cycle_id=cycle_id,
                duration_seconds=time.monotonic() - cycle_started,
                pair_count=len(live_pairs),
                found_count=len(signals),
                sent_count=sent_count,
                shadow_would_block_count=sum(1 for row in pre_trade_shadow_rows if row.get("would_block")),
            )
            heartbeat.scan_completed(
                cycle_id=cycle_id,
                pairs=live_pairs,
                scan_started_at=cycle_started_at,
                duration_seconds=time.monotonic() - cycle_started,
                found_count=len(signals),
                sent_count=sent_count,
                scan_interval_minutes=settings.scan_interval_minutes,
            )
            if shadow_logger is not None:
                try:
                    await asyncio.to_thread(shadow_logger.compare_market_data, live_pairs)
                    if shadow_engine is not None:
                        shadow_signals = await asyncio.to_thread(shadow_engine.scan_pairs, live_pairs)
                        shadow_logger.compare_signals(signals, shadow_signals)
                except Exception as exc:
                    logger.warning("Market data shadow cycle failed: %s", exc)

            elapsed = time.monotonic() - cycle_started
            sleep_for = max(1.0, settings.scan_interval_minutes * 60 - elapsed)
            await asyncio.sleep(sleep_for)
    finally:
        if itick_websocket_shadow is not None:
            await itick_websocket_shadow.stop()
        if live_bar_builder is not None:
            live_bar_builder.flush()
        market_data.close()
        if shadow_market_data is not None:
            shadow_market_data.close()
        await telegram.close()


def main() -> None:
    configure_logging()
    try:
        asyncio.run(run_engine())
    except KeyboardInterrupt:
        logging.getLogger("engine").info("Engine stopped by user")


if __name__ == "__main__":
    main()
