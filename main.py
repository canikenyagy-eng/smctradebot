from __future__ import annotations

import asyncio
import logging
import time

from config import Settings
from core.signal_engine import SignalEngine
from data.market_data import MarketDataClient
from execution.news import NewsFilter
from services.telegram import TelegramSignalService


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def run_engine() -> None:
    settings = Settings.from_env()

    market_data = MarketDataClient(
        history_limit=settings.history_limit,
        data_source=settings.data_source,
        mt5_login=settings.mt5_login,
        mt5_password=settings.mt5_password,
        mt5_server=settings.mt5_server,
    )
    news_filter = NewsFilter(
        blackout_before_minutes=settings.news_blackout_before_minutes,
        blackout_after_minutes=settings.news_blackout_after_minutes,
        surprise_threshold=settings.news_surprise_threshold,
    )
    engine = SignalEngine(
        market_data=market_data,
        news_filter=news_filter,
        htf_timeframe=settings.htf_timeframe,
        ltf_timeframe=settings.ltf_timeframe,
        trigger_timeframe=settings.trigger_timeframe,
        min_score=settings.min_score,
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
        session_min_score=settings.session_min_score,
        enable_smt_confirmation=settings.enable_smt_confirmation,
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
        enable_adaptive_weights=settings.enable_adaptive_weights,
        adaptive_regime_weights=settings.adaptive_regime_weights,
        enable_score_normalization=settings.enable_score_normalization,
        score_normalization_method=settings.score_normalization_method,
        score_normalization_window=settings.score_normalization_window,
        score_normalization_scale_factor=settings.score_normalization_scale_factor,
        score_normalization_backtest_only=settings.score_normalization_backtest_only,
        allow_live_score_normalization=settings.allow_live_score_normalization,
        runtime_mode="live",
        enable_dynamic_threshold=settings.enable_dynamic_threshold,
        threshold_percentile=settings.threshold_percentile,
        threshold_rolling_window=settings.threshold_rolling_window,
        apply_dynamic_threshold=settings.apply_dynamic_threshold,
        dynamic_threshold_backtest_only=settings.dynamic_threshold_backtest_only,
        allow_live_dynamic_threshold=settings.allow_live_dynamic_threshold,
    )
    telegram = TelegramSignalService(
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )

    logger = logging.getLogger("engine")
    logger.info("Started signal engine for pairs: %s", ", ".join(settings.pairs))

    try:
        while True:
            cycle_started = time.monotonic()
            signals = await asyncio.to_thread(engine.scan_pairs, settings.pairs)

            sent_count = 0
            for signal in signals:
                delivered = await telegram.send_signal(signal)
                if delivered:
                    sent_count += 1

            logger.info(
                "Scan completed | found=%s sent=%s pairs=%s",
                len(signals),
                sent_count,
                len(settings.pairs),
            )

            elapsed = time.monotonic() - cycle_started
            sleep_for = max(1.0, settings.scan_interval_minutes * 60 - elapsed)
            await asyncio.sleep(sleep_for)
    finally:
        await telegram.close()


def main() -> None:
    configure_logging()
    try:
        asyncio.run(run_engine())
    except KeyboardInterrupt:
        logging.getLogger("engine").info("Engine stopped by user")


if __name__ == "__main__":
    main()
