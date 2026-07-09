from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from core.signal_engine import TradeSignal
from utils.dedup import SignalDeduplicator

logger = logging.getLogger(__name__)


def _format_signal(signal: TradeSignal) -> str:
    profile = signal.meta.get("live_profile") if isinstance(signal.meta, dict) else None
    profile_line = ""
    if isinstance(profile, dict) and profile.get("enabled"):
        profile_line = (
            f"🧬 <b>Profile:</b> {profile.get('preset')} / {profile.get('regime_profile')} "
            f"| RR {profile.get('target_rr')} | LiqTrail {'ON' if profile.get('liquidity_trailing_enabled') else 'OFF'}\n"
        )
    return (
        f"🚀 <b>SMC AI SIGNAL ({signal.symbol})</b>\n\n"
        f"📊 <b>Type:</b> {signal.side}\n"
        f"📍 <b>Entry:</b> {signal.entry}\n"
        f"🛑 <b>SL:</b> {signal.stop_loss}\n"
        f"🎯 <b>TP:</b> {signal.take_profit}\n"
        f"🧭 <b>Mode:</b> {signal.entry_mode} | {signal.entry_source}\n"
        f"📝 <b>Plan:</b> {signal.entry_summary}\n\n"
        f"{profile_line}"
        f"🛠 <b>Mgmt:</b> {signal.management_summary}\n"
        f"🧷 <b>Partial:</b> {signal.partial_take_profit if signal.partial_take_profit is not None else 'OFF'} | {int(signal.partial_take_fraction * 100)}%\n"
        f"♻️ <b>Break-even:</b> {signal.break_even_r}R | <b>Trail:</b> {'ON' if signal.trailing_enabled else 'OFF'} @{signal.trailing_start_r}R\n"
        f"⏳ <b>Time stop:</b> {signal.time_stop_bars} bars\n\n"
        f"🧠 <b>Score:</b> {signal.score}/100\n"
        f"🧪 <b>Shadow:</b> FVG {signal.score_breakdown.fvg_alignment} | OB {signal.score_breakdown.order_block_alignment} | MIT {signal.score_breakdown.mitigation_alignment} | SMT {signal.score_breakdown.smt_alignment} | +{signal.score_breakdown.shadow_bonus}\n"
        f"📊 <b>HTF:</b> {signal.htf_bias}\n"
        f"🧭 <b>Regime:</b> {signal.regime_label} ({signal.regime_direction})\n"
        f"⚡ <b>Trigger:</b> {signal.trigger_direction} | {signal.trigger_event} | {signal.trigger_strength}/20\n"
        f"📌 <b>Zone:</b> {signal.zone}\n"
        f"🧩 <b>Structure:</b> {signal.structure_event} ({signal.structure_trend})\n"
        f"🕒 <b>UTC:</b> {signal.generated_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )


class TelegramSignalService:
    def __init__(
        self,
        token: str,
        chat_id: str,
        dedup_cache_size: int = 2000,
        send_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
    ) -> None:
        self.bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self.chat_id = chat_id
        self.deduplicator = SignalDeduplicator(max_cache=dedup_cache_size)
        self._lock = asyncio.Lock()
        self.send_retries = max(1, int(send_retries))
        self.retry_base_delay_seconds = max(0.1, float(retry_base_delay_seconds))

    async def close(self) -> None:
        await self.bot.session.close()

    async def send_signal(self, signal: TradeSignal) -> bool:
        fp = signal.fingerprint()

        async with self._lock:
            if self.deduplicator.seen(fp):
                return False

        delivered = await self._send_message_with_retry(_format_signal(signal), signal.symbol)
        if not delivered:
            return False

        async with self._lock:
            self.deduplicator.remember(fp)
        return True

    async def send_text(self, text: str, *, label: str = "message") -> bool:
        return await self._send_message_with_retry(text, label)

    async def _send_message_with_retry(self, text: str, label: str) -> bool:
        for attempt in range(1, self.send_retries + 1):
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=text)
                return True
            except Exception as exc:
                if attempt >= self.send_retries:
                    logger.warning(
                        "Telegram send failed for %s after %s attempts: %s",
                        label,
                        attempt,
                        exc,
                    )
                    return False

                retry_after = getattr(exc, "retry_after", None)
                delay = (
                    float(retry_after)
                    if retry_after is not None
                    else self.retry_base_delay_seconds * (2 ** (attempt - 1))
                )
                delay = max(0.1, min(60.0, delay))
                logger.warning(
                    "Telegram send attempt %s/%s failed for %s: %s | retrying in %.1fs",
                    attempt,
                    self.send_retries,
                    label,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        return False

    async def send_debug_payload(self, signal: TradeSignal) -> None:
        payload = asdict(signal)
        await self.bot.send_message(chat_id=self.chat_id, text=f"<pre>{payload}</pre>")
