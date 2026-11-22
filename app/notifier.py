import html
import logging
import re
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token, self.chat_id = token, chat_id
        self.last_delivery = {"status": "never_attempted", "http_status": None, "error": None}

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    @staticmethod
    def tradingview_chart_url(symbol: str) -> str:
        """Build a Delta Exchange India TradingView perpetual-futures chart URL.

        Delta India symbols use the DELTAIN: exchange prefix and .P perpetual
        suffix on TradingView (for example DELTAIN:BTCUSD.P). Keep a generic
        TradingView symbol search fallback for non-market/system test symbols.
        """
        raw_symbol = str(symbol or "").strip().upper()
        if raw_symbol.startswith("DELTAIN:"):
            raw_symbol = raw_symbol.split(":", 1)[1]
        if raw_symbol.endswith(".P"):
            raw_symbol = raw_symbol[:-2]

        if re.fullmatch(r"[A-Z0-9]+", raw_symbol):
            ticker = f"DELTAIN:{raw_symbol}.P"
        else:
            ticker = raw_symbol or "BTCUSD"
        return (
            "https://www.tradingview.com/chart/?symbol="
            + quote(ticker, safe="")
            + "&interval=5"
        )

    @staticmethod
    def _format_signal(signal: dict) -> str:
        """Format every signal field in bold HTML, escaping all dynamic values."""
        strategy = str(signal.get("strategy", "PULLBACK")).upper()
        strategy_label = {
            "PULLBACK": "NORMAL",
            "NORMAL": "NORMAL",
            "ORB": "ORB",
            "MEX": "MEX",
        }.get(strategy, strategy)
        side = str(signal.get("side", "")).upper()
        dot = {"LONG": "🟢", "SHORT": "🔴"}.get(side, "⚪")
        symbol = html.escape(str(signal.get("symbol", "UNKNOWN")))
        heading = f"{dot} <b>{symbol} | {html.escape(strategy_label)}</b>"

        fields = [
            ("Side", side or "N/A"),
            ("Entry", signal.get("entry", "N/A")),
            ("Stop Loss", signal.get("stop_loss", "N/A")),
            ("Target (2R)", signal.get("take_profit_1", "N/A")),
            ("Setup Time (UTC)", signal.get("setup_time", "N/A")),
            ("15m Context (informational)", signal.get("context_15m", "UNAVAILABLE")),
            ("Reason", signal.get("reason", "N/A")),
        ]
        lines = [heading]
        lines.extend(
            f"<b>{html.escape(label)}: {html.escape(str(value))}</b>"
            for label, value in fields
        )
        chart_url = html.escape(
            TelegramNotifier.tradingview_chart_url(signal.get("symbol", "")),
            quote=True,
        )
        lines.append(f'<a href="{chart_url}">📊 OPEN TRADINGVIEW CHART (5m)</a>')
        lines.append("<b>Signal only — no order placed.</b>")
        return "\n".join(lines)

    async def send_signal(self, signal: dict, feedback_token: str = "") -> bool:
        if not self.configured:
            self.last_delivery = {"status": "not_configured", "http_status": None, "error": "telegram_not_configured"}
            log.error("Signal trigger reached but Telegram is not configured; setup retained, alert not delivered")
            return False
        text = self._format_signal(signal)
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            request_payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }
            if feedback_token:
                request_payload["reply_markup"] = {
                    "inline_keyboard": [[
                        {"text": "👍 Good", "callback_data": f"r|good|{feedback_token}"},
                        {"text": "👎 Bad", "callback_data": f"r|bad|{feedback_token}"},
                        {"text": "⏰ Late", "callback_data": f"r|late|{feedback_token}"},
                    ], [
                        {"text": "🚫 Missed", "callback_data": f"r|missed|{feedback_token}"},
                        {"text": "🔎 Needs Review", "callback_data": f"r|needs_review|{feedback_token}"},
                    ]]
                }
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(url, json=request_payload)
            http_status = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if http_status < 200 or http_status >= 300:
                error = f"telegram_http_{http_status}"
                self.last_delivery = {"status": "failed", "http_status": http_status, "error": error}
                log.warning("Telegram delivery failed http_status=%s", http_status)
                return False
            if not payload.get("ok"):
                description = str(payload.get("description", "api_rejected"))[:160]
                self.last_delivery = {"status": "failed", "http_status": http_status, "error": f"telegram_api_rejected:{description}"}
                log.warning("Telegram API rejected alert http_status=%s description=%s", http_status, description)
                return False
        except httpx.HTTPError as exc:
            error = f"telegram_transport_error:{type(exc).__name__}"
            self.last_delivery = {"status": "failed", "http_status": None, "error": error}
            log.warning("Telegram transport failure error_type=%s", type(exc).__name__)
            return False
        self.last_delivery = {"status": "sent", "http_status": http_status, "error": None}
        return True

    async def send_text(self, message: str) -> bool:
        """Send a plain-text training/review reply to the configured chat."""
        if not self.configured:
            self.last_delivery = {"status": "not_configured", "http_status": None, "error": "telegram_not_configured"}
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(url, json={"chat_id": self.chat_id, "text": str(message)[:3900]})
            payload = response.json() if response.content else {}
            if response.status_code < 200 or response.status_code >= 300 or not payload.get("ok"):
                self.last_delivery = {"status": "failed", "http_status": response.status_code, "error": "telegram_api_rejected"}
                return False
        except (httpx.HTTPError, ValueError) as exc:
            self.last_delivery = {"status": "failed", "http_status": None, "error": f"telegram_transport_error:{type(exc).__name__}"}
            return False
        self.last_delivery = {"status": "sent", "http_status": response.status_code, "error": None}
        return True

    async def register_webhook(self, webhook_url: str, secret: str) -> bool:
        """Register this service's authenticated Telegram callback endpoint."""
        return await self._telegram_api(
            "setWebhook",
            {"url": webhook_url, "secret_token": secret,
                "allowed_updates": ["message", "edited_message", "callback_query"]},
        )

    async def _telegram_api(self, method: str, payload: dict) -> bool:
        if not self.configured:
            return False
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(url, json=payload)
            data = response.json() if response.content else {}
            return 200 <= response.status_code < 300 and bool(data.get("ok"))
        except (httpx.HTTPError, ValueError):
            return False

    async def answer_callback_query(self, callback_query_id: str, message: str) -> bool:
        if not callback_query_id:
            return False
        return await self._telegram_api(
            "answerCallbackQuery",
            {"callback_query_id": callback_query_id, "text": str(message)[:180]},
        )

    async def clear_signal_review_buttons(self, chat_id: int, message_id: int) -> bool:
        return await self._telegram_api(
            "editMessageReplyMarkup",
            {"chat_id": chat_id, "message_id": message_id, "reply_markup": {"inline_keyboard": []}},
        )
