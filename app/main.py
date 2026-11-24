import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, Response

from .config import settings
from .notifier import TelegramNotifier
from .outcomes import evaluate_market_outcome
from .signal_engine import SignalEngine
from .logging_config import configure_logging, redact_payload

configure_logging(getattr(logging, settings.log_level.upper(), logging.INFO))

engine: SignalEngine | None = None
engine_task: asyncio.Task | None = None

class SignalFeedback(BaseModel):
    evaluation_id: str = Field(min_length=3, max_length=512)
    feedback: str
    note: str = Field(default="", max_length=2000)
    outcome: dict = Field(default_factory=dict)




def _enabled_modules() -> list[str]:
    return list(getattr(settings, "enabled_modules", ["ORB", "MEX"]))


def _module_is_enabled(module: str) -> bool:
    checker = getattr(settings, "module_enabled", None)
    if callable(checker):
        return bool(checker(module))
    return module.upper() in _enabled_modules()


def _enabled_strategy_count() -> int:
    """Count enabled strategy types, not only top-level modules."""
    count = 0
    if _module_is_enabled("ORB"):
        count += sum(bool(getattr(settings, name, False)) for name in (
            "orb_enabled", "rb_enabled", "tpr_enabled"
        ))
    if _module_is_enabled("MEX"):
        count += int(bool(getattr(settings, "mex_enabled", False)))
    if _module_is_enabled("ADVIKA"):
        count += int(bool(getattr(settings, "advika_enabled", True)))
    return count


def _require_journal_key(provided: str | None) -> None:
    expected = settings.journal_admin_key
    if not expected:
        raise HTTPException(status_code=503, detail="Signal journal admin key is not configured")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")



@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, engine_task
    settings.validate_safety()
    notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
    engine = SignalEngine(settings, notifier)
    try:
        # This deployment is outbound-only: never register an inbound Telegram
        # webhook, even if an old TELEGRAM_WEBHOOK_SECRET remains configured.
        if notifier.token:
            removed = await notifier.delete_webhook()
            if removed:
                logging.getLogger(__name__).info(
                    "Telegram webhook removal confirmed; outbound notifications remain enabled"
                )
            else:
                logging.getLogger(__name__).warning(
                    "Telegram webhook removal could not be confirmed; verify Telegram API connectivity"
                )
        logging.getLogger(__name__).info(
            "Telegram inbound webhook disabled; outbound notifications remain enabled"
        )
        # Fail startup rather than silently scan without recovering durable setups.
        engine.restore_pending_setups()
        engine_task = asyncio.create_task(engine.run(), name="delta-signal-engine")
        logging.getLogger(__name__).info(
            "Started signal engine symbols=%s database_backend=%s pending_restored=%s "
            "signal_only=%s orders_enabled=%s",
            settings.symbol_list, engine.store.backend, engine.pending_restore["restored"],
            settings.signal_only, settings.orders_enabled,
        )
        yield
    finally:
        if engine_task:
            engine_task.cancel()
            try:
                await engine_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.getLogger(__name__).exception("Signal engine task failed during shutdown")
            finally:
                engine_task = None
        if engine is not None:
            try:
                await engine.close()
            except Exception:
                logging.getLogger(__name__).exception("Signal engine resource cleanup failed")


app = FastAPI(
    title="Delta Exchange India Signal Bot",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/ping", include_in_schema=False)
async def ping() -> Response:
    """External keep-alive probe only: no body, engine access, scan, or side effects."""
    return Response(status_code=200)


def _operational_snapshot() -> tuple[dict, dict]:
    """Build readiness from live engine state without mutating trading state."""
    checked_at = datetime.now(timezone.utc).isoformat()
    if engine is None:
        report = {
            "ready": False,
            "status": "starting",
            "checked_at": checked_at,
            "engine_task": "not_started",
            "checks": {"engine_started": False},
            "reasons": ["engine_not_started"],
            "details": {},
        }
        return {"engine": "not_started"}, report

    try:
        data = engine.health()
    except Exception as exc:
        logging.getLogger(__name__).exception("Operational health snapshot failed")
        report = {
            "ready": False,
            "status": "degraded",
            "checked_at": checked_at,
            "engine_task": "unknown",
            "checks": {"engine_health": False},
            "reasons": [f"engine_health_error:{type(exc).__name__}"],
            "details": {},
        }
        return {}, report

    task_state = "running"
    task_ok = bool(engine_task is not None and not engine_task.done())
    if engine_task is None:
        task_state = "not_started"
    elif engine_task.cancelled():
        task_state = "cancelled"
    elif engine_task.done():
        task_state = "failed"
    configured = list(settings.symbol_list)
    active = list(getattr(engine, "active_symbols", []) or [])
    required = set(active)
    quote_missing = sorted(required.intersection(data.get("quote_missing_symbols", [])))
    quote_stale = sorted(required.intersection(data.get("quote_stale_symbols", [])))
    candle_gaps = sorted(required.intersection(data.get("candle_gap_list", [])))
    stale_candles = sorted(required.intersection(data.get("stale_candle_list", [])))
    candle_ready = int(data.get("candle_ready_count", 0) or 0)
    candle_diagnostics = data.get("candle_diagnostics", {})
    ready_candle_symbols = {
        symbol for symbol, diagnostic in candle_diagnostics.items()
        if diagnostic.get("state") == "ready"
    } if isinstance(candle_diagnostics, dict) else set()
    missing_ready_candles = sorted(required.difference(ready_candle_symbols))
    checks = {
        "engine_started": True,
        "engine_task_running": task_ok,
        "active_symbols_available": bool(active),
        "websocket_connected": bool(data.get("websocket_connected", False)),
        "websocket_heartbeat_fresh": (
            "websocket_heartbeat_stale" in data and not bool(data.get("websocket_heartbeat_stale"))
        ),
        "active_symbols_have_ready_candles": (
            bool(active) and "candle_diagnostics" in data and not missing_ready_candles
        ),
        "active_symbols_have_quotes": (
            bool(active) and "quote_missing_symbols" in data and not quote_missing
        ),
        "active_quotes_fresh": (
            bool(active) and "quote_stale_symbols" in data and not quote_stale
        ),
        "active_candles_contiguous": (
            bool(active) and "candle_gap_list" in data and not candle_gaps
        ),
        "active_candles_fresh": (
            bool(active) and "stale_candle_list" in data and not stale_candles
        ),
        "state_store_healthy": bool(data.get("state_store_ok", False)),
        "telegram_configured": bool(settings.telegram_bot_token and settings.telegram_chat_id),
        "signal_only_safety": bool(settings.signal_only and not settings.orders_enabled),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    report = {
        "ready": all(checks.values()),
        "status": "ready" if all(checks.values()) else "degraded",
        "checked_at": checked_at,
        "engine_task": task_state,
        "checks": checks,
        "reasons": reasons,
        "details": {
            "configured_symbols": len(configured),
            "active_symbols": len(active),
            "inactive_symbols": max(0, len(configured) - len(active)),
            "ready_candles": candle_ready,
            "missing_ready_candles": missing_ready_candles,
            "missing_quotes": quote_missing,
            "stale_quotes": quote_stale,
            "candle_gaps": candle_gaps,
            "stale_candles": stale_candles,
            "database_backend": data.get("database_backend"),
            "telegram_outbox": data.get("telegram_outbox"),
        },
    }
    return data, report


@app.get("/ready")
async def ready():
    """Render process-readiness probe; market-data health is diagnostic, not a restart trigger.

    A quote feed is naturally sparse per symbol and the Delta socket can reconnect.
    Neither condition means the ASGI process is unable to serve requests. Detailed
    market-data, candle, database and Telegram health remains available at /status.
    """
    if engine is None:
        return JSONResponse(status_code=503, content={
            "ready": False, "status": "starting", "engine_task": "not_started",
            "checks": {"engine_started": False},
            "reasons": ["engine_not_started"],
        })
    task_running = bool(engine_task is not None and not engine_task.done())
    safety_ok = bool(settings.signal_only and not settings.orders_enabled)
    checks = {
        "engine_started": True,
        "engine_task_running": task_running,
        "signal_only_safety": safety_ok,
    }
    ready_state = all(checks.values())
    task_state = (
        "not_started" if engine_task is None
        else "cancelled" if engine_task.cancelled()
        else "failed" if engine_task.done()
        else "running"
    )
    payload = {
        "ready": ready_state,
        "status": "ready" if ready_state else "starting",
        "engine_task": task_state,
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "note": "Market-data degradation is reported by /status and does not restart the process.",
    }
    return JSONResponse(status_code=200 if ready_state else 503, content=payload)


@app.get("/status")
async def status():
    """Compact operational snapshot for debugging; contains no credentials or secrets."""
    data, report = _operational_snapshot()
    shadow_ml = {}
    journal_counts = {}
    market_outcome_counts = {}
    if engine is not None:
        journal = getattr(engine, "journal", None)
        if journal is not None:
            try:
                journal_counts = journal.counts()
                market_outcome_counts = journal.market_outcome_counts()
            except Exception:
                logging.getLogger(__name__).exception("Journal diagnostics snapshot failed")
        if getattr(engine, "shadow_ml", None) is not None:
            try:
                shadow_ml = engine.shadow_ml.status(limit=5)
            except Exception:
                logging.getLogger(__name__).exception("Shadow ML diagnostics snapshot failed")
                shadow_ml = {"status": "unavailable"}
    return redact_payload({
        "service": "delta-exchange-india-signal-bot",
        "status": report["status"],
        "checked_at": report["checked_at"],
        "engine_task": report["engine_task"],
        "readiness": report,
        "mode": {
            "bot_module": str(getattr(settings, "bot_module", "ALL")).upper(),
            "enabled_modules": _enabled_modules(),
            "signal_only": bool(settings.signal_only),
            "orders_enabled": bool(settings.orders_enabled),
            "execution": "disabled",
        },
        "market_data": {
            key: data.get(key) for key in (
                "websocket_connected", "websocket_reconnects",
                "websocket_heartbeat_age_seconds", "symbols_with_quotes",
                "quote_rejection_diagnostics", "entry_quote_max_age_seconds",
                "entry_stale_quote_symbols", "quote_stale_symbols",
                "symbols_with_candles", "active_symbol_count",
                "stale_quotes", "stale_candle_symbols", "candle_gap_symbols",
                "last_ws_message_at", "last_live_candlestick_at",
                "live_candlestick_volume", "websocket_heartbeats_received",
                "scanner_cycles", "last_scan_cycle_at",
                "last_scan_cycle_duration_seconds", "last_scan_cycle_errors",
            ) if key in data
        },
        "strategy_modules": data.get("strategy_modules", _enabled_modules()),
        "strategy_evaluations": data.get("strategy_evaluations", {}),
        "trigger_lifecycle": data.get("trigger_lifecycle", {}),
        "strategy_diagnostics": data.get("strategy_diagnostics", {}),
        "pending_by_module": data.get("pending_by_module", {}),
        "pending_by_setup_type": data.get("pending_by_setup_type", {}),
        "telegram_delivery": data.get("telegram_delivery", {}),
        "advika_signal_delivery": data.get("advika_signal_delivery", {}),
        "telegram_outbox": data.get("telegram_outbox"),
        "shadow_ml": shadow_ml,
        "signal_journal": journal_counts,
        "market_outcomes": market_outcome_counts,
        "pending_setups": data.get("pending_setups"),
        "pending_setup_recovery": data.get("pending_setup_recovery"),
        "persistence": {
            "database_backend": data.get("database_backend"),
            "state_store_ok": data.get("state_store_ok", False),
        },
    })


@app.get("/health")
async def health():
    """Cheap process liveness probe for Render; detailed readiness remains at /ready.

    Render must not restart a healthy ASGI process because the engine is still
    warming up or a background task is restarting. Full diagnostics remain at
    /status, while /ready reports whether the signal engine can currently scan.
    """
    return await ping()


@app.get("/admin/signal-history")
async def signal_history(
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    strategy: str | None = None,
    decision: str | None = None,
    feedback: str | None = None,
):
    """Read-only, authenticated journal query for backtesting and review."""
    _require_journal_key(x_journal_key)
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    return {"summary": engine.journal.counts(),
            "records": engine.journal.list_records(limit, offset, strategy, decision, feedback)}


@app.get("/admin/telegram-delivery/{signal_key}")
async def telegram_delivery_record(
    signal_key: str,
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
):
    """Inspect persisted outbox and sent-dedupe rows without returning alert payloads."""
    _require_journal_key(x_journal_key)
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    try:
        record = engine.store.delivery_record(signal_key)
    except Exception:
        logging.getLogger(__name__).exception("Telegram delivery record lookup failed")
        raise HTTPException(status_code=503, detail="Delivery record temporarily unavailable")
    if record is None:
        raise HTTPException(status_code=404, detail="Telegram delivery record not found")
    return {"delivery": record}


@app.get("/admin/signal-history.csv")
async def signal_history_csv(
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
    strategy: str | None = None,
    decision: str | None = None,
    feedback: str | None = None,
):
    """Export the most recent 500 journal rows as CSV."""
    _require_journal_key(x_journal_key)
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    content = engine.journal.export_csv(strategy=strategy, decision=decision, feedback=feedback)
    return Response(content=content, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=signal-history.csv"})


@app.post("/admin/signal-feedback")
async def signal_feedback(
    payload: SignalFeedback,
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
):
    """Attach a human review label; labels never alter live strategy thresholds."""
    _require_journal_key(x_journal_key)
    allowed = {"good", "bad", "late", "missed", "needs_review"}
    label = payload.feedback.strip().lower()
    if label not in allowed:
        raise HTTPException(status_code=422, detail=f"feedback must be one of: {', '.join(sorted(allowed))}")
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    updated = engine.journal.add_feedback(
        payload.evaluation_id, label, payload.note, payload.outcome, source="admin_api"
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Journal evaluation not found")
    return {"status": "saved", "evaluation_id": payload.evaluation_id, "feedback": label}


@app.post("/admin/signal-feedback/undo")
async def undo_signal_feedback(
    evaluation_id: str = Query(min_length=3, max_length=512),
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
):
    _require_journal_key(x_journal_key)
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    if not engine.journal.undo_feedback(evaluation_id):
        raise HTTPException(status_code=404, detail="Reviewed evaluation not found")
    return {"status": "feedback_cleared", "evaluation_id": evaluation_id}



@app.get("/admin/market-outcomes")
async def list_market_outcomes(
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
    limit: int = Query(default=100, ge=1, le=500),
    status: str | None = Query(default=None),
):
    """Read normalized persisted outcome rows without changing live signals."""
    _require_journal_key(x_journal_key)
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    try:
        outcomes = engine.journal.list_market_outcomes(limit=limit, status=status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logging.getLogger(__name__).exception("Market outcome record lookup failed")
        raise HTTPException(status_code=503, detail="Market outcome records temporarily unavailable")
    return {"count": len(outcomes), "outcomes": outcomes, "execution": "disabled"}


@app.get("/admin/ml/status")
async def admin_ml_status(
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Inspect isolated ORB/MEX shadow models, training runs and predictions."""
    _require_journal_key(x_journal_key)
    if engine is None or getattr(engine, "shadow_ml", None) is None:
        raise HTTPException(status_code=503, detail="Shadow ML service is not available")
    try:
        return engine.shadow_ml.status(limit=limit)
    except Exception:
        logging.getLogger(__name__).exception("Shadow ML status lookup failed")
        raise HTTPException(status_code=503, detail="Shadow ML status temporarily unavailable")


@app.post("/admin/ml/train")
async def admin_ml_train(
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
    strategy: str | None = Query(default=None),
):
    """Run a controlled chronological shadow-training batch; never promotes a model."""
    _require_journal_key(x_journal_key)
    if engine is None or getattr(engine, "shadow_ml", None) is None:
        raise HTTPException(status_code=503, detail="Shadow ML service is not available")
    strategies = [strategy.upper()] if strategy else ["ORB", "MEX"]
    if any(item not in {"ORB", "MEX"} for item in strategies):
        raise HTTPException(status_code=400, detail="strategy must be ORB or MEX")
    results = []
    for item in strategies:
        try:
            results.append(
                await asyncio.to_thread(engine.shadow_ml.train_shadow_model, item)
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Shadow ML training failed for %s", item)
            results.append({
                "strategy": item, "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            })
    return {
        "mode": "shadow_only",
        "automatic_promotion": False,
        "promotion_requires_explicit_approval": True,
        "live_signal_decisions_modified": False,
        "results": results,
    }


@app.post("/admin/market-outcomes/refresh")
async def refresh_market_outcomes(
    x_journal_key: str | None = Header(default=None, alias="X-Journal-Key"),
    limit: int = Query(default=20, ge=1, le=50),
    horizon_bars: int | None = Query(default=None, ge=1, le=288),
):
    """Refresh forward outcomes from completed exchange candles; never changes live signals."""
    _require_journal_key(x_journal_key)
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    records = engine.journal.pending_market_outcomes(limit=limit)
    updated = 0
    pending = 0
    errors = []
    for record in records:
        try:
            effective_horizon = int(horizon_bars or record.get("horizon_bars") or settings.outcome_horizon_bars)
            bars = await engine.rest.candles(
                record["symbol"], "5m", min(300, effective_horizon + 4)
            )
            outcome = evaluate_market_outcome(
                record, bars, horizon_bars=effective_horizon,
                tp2_r_multiple=settings.outcome_tp2_r_multiple,
            )
            status = "complete" if outcome["complete"] else "pending"
            if engine.journal.save_market_outcome(record["evaluation_id"], status, outcome):
                updated += 1
            if status == "pending":
                pending += 1
        except Exception as exc:
            safe_error = f"{type(exc).__name__}: {str(exc)[:180]}"
            engine.journal.save_market_outcome(
                record["evaluation_id"], "error",
                {"method": "completed_5m_ohlc_first_touch", "error": safe_error},
            )
            errors.append({"evaluation_id": record["evaluation_id"], "error": safe_error})
    return {
        "status": "completed",
        "checked": len(records),
        "updated": updated,
        "pending": pending,
        "errors": errors,
        "horizon_bars": horizon_bars or settings.outcome_horizon_bars,
        "horizon_bars_override": horizon_bars,
        "source": "Delta Exchange completed 5m REST candles",
        "execution": "disabled",
    }


@app.get("/admin/learning")
async def learning_report(x_journal_key: str | None = Header(default=None, alias="X-Journal-Key")):
    _require_journal_key(x_journal_key)
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    return engine.journal.learning_report()


def _telegram_command_text(value: str):
    parts = str(value or "").strip().split(maxsplit=1)
    if not parts:
        return "", ""
    return parts[0].split("@", 1)[0].lower(), parts[1].strip() if len(parts) > 1 else ""


def _telegram_user_is_authorized(user_id: str) -> bool:
    """Fail closed unless the sender is in the explicit or private-chat allowlist."""
    allowed = {
        str(value).strip()
        for value in getattr(settings, "telegram_feedback_user_ids", [])
        if str(value).strip()
    }
    return bool(user_id and user_id in allowed)


async def _ack_telegram_callback(notifier, callback_id: str, message: str) -> None:
    try:
        if not await notifier.answer_callback_query(callback_id, message):
            logging.getLogger(__name__).warning("Telegram callback acknowledgement was not confirmed")
    except Exception:
        logging.getLogger(__name__).exception("Telegram callback acknowledgement failed")


async def _clear_telegram_review_buttons(notifier, chat_id: int, message_id: int) -> None:
    try:
        if not await notifier.clear_signal_review_buttons(chat_id, message_id):
            logging.getLogger(__name__).warning("Telegram review button cleanup was not confirmed")
    except Exception:
        logging.getLogger(__name__).exception("Telegram review button cleanup failed")


@app.post("/telegram/webhook", include_in_schema=False)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None, alias="X-Telegram-Bot-Api-Secret-Token"),
):
    """Accept Telegram callbacks and commands only from the configured chat and user."""
    secret = settings.telegram_webhook_secret
    if not secret:
        raise HTTPException(status_code=503, detail="Telegram training webhook is not configured")
    if not x_telegram_bot_api_secret_token or not hmac.compare_digest(x_telegram_bot_api_secret_token, secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    try:
        update = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Telegram update")
    if not isinstance(update, dict):
        raise HTTPException(status_code=400, detail="Invalid Telegram update")

    callback = update.get("callback_query")
    if callback is not None:
        if not isinstance(callback, dict):
            raise HTTPException(status_code=400, detail="Invalid review callback")
        callback_id = str(callback.get("id", "")).strip()
        callback_message = callback.get("message")
        callback_user = callback.get("from") or {}
        if not isinstance(callback_message, dict) or not isinstance(callback_user, dict):
            raise HTTPException(status_code=400, detail="Invalid review callback")
        chat = callback_message.get("chat") or {}
        chat_id = str(chat.get("id", "")).strip()
        user_id = str(callback_user.get("id", "")).strip()
        message_id = callback_message.get("message_id")
        try:
            chat_id_int = int(chat_id)
            message_id_int = int(message_id)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid review callback message")
        if not callback_id or not user_id:
            raise HTTPException(status_code=400, detail="Invalid review callback identity")
        if not settings.telegram_chat_id or not hmac.compare_digest(chat_id, str(settings.telegram_chat_id)):
            raise HTTPException(status_code=403, detail="Chat is not authorized")
        if not _telegram_user_is_authorized(user_id):
            raise HTTPException(status_code=403, detail="Telegram user is not authorized")

        callback_data = str(callback.get("data", ""))
        parts = callback_data.split("|")
        if len(parts) != 3 or parts[0] != "r" or len(callback_data.encode("utf-8")) > 64:
            raise HTTPException(status_code=400, detail="Invalid review callback")
        label, token = parts[1], parts[2]
        if label not in {"good", "bad", "late", "missed", "needs_review"} or not token:
            raise HTTPException(status_code=400, detail="Invalid review label")

        try:
            evaluation_id = engine.journal.save_feedback_from_token(
                token, label,
                "Reviewed from the inline buttons on the Telegram signal",
                source="telegram_inline", user_id=user_id,
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "Telegram feedback persistence failed source=telegram_inline user_id=%s", user_id
            )
            await _ack_telegram_callback(
                engine.notifier, callback_id, "Temporary database error. Please try again."
            )
            # Non-2xx keeps this update retryable; the feedback transaction did not commit.
            raise HTTPException(status_code=503, detail="Feedback persistence temporarily unavailable")

        if not evaluation_id:
            await _ack_telegram_callback(
                engine.notifier, callback_id,
                "This review was already used or the signal is no longer available.",
            )
            await _clear_telegram_review_buttons(engine.notifier, chat_id_int, message_id_int)
            return {"ok": True, "ignored": True, "reason": "review_already_used"}

        # A success acknowledgement is sent only after the database transaction commits.
        await _ack_telegram_callback(
            engine.notifier, callback_id, f"Review saved: {label.replace('_', ' ').title()}"
        )
        await _clear_telegram_review_buttons(engine.notifier, chat_id_int, message_id_int)
        return {"ok": True, "review": label, "evaluation_id": evaluation_id, "source": "telegram_inline"}

    message = update.get("message") or update.get("edited_message") or {}
    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail="Invalid Telegram message")
    chat_id = str((message.get("chat") or {}).get("id", "")).strip()
    sender = message.get("from") or {}
    user_id = str(sender.get("id", "")).strip() if isinstance(sender, dict) else ""
    if not settings.telegram_chat_id or not hmac.compare_digest(chat_id, str(settings.telegram_chat_id)):
        raise HTTPException(status_code=403, detail="Chat is not authorized")
    if not _telegram_user_is_authorized(user_id):
        raise HTTPException(status_code=403, detail="Telegram user is not authorized")
    command, argument = _telegram_command_text(message.get("text", ""))
    if not command:
        return {"ok": True, "ignored": True}

    journal = engine.journal
    help_text = "Commands: /teach note, /good [id] [note], /bad [id] [note], /late [id] [note], /missed [id] [note], /review, /learning, /undo id"
    if command == "/teach":
        if not argument:
            reply = "Usage: /teach describe your observation. Saved as a research note only."
        else:
            journal.add_training_note(argument, "telegram")
            reply = "Training note saved. Live strategy rules were not changed."
    elif command in ("/good", "/bad", "/late", "/missed"):
        label = command[1:]
        tokens = argument.split(maxsplit=1)
        supplied_id = tokens[0] if tokens and tokens[0].count(":") >= 2 else ""
        note = tokens[1] if supplied_id and len(tokens) > 1 else (argument if not supplied_id else "")
        latest = journal.latest_record(decision="setup", lifecycle="telegram_sent") if not supplied_id else None
        evaluation_id = supplied_id or (latest or {}).get("evaluation_id", "")
        if not evaluation_id:
            reply = "No setup evaluation found. Use /review and provide its evaluation_id."
        elif not journal.add_feedback(
            evaluation_id, label, note, source="telegram_command", user_id=user_id
        ):
            reply = "Evaluation not found or already reviewed. Use /review to copy an evaluation_id."
        else:
            reply = f"Feedback '{label}' saved for {evaluation_id}. Research only; live strategy unchanged."
    elif command == "/review":
        rows = journal.list_records(limit=5)
        reply = "Latest evaluations:\n" + "\n".join(
            f"{r['evaluation_id']} | {r['decision']} | {r.get('side') or '-'} | {r.get('feedback') or 'unreviewed'}"
            for r in rows
        ) if rows else "Journal is empty."
    elif command == "/learning":
        report = journal.learning_report()
        labels = report["feedback_counts"]
        reply = (
            f"Learning journal\nEvaluations: {report['total_evaluations']}\nReviewed: {report['human_reviewed']}\n"
            f"Good: {labels.get('good', 0)} | Bad: {labels.get('bad', 0)} | Late: {labels.get('late', 0)} | Missed: {labels.get('missed', 0)}\n"
            f"Training notes: {report['training_notes']}\nResearch status: {report['research_status']}\n"
            "Automatic strategy changes: OFF."
        )
    elif command == "/undo":
        latest_reviewed = next((r for r in journal.list_records(limit=100) if r.get("feedback")), None) if not argument else None
        evaluation_id = argument.split(maxsplit=1)[0] if argument else (latest_reviewed or {}).get("evaluation_id", "")
        reply = (
            f"Feedback cleared for {evaluation_id}; original evaluation retained."
            if evaluation_id and journal.undo_feedback(evaluation_id)
            else "No reviewed evaluation found. Usage: /undo evaluation_id"
        )
    else:
        reply = help_text
    if not await engine.notifier.send_text(reply):
        logging.getLogger(__name__).warning("Telegram command reply delivery failed command=%s", command)
    return {"ok": True, "command": command}


@app.post("/admin/telegram-test")
async def telegram_test(x_telegram_test_key: str | None = Header(default=None, alias="X-Telegram-Test-Key")):
    """Send one controlled test message to the configured Telegram chat only."""
    expected = settings.telegram_test_key
    if not expected:
        raise HTTPException(status_code=503, detail="Telegram test endpoint is not configured")
    if not x_telegram_test_key or not hmac.compare_digest(x_telegram_test_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if engine is None:
        raise HTTPException(status_code=503, detail="Signal engine is not started")
    test_signal = {
        "symbol": "TELEGRAM-TEST", "strategy": "SYSTEM_TEST", "side": "TEST",
        "entry": "N/A", "stop_loss": "N/A", "take_profit_1": "N/A",
        "setup_time": datetime.now(timezone.utc).isoformat(),
        "context_15m": "NOT_APPLICABLE",
        "reason": "Controlled Telegram delivery-path test; no trade signal and no order placed.",
    }
    delivered = await engine.notifier.send_signal(test_signal)
    delivery = getattr(engine.notifier, "last_delivery", {}) or {}
    if not delivered:
        logging.getLogger(__name__).warning(
            "Controlled Telegram test failed status=%s http_status=%s",
            delivery.get("status"), delivery.get("http_status"),
        )
        raise HTTPException(status_code=502, detail="Telegram test delivery failed; inspect notifier status in logs")
    return {"status": "sent", "http_status": delivery.get("http_status")}
