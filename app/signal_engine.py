"""Signal setup -> live trigger lifecycle. No order APIs exist in this project."""
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from .market_data import DeltaRestClient, DeltaPublicStream
from .models import Candle, Quote
from .momentum import _frame, _indicators, evaluate_orb
from .tpr_rb import evaluate_tpr, evaluate_rb
from .state_store import SignalStore
from .signal_journal import SignalJournal
from .validation import validate_quote
from .risk import LIVE_TARGET_R_MULTIPLE
from .advika_choice import HISTORY_BARS as ADVIKA_HISTORY_BARS, evaluate_advika_choice

log = logging.getLogger(__name__)


def _stale_quote_symbols(
    active_symbols: list[str],
    received_at: dict[str, datetime],
    now: datetime,
    stale_after_seconds: float,
) -> set[str]:
    """Return only symbols whose WebSocket quote is old enough for REST fallback."""
    return {
        symbol for symbol in active_symbols
        if symbol not in received_at
        or (now - received_at[symbol]).total_seconds() >= stale_after_seconds
    }


def _rest_quote_fallback_needed(
    active_symbols: list[str],
    received_at: dict[str, datetime],
    now: datetime,
    stale_after_seconds: float,
) -> bool:
    """Only call the full-universe REST ticker endpoint when an L1 quote is stale."""
    return bool(_stale_quote_symbols(active_symbols, received_at, now, stale_after_seconds))


@dataclass
class PendingSetup:
    symbol: str
    side: str
    candle_time: datetime
    trigger_price: float
    reference: float
    stop_loss: float
    take_profit_1: float
    atr: float
    bars_remaining: int
    reason: str
    context_15m: str
    expires_at: datetime
    strategy: str = "TPR"
    armed_at: datetime | None = None
    trigger_candidate_at: datetime | None = None
    trigger_candidate_last_at: datetime | None = None
    lifecycle_state: str = "ARMED"


class SignalEngine:
    def __init__(self, settings, notifier):
        self.settings = settings
        self.notifier = notifier
        if hasattr(settings, "validate_safety"):
            settings.validate_safety()
        # Apply versioned schema migrations at startup in every environment.
        # PostgreSQL migrations are transactional and guarded by an advisory lock,
        # so concurrent Render instances serialize safely. This is required on
        # Render Free, where a separate pre-deploy migration command is unavailable.
        migrate_on_startup = True
        self.rest = DeltaRestClient(settings.delta_base_url)
        self.stream = DeltaPublicStream(
            [], self.on_quote, self.on_stream_disconnect
        )
        self.configured_symbols = list(settings.symbol_list)
        # Snapshot symbol universes once: Settings properties read and validate
        # symbol files, so they must not run inside quote, scan, or health paths.
        self.configured_symbol_set = frozenset(self.configured_symbols)
        self.tpr_rb_symbols = frozenset(
            getattr(settings, "tpr_rb_symbol_list", self.configured_symbols)
        )
        self.active_symbols: list[str] = []
        self.unavailable_symbols: list[str] = list(self.configured_symbols)
        self.symbol_validation = {"status": "pending", "last_attempt_at": None, "last_error": None}
        self.quotes: dict[str, Quote] = {}
        self.received_at: dict[str, datetime] = {}
        self.last_quote_exchange_at: dict[str, datetime] = {}
        self.quote_rejection_diagnostics = {"counts": {}, "recent": [], "last": None}
        self.rest_quote_polls = 0
        self.rest_quote_updates = 0
        self.last_rest_quote_poll_at: str | None = None
        self.last_rest_quote_error: str | None = None
        self.candles: dict[str, list[Candle]] = {}
        self.last_evaluated: dict[str, datetime] = {}
        self.scan_cycles = 0
        self.last_scan_cycle_at: str | None = None
        self.last_scan_cycle_duration_seconds: float | None = None
        self.last_scan_cycle_errors = 0
        self.strategy_evaluations: dict[str, dict] = {
            strategy: {
                "evaluated": 0, "setups_armed": 0, "no_setup": 0, "errors": 0,
                "last_evaluated_at": None, "last_candle_at": None,
                "last_error": None, "last_error_at": None,
            }
            for strategy in ("TPR", "RB", "ORB", "MEX", "ADVIKA")
        }
        self.strategy_diagnostics: dict[str, dict] = {
            strategy: {
                "rejections": 0, "rejections_by_reason": {},
                "detail_counts": {}, "recent_rejections": [], "last_rejection": None,
                "last_rejection_by_reason": {},
                "mex_gate_counts": {"evaluated": 0, "rvol_pass": 0, "rvol_fail": 0,
                                    "atr_expansion_pass": 0, "atr_expansion_fail": 0},
                "mex_evaluation_sources": {
                    "live_forming": {"evaluations": 0, "unique_symbol_candles": 0,
                                     "setups": 0, "rejections": 0},
                    "completed_candle": {"evaluations": 0, "unique_symbol_candles": 0,
                                         "setups": 0, "rejections": 0},
                },
            }
            for strategy in ("TPR", "RB", "ORB", "MEX")
        }
        self.trigger_lifecycle = {
            strategy: {"events": {}, "last_event": None}
            for strategy in ("TPR", "RB", "ORB", "MEX")
        }
        self.telegram_delivery = {
            "attempts": 0, "sent": 0, "failed": 0, "last_status": None,
            "last_http_status": None, "last_error": None, "last_attempt_at": None,
            "last_success_at": None,
        }
        self.context_last_fetch: dict[str, datetime] = {}
        self.context_last_bucket: dict[str, int] = {}
        self.context_15m: dict[str, str] = {}
        self.last_candle_fetch_at: dict[str, datetime] = {}
        self.force_full_candle_refresh: set[str] = set()
        # Explain why each configured symbol is or is not candle-ready.
        self.candle_diagnostics: dict[str, dict] = {
            symbol: {"state": "pending", "reason": "not_fetched", "bars": 0,
                     "last_attempt_at": None, "last_success_at": None,
                     "last_bar_time": None, "last_error": None,
                     "last_attempt_status": "not_attempted"}
            for symbol in settings.symbol_list
        }
        self.pending: dict[str, PendingSetup] = {}
        self.advika_signal_count = 0
        self.advika_last_signal_at: str | None = None
        self.pending_restore = {"restored": 0, "expired": 0, "invalid": 0, "last_restore_at": None}
        # Momentum-family detectors and the combined TPR+RB module keep independent setup lifecycles.
        self.momentum_pending: dict[tuple[str, str], PendingSetup] = {}
        self.sent_keys: set[str] = set()
        self.store = SignalStore(
            settings.state_db_path, getattr(settings, "database_url", ""),
            migrate_on_startup=migrate_on_startup,
        )
        self.journal = SignalJournal(
            settings.state_db_path, getattr(settings, "database_url", ""),
            migrate_on_startup=migrate_on_startup,
        )
        self.shadow_ml = None
        self.shadow_ml_tasks: set[asyncio.Task] = set()
        if bool(getattr(settings, "shadow_ml_enabled", False)):
            try:
                from .ml_shadow import ShadowML
                self.shadow_ml = ShadowML(self.journal)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Shadow ML service initialization failed; live signal flow remains independent"
                )
        else:
            log.info("Shadow ML disabled by lightweight runtime profile")
        self.delivery_retry_at: dict[str, datetime] = {}
        self.outbox_lock = asyncio.Lock()

    async def close(self) -> None:
        """Drain non-blocking shadow capture tasks and release the Delta REST client."""
        if self.shadow_ml_tasks:
            await asyncio.gather(*tuple(self.shadow_ml_tasks), return_exceptions=True)
        await self.rest.close()
        close_notifier = getattr(self.notifier, "close", None)
        if close_notifier is not None:
            await close_notifier()

    def _on_shadow_ml_task_done(self, task: asyncio.Task) -> None:
        self.shadow_ml_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logging.getLogger(__name__).exception(
                "Background shadow ML task failed; live signal flow is unaffected"
            )

    async def validate_symbol_universe(self) -> bool:
        """Subscribe only configured symbols that Delta currently lists as live perpetuals."""
        attempted = datetime.now(timezone.utc)
        self.symbol_validation["last_attempt_at"] = attempted.isoformat()
        try:
            live_symbols = await self.rest.live_perpetual_symbols()
        except Exception as exc:
            self.active_symbols = []
            self.unavailable_symbols = list(self.configured_symbols)
            self.stream.set_symbols([])
            self.symbol_validation.update(
                status="error", last_error=f"{type(exc).__name__}: {str(exc)[:200]}"
            )
            log.exception("Delta live perpetual product validation failed; waiting before subscribing")
            return False

        self.active_symbols = [symbol for symbol in self.configured_symbols if symbol in live_symbols]
        self.unavailable_symbols = [
            symbol for symbol in self.configured_symbols if symbol not in live_symbols
        ]
        self.stream.set_symbols(self.active_symbols)
        self.symbol_validation.update(
            status="ok" if not self.unavailable_symbols else "partial",
            last_error=None,
            live_perpetual_count=len(live_symbols),
            configured_count=len(self.configured_symbols),
            active_count=len(self.active_symbols),
            unavailable_count=len(self.unavailable_symbols),
        )
        if self.unavailable_symbols:
            log.error(
                "Configured symbols unavailable as live perpetuals; skipped without substitution: %s",
                self.unavailable_symbols,
            )
        log.info(
            "Delta symbol validation complete configured=%s active=%s unavailable=%s",
            len(self.configured_symbols), len(self.active_symbols), len(self.unavailable_symbols),
        )
        return bool(self.active_symbols)

    def _serialize_pending_setup(self, setup: PendingSetup) -> dict:
        return {
            "symbol": setup.symbol, "side": setup.side,
            "candle_time": setup.candle_time.isoformat(),
            "trigger_price": setup.trigger_price, "reference": setup.reference,
            "stop_loss": setup.stop_loss, "take_profit_1": setup.take_profit_1,
            "atr": setup.atr, "bars_remaining": setup.bars_remaining,
            "reason": setup.reason, "context_15m": setup.context_15m,
            "expires_at": setup.expires_at.isoformat(), "strategy": setup.strategy,
            "armed_at": (setup.armed_at or setup.candle_time).isoformat(),
            "lifecycle_state": setup.lifecycle_state,
        }

    def _persist_pending_setup(self, setup: PendingSetup) -> None:
        # Legacy callers and restored fixtures may not have a lifecycle row yet.
        current = self.store.setup_lifecycle(
            setup.symbol, setup.strategy, setup.candle_time.isoformat()
        )
        if current is None:
            if not self.store.transition_setup(
                setup.symbol, setup.strategy, setup.candle_time.isoformat(),
                None, "ARMED", "pending_setup_adopted",
            ):
                raise RuntimeError("unable_to_initialize_setup_lifecycle")
        elif current.get("state") != setup.lifecycle_state:
            raise RuntimeError("pending_setup_lifecycle_state_mismatch")
        # Persist intent before it can be used for live quote confirmation.
        self.store.save_pending_setup(
            setup.symbol, setup.strategy, self._serialize_pending_setup(setup)
        )

    def _transition_setup(self, setup: PendingSetup, to_state: str, reason: str) -> bool:
        """Persist one legal setup transition; stale workers cannot regress state."""
        from_state = setup.lifecycle_state
        try:
            current = self.store.setup_lifecycle(
                setup.symbol, setup.strategy, setup.candle_time.isoformat()
            )
            if current is None and from_state == "ARMED":
                self.store.transition_setup(
                    setup.symbol, setup.strategy, setup.candle_time.isoformat(),
                    None, "ARMED", "legacy_in_memory_setup_adopted",
                )
            changed = self.store.transition_setup(
                setup.symbol, setup.strategy, setup.candle_time.isoformat(),
                from_state, to_state, reason,
                event_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            log.exception(
                "Unable to persist setup lifecycle transition symbol=%s strategy=%s",
                setup.symbol, setup.strategy,
            )
            return False
        if changed:
            setup.lifecycle_state = to_state
            self._record_trigger_event(setup.strategy, setup.symbol, f"state_{to_state.lower()}")
        else:
            log.warning(
                "Setup lifecycle transition fenced symbol=%s strategy=%s expected=%s target=%s",
                setup.symbol, setup.strategy, from_state, to_state,
            )
        return changed

    def _module_enabled(self, module: str) -> bool:
        checker = getattr(self.settings, "module_enabled", None)
        if callable(checker):
            return bool(checker(module))
        mode = str(getattr(self.settings, "bot_module", "ALL")).strip().upper()
        return mode == "ALL" or mode == module.upper()

    def _strategy_enabled(self, strategy: str) -> bool:
        checker = getattr(self.settings, "strategy_enabled", None)
        if callable(checker):
            return bool(checker(strategy))
        module = ("ORB" if strategy in {"ORB", "RB", "TPR"} else
                  "ADVIKA" if strategy == "ADVIKA" else "MEX")
        return self._module_enabled(module)

    def restore_pending_setups(self) -> dict:
        """Restore still-valid setup intents after process restart.

        A partial two-quote confirmation is deliberately never restored: after a
        restart the setup must earn a fresh, consecutive quote confirmation.
        """
        now = datetime.now(timezone.utc)
        restored = expired = invalid = 0
        for symbol, strategy, payload in self.store.load_pending_setups():
            try:
                if strategy == "NORMAL":
                    self.store.delete_pending_setup(symbol, strategy)
                    continue
                if strategy not in ("TPR", "RB", "ORB", "MEX", "ADVIKA"):
                    raise ValueError("unknown_strategy")
                # Leave another module's durable setup untouched. Each deployed bot
                # should still use its own database, but this avoids cross-module deletion.
                if not self._strategy_enabled(strategy):
                    continue
                if symbol not in self.configured_symbol_set:
                    raise ValueError("symbol_not_configured")
                candle_time = datetime.fromisoformat(payload["candle_time"]).astimezone(timezone.utc)
                expires_at = datetime.fromisoformat(payload["expires_at"]).astimezone(timezone.utc)
                lifecycle = self.store.setup_lifecycle(symbol, strategy, candle_time.isoformat())
                if lifecycle is None:
                    if not self.store.transition_setup(
                        symbol, strategy, candle_time.isoformat(), None, "ARMED",
                        "legacy_pending_setup_recovered",
                    ):
                        raise ValueError("unable_to_initialize_setup_lifecycle")
                    lifecycle_state = "ARMED"
                else:
                    lifecycle_state = lifecycle["state"]
                if lifecycle_state == "CONFIRMING":
                    if not self.store.transition_setup(
                        symbol, strategy, candle_time.isoformat(), "CONFIRMING", "ARMED",
                        "restart_confirmation_reset",
                    ):
                        raise ValueError("unable_to_reset_confirmation_after_restart")
                    lifecycle_state = "ARMED"
                if lifecycle_state != "ARMED":
                    self.store.delete_pending_setup(symbol, strategy)
                    continue
                if expires_at <= now:
                    expired += 1
                    self.store.transition_setup(
                        symbol, strategy, candle_time.isoformat(), "ARMED", "EXPIRED",
                        "expired_during_restart_recovery",
                    )
                    self.store.delete_pending_setup(symbol, strategy)
                    continue
                side = str(payload["side"]).upper()
                if side not in ("LONG", "SHORT"):
                    raise ValueError("invalid_side")
                numeric = {
                    key: float(payload[key])
                    for key in ("trigger_price", "reference", "stop_loss", "take_profit_1", "atr")
                }
                if not all(math.isfinite(value) for value in numeric.values()) or numeric["atr"] <= 0:
                    raise ValueError("invalid_numeric_setup")
                setup = PendingSetup(
                    symbol=symbol, side=side, candle_time=candle_time,
                    trigger_price=numeric["trigger_price"], reference=numeric["reference"],
                    stop_loss=numeric["stop_loss"], take_profit_1=numeric["take_profit_1"],
                    atr=numeric["atr"], bars_remaining=max(0, int(payload["bars_remaining"])),
                    reason=str(payload.get("reason", "")),
                    context_15m=str(payload.get("context_15m", "UNAVAILABLE")),
                    expires_at=expires_at, strategy=strategy,
                    armed_at=datetime.fromisoformat(payload.get("armed_at", now.isoformat())).astimezone(timezone.utc),
                    trigger_candidate_at=None, trigger_candidate_last_at=None,
                    lifecycle_state="ARMED",
                )
                if strategy == "TPR":
                    self.pending[symbol] = setup
                else:
                    self.momentum_pending[(symbol, strategy)] = setup
                restored += 1
            except (KeyError, TypeError, ValueError, OverflowError):
                invalid += 1
                log.exception("Discarding invalid persisted setup symbol=%s strategy=%s", symbol, strategy)
                self.store.delete_pending_setup(symbol, strategy)
        self.pending_restore = {
            "restored": restored, "expired": expired, "invalid": invalid,
            "last_restore_at": now.isoformat(),
        }
        log.info("Pending setup recovery complete restored=%s expired=%s invalid=%s", restored, expired, invalid)
        return dict(self.pending_restore)

    def _setups_for_symbol(self, symbol: str):
        setups = []
        tpr = self.pending.get(symbol)
        if tpr is not None:
            setups.append(("TPR", tpr))
        setups.extend(
            (strategy, setup)
            for (pending_symbol, strategy), setup in list(self.momentum_pending.items())
            if pending_symbol == symbol
        )
        return setups

    def _record_strategy_evaluation(self, strategy: str, latest: Candle, result, symbol: str, diagnostics=None) -> None:
        stats = self.strategy_evaluations[strategy]
        stats["evaluated"] += 1
        stats["last_evaluated_at"] = datetime.now(timezone.utc).isoformat()
        stats["last_candle_at"] = latest.timestamp.isoformat()
        stats["last_error"] = None
        if strategy == "MEX":
            diagnostic = diagnostics if isinstance(diagnostics, dict) else {}
            source = diagnostic.get("evaluation_source", "completed_candle")
            if source not in ("live_forming", "completed_candle"):
                source = "completed_candle"
            source_map = self.strategy_diagnostics[strategy].setdefault("mex_evaluation_sources", {
                "live_forming": {"evaluations": 0, "unique_symbol_candles": 0,
                                 "setups": 0, "rejections": 0},
                "completed_candle": {"evaluations": 0, "unique_symbol_candles": 0,
                                     "setups": 0, "rejections": 0},
            })
            source_stats = source_map[source]
            source_stats["evaluations"] += 1
            unique_key = (source, symbol, latest.timestamp.isoformat())
            seen = getattr(self, "_mex_unique_evaluation_keys", None)
            if seen is None:
                self._mex_unique_evaluation_keys = set()
                self._mex_unique_evaluation_order = []
                seen = self._mex_unique_evaluation_keys
            if unique_key not in seen:
                seen.add(unique_key)
                self._mex_unique_evaluation_order.append(unique_key)
                source_stats["unique_symbol_candles"] += 1
                if len(self._mex_unique_evaluation_order) > 5000:
                    expired = self._mex_unique_evaluation_order.pop(0)
                    seen.discard(expired)
            outcome = "setups" if result else "rejections"
            # These count evaluations, while unique_symbol_candles counts distinct
            # symbol/timeframe bars. Repeated live updates are deliberately visible.
            source_stats[outcome] += 1
            diagnostic["evaluation_source"] = source
        try:
            self.journal.record_evaluation(
                strategy, symbol, latest.timestamp.isoformat(),
                "setup" if result else "rejected", diagnostics or {}, result or {},
            )
        except Exception:
            log.exception("Unable to persist signal journal evaluation symbol=%s strategy=%s", symbol, strategy)
        if result:
            if strategy == "MEX":
                gates = self.strategy_diagnostics[strategy].setdefault("mex_gate_counts", {
                    "evaluated": 0, "rvol_pass": 0, "rvol_fail": 0,
                    "atr_expansion_pass": 0, "atr_expansion_fail": 0,
                })
                gates["evaluated"] += 1
                gates["rvol_pass"] += 1
                gates["atr_expansion_pass"] += 1
            stats["setups_armed"] += 1
        else:
            stats["no_setup"] += 1
            diagnostic = diagnostics or {}
            reason = diagnostic.get("reason", "no_candidate_unclassified")
            summary = self.strategy_diagnostics[strategy]
            summary["rejections"] += 1
            histogram = summary["rejections_by_reason"]
            histogram[reason] = histogram.get(reason, 0) + 1
            values = diagnostic.get("values", {}) if isinstance(diagnostic, dict) else {}
            if strategy == "MEX":
                gates = summary.setdefault("mex_gate_counts", {
                    "evaluated": 0, "rvol_pass": 0, "rvol_fail": 0,
                    "atr_expansion_pass": 0, "atr_expansion_fail": 0,
                })
                failed = set(values.get("failed_filters", [])) if isinstance(values, dict) else set()
                # A detector rejection after the participation/expansion stage
                # necessarily passed both gates. Warmup/data errors are excluded.
                if reason != "indicator_warmup_or_invalid":
                    gates["evaluated"] += 1
                    for gate in ("rvol", "atr_expansion"):
                        passed = gate not in failed
                        key = f"{gate}_{'pass' if passed else 'fail'}"
                        gates[key] += 1
            # Preserve bounded, numeric evidence so /status can distinguish
            # low participation from weak volatility expansion without
            # changing detector eligibility or flooding application logs.
            detail_counts = summary.setdefault("detail_counts", {})
            for detail in values.get("failed_filters", []) if isinstance(values, dict) else []:
                detail_counts[detail] = detail_counts.get(detail, 0) + 1
            sample_keys = (
                "rvol", "min_rvol", "session_rvol", "atr_expansion", "min_atr_expansion",
                "live_volume_projection_factor", "failed_filters",
                "range_width_atr", "min_range_atr", "max_range_atr",
                "session", "session_timezone", "opening_candle_at",
                "opening_open", "opening_high", "opening_low", "opening_close",
                "opening_volume", "current_atr", "minimum_range_atr", "maximum_range_atr",
                "session_open_utc", "current_candle_at", "session_bars", "required_session_bars",
                "candle_range", "atr", "volume", "open", "high", "low", "close",
                "source_candle_quality", "opening_source_quality",
                "range_zero_range_bar_count", "range_zero_range_zero_volume_count",
                "range_zero_range_positive_volume_count",
                "ema_spread_atr", "ema21_slope_atr", "adx", "adx_min",
                "risk_atr", "extension_atr", "close_location", "body_atr",
                "range_atr", "candle_body_atr", "candle_range_atr",
                "range_high", "range_low", "range_lookback", "range_bar_count",
                "efficiency_ratio", "min_efficiency", "trend_consistency", "min_consistency",
                "lookback", "consistency_bars", "path_length", "net_move",
                "regime_side", "regime_rejection_reason",
                "ema9", "ema21", "ema20", "ema50", "rsi", "vwap", "plus_di", "minus_di",
                "long_touch", "short_touch", "long_rejection", "short_rejection",
                "not_extended_long", "not_extended_short", "long_trend_alignment",
                "short_trend_alignment", "long_rsi_in_range", "short_rsi_in_range",
                "previous_high", "previous_low", "candle_body", "breakout_close_age",
                "edge", "buffer", "di_ok", "trend_ok", "vwap_ok", "rsi_ok",
                "context_score", "required", "checks", "impulse_atr", "roc3_atr",
                "prior_8_high", "prior_8_low", "previous_close", "max_extension_atr",
            )
            sample = {
                "symbol": symbol, "candle_at": latest.timestamp.isoformat(),
                "stage": diagnostic.get("stage", "unknown"), "reason": reason,
                "values": {key: values[key] for key in sample_keys if key in values},
            }
            recent = summary.setdefault("recent_rejections", [])
            recent.append(sample)
            del recent[:-20]
            summary["last_rejection"] = {
                "symbol": symbol, "candle_at": latest.timestamp.isoformat(),
                "stage": diagnostic.get("stage", "unknown"), "reason": reason,
                "values": values,
            }
            # Preserve the most recent example for each reason independently;
            # a frequent flat-candle reject cannot evict a regime diagnosis.
            by_reason = summary.setdefault("last_rejection_by_reason", {})
            by_reason[reason] = sample
            if len(by_reason) > 12:
                oldest_reason = next(iter(by_reason))
                if oldest_reason != reason:
                    by_reason.pop(oldest_reason, None)

    def _record_trigger_event(self, strategy: str, symbol: str, reason: str, values=None) -> None:
        diagnostic = self.trigger_lifecycle[strategy]
        events = diagnostic["events"]
        events[reason] = events.get(reason, 0) + 1
        event = {"reason": reason, "symbol": symbol,
                 "at": datetime.now(timezone.utc).isoformat()}
        if values:
            event["values"] = values
        diagnostic["last_event"] = event

    def _reset_quote_confirmation(self, symbol: str, reason: str | None = None) -> None:
        for strategy, setup in self._setups_for_symbol(symbol):
            had_candidate = setup.trigger_candidate_at is not None
            setup.trigger_candidate_at = None
            setup.trigger_candidate_last_at = None
            if had_candidate and setup.lifecycle_state == "CONFIRMING":
                self._transition_setup(setup, "ARMED", reason or "confirmation_reset")
            if reason and had_candidate:
                self._record_trigger_event(strategy, symbol, reason)

    def _remove_pending_setup(self, symbol: str, strategy: str, setup: PendingSetup) -> None:
        if strategy == "TPR" and self.pending.get(symbol) is setup:
            self.pending.pop(symbol, None)
        key = (symbol, strategy)
        if self.momentum_pending.get(key) is setup:
            self.momentum_pending.pop(key, None)
        self.delivery_retry_at.pop(symbol, None)
        try:
            self.store.delete_pending_setup(symbol, strategy)
        except Exception:
            log.exception("Unable to delete persisted pending setup symbol=%s strategy=%s", symbol, strategy)

    def _invalidate_pending_setups_for_symbol(self, symbol: str, reason: str) -> None:
        """Fail closed when recent completed-candle history is discontinuous."""
        for strategy, setup in list(self._setups_for_symbol(symbol)):
            self._record_trigger_event(strategy, symbol, reason)
            if not self._transition_setup(setup, "INVALIDATED", reason):
                log.error(
                    "Unable to persist setup invalidation symbol=%s strategy=%s reason=%s",
                    symbol, strategy, reason,
                )
            # Remove the in-memory intent even if persistence is unhealthy: a
            # known candle-history gap must never leave it eligible for quotes.
            self._remove_pending_setup(symbol, strategy, setup)

    def on_stream_disconnect(self) -> None:
        """Discard partial breakout confirmation across a WebSocket outage."""
        for symbol in set(self.pending) | {key[0] for key in self.momentum_pending}:
            for strategy, _setup in self._setups_for_symbol(symbol):
                self._record_trigger_event(strategy, symbol, "websocket_disconnect")
            self._reset_quote_confirmation(symbol, "confirmation_reset_websocket_disconnect")
        self.last_quote_exchange_at.clear()
        # A forming candle cannot be trusted across a socket outage. Discard it
        # and force REST to repair any missing completed intervals immediately.
        log.warning("WebSocket disconnected; quote confirmations reset")


    async def on_quote(self, symbol: str, quote: Quote, received: datetime) -> None:
        # Delta's exchange timestamp is the last book/trade update time, not
        # necessarily the time this unchanged L1 state was delivered. Use the
        # local receive time for entry freshness (also passed to validate_quote
        # below), while retaining exchange timestamps for sequence ordering
        # and diagnostics. Do not reject a quote solely because its exchange
        # timestamp is old.
        if quote.source != "rest":
            previous_exchange_at = self.last_quote_exchange_at.get(symbol)
            if previous_exchange_at is not None and quote.received_at <= previous_exchange_at:
                self._reset_quote_confirmation(symbol, "confirmation_reset_out_of_order_quote")
                for strategy, _setup in self._setups_for_symbol(symbol):
                    self._record_trigger_event(strategy, symbol, "out_of_order_quote")
                log.debug(
                    "Ignoring duplicate/out-of-order exchange quote symbol=%s quote_time=%s previous=%s",
                    symbol, quote.received_at.isoformat(), previous_exchange_at.isoformat(),
                )
                return
            self.last_quote_exchange_at[symbol] = quote.received_at

        self.quotes[symbol] = quote
        self.received_at[symbol] = received

        # REST ticker polling is a snapshot fallback for display/freshness only.
        # The same unchanged book can be returned on consecutive polls with a new
        # local receipt time; counting those polls as independent observations can
        # falsely satisfy the breakout hold timer. Only exchange-published WS L1
        # updates may start or advance trigger confirmation. Fail closed during a
        # WebSocket outage rather than manufacturing confirmation from REST polls.
        if quote.source == "rest":
            return

        # Evaluate each independent strategy's pending trigger against the
        # same locally received exchange update. A TPR/RB candidate does not suppress MEX.
        for strategy, setup in self._setups_for_symbol(symbol):
            await self._process_pending_quote(symbol, strategy, setup, quote, received)

    async def _process_pending_quote(
        self, symbol: str, strategy: str, setup: PendingSetup,
        quote: Quote, received: datetime,
    ) -> None:
        if received >= setup.expires_at:
            self._record_trigger_event(strategy, symbol, "setup_expired_on_quote")
            self._transition_setup(setup, "EXPIRED", "setup_expired_on_quote")
            self._remove_pending_setup(symbol, strategy, setup)
            return
        if setup.side == "LONG" and quote.ask is None:
            self._record_trigger_event(strategy, symbol, "missing_ask_for_long")
            setup.trigger_candidate_at = None
            setup.trigger_candidate_last_at = None
            if setup.lifecycle_state == "CONFIRMING":
                self._transition_setup(setup, "ARMED", "missing_ask_for_long")
            return
        if setup.side == "SHORT" and quote.bid is None:
            self._record_trigger_event(strategy, symbol, "missing_bid_for_short")
            setup.trigger_candidate_at = None
            setup.trigger_candidate_last_at = None
            if setup.lifecycle_state == "CONFIRMING":
                self._transition_setup(setup, "ARMED", "missing_bid_for_short")
            return
        entry_price = quote.ask if setup.side == "LONG" else quote.bid
        if entry_price is None:
            return
        # validate_quote must use the same local receipt clock as this
        # confirmation event; the exchange payload timestamp is diagnostic.
        effective_quote = Quote(price=entry_price, received_at=received, source=quote.source)
        ok, _reason = validate_quote(
            effective_quote, received, self.settings.quote_max_age_seconds,
            setup.reference, setup.atr, self.settings.max_entry_extension_atr,
        )
        if not ok:
            self._record_trigger_event(strategy, symbol, f"quote_validation_rejected_{_reason}")
            setup.trigger_candidate_at = None
            setup.trigger_candidate_last_at = None
            if setup.lifecycle_state == "CONFIRMING":
                self._transition_setup(setup, "ARMED", f"quote_validation_rejected_{_reason}")
            return
        crossed = entry_price >= setup.trigger_price if setup.side == "LONG" else entry_price <= setup.trigger_price
        if not crossed:
            if setup.trigger_candidate_at is not None:
                self._record_trigger_event(strategy, symbol, "trigger_confirmation_reversed")
            else:
                self._record_trigger_event(strategy, symbol, "trigger_not_crossed")
            setup.trigger_candidate_at = None
            setup.trigger_candidate_last_at = None
            if setup.lifecycle_state == "CONFIRMING":
                self._transition_setup(setup, "ARMED", "trigger_confirmation_reversed")
            return
        if setup.trigger_candidate_at is None:
            if setup.lifecycle_state == "ARMED" and not self._transition_setup(setup, "CONFIRMING", "quote_trigger_crossed"):
                return
            self._record_trigger_event(strategy, symbol, "confirmation_started")
            setup.trigger_candidate_at = received
            setup.trigger_candidate_last_at = received
            return
        if setup.trigger_candidate_last_at is None or received <= setup.trigger_candidate_last_at:
            return
        hold_seconds = max(0.0, float(getattr(self.settings, "trigger_confirmation_seconds", 0.5)))
        max_quote_gap = max(
            hold_seconds,
            float(getattr(self.settings, "trigger_confirmation_max_gap_seconds", 1.5)),
        )
        quote_gap = (received - setup.trigger_candidate_last_at).total_seconds()
        if quote_gap > max_quote_gap:
            # A single late quote must not complete a "held" breakout. Start a
            # fresh confirmation window and require another timely quote.
            self._record_trigger_event(strategy, symbol, "confirmation_gap_reset")
            setup.trigger_candidate_at = received
            setup.trigger_candidate_last_at = received
            return
        setup.trigger_candidate_last_at = received
        if (received - setup.trigger_candidate_at).total_seconds() < hold_seconds:
            return

        if not math.isfinite(setup.stop_loss):
            self._record_trigger_event(strategy, symbol, "rejected_nonfinite_stop")
            setup.trigger_candidate_at = None
            setup.trigger_candidate_last_at = None
            if setup.lifecycle_state == "CONFIRMING":
                self._transition_setup(setup, "ARMED", "rejected_nonfinite_stop")
            log.warning("Trigger rejected because stop-loss is non-finite symbol=%s strategy=%s", symbol, strategy)
            return
        if ((setup.side == "LONG" and setup.stop_loss >= entry_price)
                or (setup.side == "SHORT" and setup.stop_loss <= entry_price)):
            self._record_trigger_event(strategy, symbol, "rejected_stop_wrong_side")
            setup.trigger_candidate_at = None
            setup.trigger_candidate_last_at = None
            if setup.lifecycle_state == "CONFIRMING":
                self._transition_setup(setup, "ARMED", "rejected_stop_wrong_side")
            log.warning(
                "Trigger rejected because stop-loss is on the wrong side of entry symbol=%s strategy=%s",
                symbol, strategy,
            )
            return
        risk = (entry_price - setup.stop_loss) if setup.side == "LONG" else (setup.stop_loss - entry_price)
        if not math.isfinite(risk) or risk <= 0 or risk > 3.0 * setup.atr:
            risk_atr = risk / setup.atr if math.isfinite(setup.atr) and setup.atr > 0 else float("inf")
            risk_evidence = {
                "side": setup.side, "entry": round(float(entry_price), 10),
                "stop_loss": round(float(setup.stop_loss), 10),
                "risk": round(float(risk), 10) if math.isfinite(risk) else None,
                "atr": round(float(setup.atr), 10) if math.isfinite(setup.atr) else None,
                "risk_atr": round(float(risk_atr), 4) if math.isfinite(risk_atr) else None,
                "max_risk_atr": 3.0,
            }
            self._record_trigger_event(strategy, symbol, "rejected_structural_risk_cap", risk_evidence)
            setup.trigger_candidate_at = None
            setup.trigger_candidate_last_at = None
            if setup.lifecycle_state == "CONFIRMING":
                self._transition_setup(setup, "ARMED", "rejected_structural_risk_cap")
            log.info(
                "Trigger rejected by structural risk cap symbol=%s strategy=%s side=%s "
                "entry=%.10g stop_loss=%.10g risk=%.10g atr=%.10g risk_atr=%.4f max_risk_atr=3.0",
                symbol, strategy, setup.side, entry_price, setup.stop_loss, risk,
                setup.atr, risk_atr,
            )
            return
        target_1 = (entry_price + LIVE_TARGET_R_MULTIPLE * risk if setup.side == "LONG"
                    else entry_price - LIVE_TARGET_R_MULTIPLE * risk)
        outcome_tp2_r = float(getattr(self.settings, "outcome_tp2_r_multiple", 2.0))
        target_2 = (
            entry_price + outcome_tp2_r * risk
            if setup.side == "LONG" else entry_price - outcome_tp2_r * risk
        )
        outcome_horizon = int(getattr(self.settings, "outcome_horizon_bars", 12))
        outcome_fee = float(getattr(self.settings, "outcome_fee_pct", 0.0))
        outcome_slippage = float(getattr(self.settings, "outcome_slippage_pct", 0.0))
        key = (
            f"{symbol}:{strategy}:{setup.side}:{setup.candle_time.isoformat()}"
        )
        # Win the durable lifecycle fence before creating a deliverable alert.
        # A concurrent worker may have invalidated this setup after its in-memory
        # quote check; never enqueue unless CONFIRMING -> TRIGGER_CONFIRMED succeeds.
        if not self._transition_setup(setup, "TRIGGER_CONFIRMED", "held_breakout_confirmed"):
            self._record_trigger_event(strategy, symbol, "trigger_confirmation_fenced")
            return
        try:
            already_sent = key in self.sent_keys or self.store.is_sent(key)
            if not already_sent:
                alert = {
                    "symbol": symbol, "side": setup.side, "strategy": strategy,
                    "entry": entry_price, "stop_loss": setup.stop_loss,
                    "take_profit_1": target_1,
                    # Research-only outcome context; Telegram formatting still displays TP1 only.
                    "take_profit_2": target_2,
                    "atr": setup.atr,
                    "outcome_tp2_r_multiple": outcome_tp2_r,
                    "outcome_horizon_bars": outcome_horizon,
                    "outcome_fee_pct": outcome_fee,
                    "outcome_slippage_pct": outcome_slippage,
                    "setup_time": setup.candle_time.isoformat(), "reason": setup.reason,
                    "context_15m": setup.context_15m,
                    "trigger_confirmed_at": received.isoformat(),
                }
                self.store.enqueue_alert(key, alert)
        except Exception:
            log.exception("Unable to persist alert intent; failing closed symbol=%s strategy=%s", symbol, strategy)
            return
        self._transition_setup(setup, "ALERT_QUEUED", "telegram_outbox_enqueued")
        self._record_trigger_event(strategy, symbol, "trigger_confirmed")
        try:
            self.journal.update_lifecycle(
                strategy, symbol, setup.candle_time.isoformat(), "trigger_confirmed",
                float(entry_price), event_at=received.isoformat(),
                trigger_price=float(setup.trigger_price),
                setup_to_trigger_seconds=max(
                    0.0, (received - (setup.armed_at or setup.candle_time)).total_seconds()
                ),
                entry_extension_atr=abs(entry_price - setup.reference) / setup.atr,
                trigger_overshoot_atr=max(
                    0.0,
                    ((entry_price - setup.trigger_price) if setup.side == "LONG"
                     else (setup.trigger_price - entry_price)) / setup.atr,
                ),
                side=setup.side, stop_loss=setup.stop_loss,
                target_price=target_1, target_price_2=target_2, atr=setup.atr,
                horizon_bars=outcome_horizon, fee_pct=outcome_fee,
                slippage_pct=outcome_slippage, tp2_r_multiple=outcome_tp2_r,
            )
            if self.shadow_ml is not None and strategy in {"ORB", "MEX"}:
                try:
                    evaluation_id = self.journal.evaluation_id(
                        strategy, symbol, setup.candle_time.isoformat()
                    )
                    task = asyncio.create_task(
                        asyncio.to_thread(self.shadow_ml.capture_and_predict, evaluation_id)
                    )
                    self.shadow_ml_tasks.add(task)
                    task.add_done_callback(self._on_shadow_ml_task_done)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Shadow ML capture/prediction failed; confirmed signal is unaffected"
                    )
        except Exception:
            log.exception("Unable to update journal trigger lifecycle symbol=%s strategy=%s", symbol, strategy)
        if already_sent:
            self.store.transition_setup(
                symbol, strategy, setup.candle_time.isoformat(),
                "ALERT_QUEUED", "TELEGRAM_SENT", "existing_sent_alert_deduplicated",
                event_at=datetime.now(timezone.utc).isoformat(),
            )
            self.sent_keys.add(key)
        else:
            await self.deliver_outbox()
        self._remove_pending_setup(symbol, strategy, setup)

    async def deliver_outbox(self) -> None:
        """Deliver durable alert intents; retries survive process restarts."""
        async with self.outbox_lock:
            try:
                due = self.store.claim_due_alerts()
            except Exception:
                log.exception("Unable to read Telegram outbox")
                return
            for key, alert, lease_token in due:
                try:
                    self.telegram_delivery["attempts"] += 1
                    self.telegram_delivery["last_attempt_at"] = datetime.now(timezone.utc).isoformat()
                    # This deployment is outbound-only; do not emit inline feedback
                    # buttons that require an inbound Telegram webhook.
                    delivered = await self.notifier.send_signal(alert)
                    delivery = getattr(self.notifier, "last_delivery", {}) or {}
                    self.telegram_delivery["last_status"] = delivery.get("status", "sent" if delivered else "failed")
                    self.telegram_delivery["last_http_status"] = delivery.get("http_status")
                    self.telegram_delivery["last_error"] = delivery.get("error")
                    if not delivered:
                        self.telegram_delivery["failed"] += 1
                        self.store.schedule_retry(key, lease_token, 60, str(delivery.get("error") or "notifier returned false"))
                        log.warning("Telegram delivery failed key=%s status=%s", key, self.telegram_delivery["last_status"])
                        continue
                    if not self.store.mark_sent(key, alert, lease_token):
                        log.warning("Telegram outbox lease lost after delivery key=%s; stale worker state was ignored", key)
                        continue
                    try:
                        sent_at = datetime.now(timezone.utc)
                        trigger_at = alert.get("trigger_confirmed_at")
                        delivery_delay = None
                        if trigger_at:
                            try:
                                delivery_delay = max(
                                    0.0,
                                    (sent_at - datetime.fromisoformat(str(trigger_at))).total_seconds(),
                                )
                            except (TypeError, ValueError):
                                delivery_delay = None
                        self.journal.update_lifecycle(
                            str(alert.get("strategy", "UNKNOWN")), str(alert["symbol"]),
                            str(alert["setup_time"]), "telegram_sent",
                            event_at=sent_at.isoformat(),
                            trigger_confirmed_at=trigger_at,
                            trigger_to_telegram_seconds=delivery_delay,
                            side=alert.get("side"), stop_loss=alert.get("stop_loss"),
                            target_price=alert.get("take_profit_1"),
                            target_price_2=alert.get("take_profit_2"),
                            atr=alert.get("atr"),
                            horizon_bars=int(alert.get("outcome_horizon_bars") or getattr(self.settings, "outcome_horizon_bars", 12)),
                            fee_pct=float(alert.get("outcome_fee_pct") or getattr(self.settings, "outcome_fee_pct", 0.0)),
                            slippage_pct=float(alert.get("outcome_slippage_pct") or getattr(self.settings, "outcome_slippage_pct", 0.0)),
                            tp2_r_multiple=float(alert.get("outcome_tp2_r_multiple") or getattr(self.settings, "outcome_tp2_r_multiple", 2.0)),
                        )
                    except Exception:
                        log.exception("Unable to update journal Telegram-delivered lifecycle key=%s", key)
                    if str(alert.get("strategy", "")).upper() != "ADVIKA":
                        self.store.transition_setup(
                            str(alert["symbol"]), str(alert.get("strategy", "UNKNOWN")),
                            str(alert["setup_time"]), "ALERT_QUEUED", "TELEGRAM_SENT",
                            "telegram_delivery_confirmed", event_at=datetime.now(timezone.utc).isoformat(),
                        )
                    self.sent_keys.add(key)
                    self.telegram_delivery["sent"] += 1
                    self.telegram_delivery["last_success_at"] = datetime.now(timezone.utc).isoformat()
                    self.telegram_delivery["last_error"] = None
                    log.info("Durable Telegram alert delivered key=%s http_status=%s", key, self.telegram_delivery["last_http_status"])
                except Exception as exc:
                    self.telegram_delivery["failed"] += 1
                    safe_error = f"delivery_exception:{type(exc).__name__}"
                    self.telegram_delivery["last_status"] = "failed"
                    self.telegram_delivery["last_error"] = safe_error
                    log.error("Telegram outbox delivery failed key=%s error_type=%s", key, type(exc).__name__)
                    try:
                        self.store.schedule_retry(key, lease_token, 5, safe_error)
                    except Exception:
                        log.exception("Unable to persist Telegram retry schedule key=%s", key)

    async def refresh_context(self, symbol: str) -> None:
        now = datetime.now(timezone.utc)
        # 15m context is informational: refresh once per completed 15m bucket,
        # not every 30 seconds for every symbol. Retry unavailable context at
        # most once per minute within the same bucket.
        bucket = int(now.timestamp() // 900)
        previous_bucket = self.context_last_bucket.get(symbol)
        previous_fetch = self.context_last_fetch.get(symbol)
        if previous_bucket == bucket:
            if self.context_15m.get(symbol) != "UNAVAILABLE":
                return
            if previous_fetch and (now - previous_fetch).total_seconds() < 60:
                return
        self.context_last_bucket[symbol] = bucket
        self.context_last_fetch[symbol] = now
        bars = await self.rest.candles(symbol, "15m", 60)
        if len(bars) < 25:
            self.context_15m[symbol] = "UNAVAILABLE"
            return
        closes = pd.Series([bar.close for bar in bars], dtype="float64")
        ema9 = closes.ewm(span=9, adjust=False, min_periods=9).mean().iloc[-1]
        ema21 = closes.ewm(span=21, adjust=False, min_periods=21).mean().iloc[-1]
        if pd.isna(ema9) or pd.isna(ema21):
            trend = "UNAVAILABLE"
        elif ema9 > ema21:
            trend = "BULLISH"
        elif ema9 < ema21:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        self.context_15m[symbol] = trend

    async def refresh_symbol(self, symbol: str) -> None:
        # Keep the actionable 5m scan independent from the informational 15m
        # context fetch. The context refresher runs in its own low-concurrency
        # background task from run(), so slow context REST calls cannot delay
        # completed-candle evaluation across the 120-symbol universe.
        now = datetime.now(timezone.utc)
        poll_interval = max(0.0, float(getattr(self.settings, "candle_poll_interval_seconds", 30.0)))
        previous_fetch = self.last_candle_fetch_at.get(symbol)
        if previous_fetch and (now - previous_fetch).total_seconds() < poll_interval:
            return
        attempted_at = datetime.now(timezone.utc)
        self.last_candle_fetch_at[symbol] = attempted_at
        diagnostic = self.candle_diagnostics.setdefault(symbol, {})
        diagnostic["last_attempt_at"] = attempted_at.isoformat()
        try:
            previous_bars = self.candles.get(symbol)
            force_full = symbol in self.force_full_candle_refresh
            history_limit = ADVIKA_HISTORY_BARS if self._module_enabled("ADVIKA") else 120
            required_history = history_limit if self._module_enabled("ADVIKA") else 60
            if previous_bars and not force_full:
                # Re-fetch a two-bar overlap to absorb late exchange corrections,
                # avoiding another 120-bar payload on every polling cycle.
                since = max(0, int(previous_bars[-1].timestamp.timestamp()) - 600)
                updates = await self.rest.candles(symbol, "5m", history_limit, since=since)
                if updates and max(bar.timestamp for bar in updates) < previous_bars[-1].timestamp:
                    # A truncated/regressed incremental response cannot be merged
                    # safely. Fall back to full history before deciding readiness.
                    self.force_full_candle_refresh.add(symbol)
                    bars = await self.rest.candles(symbol, "5m", history_limit)
                else:
                    merged = {bar.timestamp: bar for bar in previous_bars}
                    merged.update({bar.timestamp: bar for bar in updates})
                    bars = sorted(merged.values(), key=lambda bar: bar.timestamp)[-history_limit:]
            else:
                bars = await self.rest.candles(symbol, "5m", history_limit)
        except Exception as exc:
            diagnostic["last_attempt_status"] = "error"
            diagnostic["last_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            if symbol not in self.candles:
                diagnostic.update(state="fetch_error", reason="rest_request_failed", bars=0)
            else:
                diagnostic["reason"] = "refresh_failed_using_cached_data"
            raise
        self.force_full_candle_refresh.discard(symbol)
        diagnostic["last_attempt_status"] = "success"
        diagnostic["last_error"] = None
        diagnostic["last_success_at"] = datetime.now(timezone.utc).isoformat()
        diagnostic["bars"] = len(bars)
        if not bars:
            if symbol not in self.candles:
                diagnostic.update(state="no_data", reason="empty_response", bars=0)
            else:
                diagnostic["reason"] = "empty_refresh_using_cached_data"
            return
        if len(bars) < required_history:
            if symbol not in self.candles:
                diagnostic.update(state="warming", reason="insufficient_history")
            else:
                diagnostic["reason"] = "insufficient_refresh_using_cached_data"
            diagnostic["last_bar_time"] = bars[-1].timestamp.isoformat()
            log.info("5m candle history warming symbol=%s bars=%s required=%s", symbol, len(bars), required_history)
            return
        latest = bars[-1]
        diagnostic["last_bar_time"] = latest.timestamp.isoformat()
        diagnostic["reason"] = "validating_history"
        previous_bars = self.candles.get(symbol)
        if previous_bars and latest.timestamp < previous_bars[-1].timestamp:
            diagnostic.update(state="regressed", reason="history_timestamp_regressed")
            self.force_full_candle_refresh.add(symbol)
            log.warning(
                "Ignoring regressed 5m history snapshot symbol=%s latest=%s previously_seen=%s",
                symbol, latest.timestamp.isoformat(), previous_bars[-1].timestamp.isoformat(),
            )
            return
        candle_close_time = latest.timestamp + timedelta(minutes=5)
        candle_lag = (datetime.now(timezone.utc) - candle_close_time).total_seconds()
        if candle_lag > self.settings.max_candle_lag_seconds:
            already_stale = diagnostic.get("state") == "stale"
            diagnostic.update(state="stale", reason="latest_completed_candle_too_old")
            # Keep the per-symbol diagnostic current without emitting 120 repeated
            # warnings every poll while the exchange history feed is delayed.
            if not already_stale:
                log.warning("Skipping stale 5m candles symbol=%s lag_seconds=%.1f", symbol, candle_lag)
            return
        recent = bars[-60:]
        if any((current.timestamp - previous.timestamp).total_seconds() != 300
               for previous, current in zip(recent, recent[1:])):
            diagnostic.update(state="gap", reason="missing_5m_intervals")
            self.candles[symbol] = bars
            self.force_full_candle_refresh.add(symbol)
            self._invalidate_pending_setups_for_symbol(
                symbol, "invalidated_missing_5m_candle_intervals",
            )
            log.warning("Skipping strategy evaluation due to missing 5m intervals symbol=%s", symbol)
            return
        self.candles[symbol] = bars
        diagnostic.update(state="ready", reason="60_plus_contiguous_fresh_candles")
        previous_evaluated = self.last_evaluated.get(symbol)
        if previous_evaluated is not None and latest.timestamp <= previous_evaluated:
            return

        current_strategy = "TPR"
        try:
            frame = pd.DataFrame([{
                "timestamp": b.timestamp, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "volume": b.volume,
            } for b in bars])
            momentum_specs = []
            if self._module_enabled("ORB") and symbol in self.tpr_rb_symbols:
                if getattr(self.settings, "tpr_enabled", True):
                    momentum_specs.append((
                        "TPR", lambda diagnostics, prepared: evaluate_tpr(
                            prepared, adx_min=float(getattr(self.settings, "adx_min", 17.0)),
                            diagnostics=diagnostics, prepared=True,
                        ),
                    ))
                if getattr(self.settings, "rb_enabled", True):
                    momentum_specs.append((
                        "RB", lambda diagnostics, prepared: evaluate_rb(
                            prepared, min_rvol=float(getattr(self.settings, "rb_min_rvol", 1.45)),
                            range_lookback=int(getattr(self.settings, "rb_range_lookback", 12)),
                            diagnostics=diagnostics, prepared=True,
                        ),
                    ))
            if (self._module_enabled("ORB") and symbol in self.configured_symbol_set
                    and getattr(self.settings, "orb_enabled", True)):
                orb_sessions = getattr(self.settings, "orb_session_config", None)
                momentum_specs.append((
                    "ORB",
                    lambda diagnostics, prepared: evaluate_orb(
                        prepared,
                        min_rvol=float(getattr(self.settings, "orb_min_rvol", 1.10)),
                        enabled_sessions=(
                            tuple(orb_sessions) if orb_sessions is not None
                            else ("asia", "london", "new_york")
                        ),
                        sessions=orb_sessions,
                        diagnostics=diagnostics,
                        prepared=True,
                    ),
                ))

            # Only prepare data when at least one completed-candle detector is
            # enabled for this symbol. All enabled detectors then share this
            # single vectorized validation + indicator calculation.
            prepared_frame = None
            if momentum_specs:
                validated_frame = _frame(frame)
                prepared_frame = _indicators(validated_frame) if validated_frame is not None else None

            for strategy, evaluator in momentum_specs:
                current_strategy = strategy
                key = (symbol, strategy)
                existing = self.pending.get(symbol) if strategy == "TPR" else self.momentum_pending.get(key)
                if existing is not None:
                    existing.bars_remaining -= 1
                    try:
                        self._persist_pending_setup(existing)
                    except Exception:
                        log.exception("Unable to update persisted %s setup symbol=%s", strategy, symbol)
                    if existing.bars_remaining < 0 or latest.timestamp >= existing.expires_at:
                        self._record_trigger_event(strategy, symbol, "setup_expired_on_candle")
                        self._transition_setup(existing, "EXPIRED", "setup_expired_on_candle")
                        self._remove_pending_setup(symbol, strategy, existing)
                        existing = None
                if existing is not None:
                    continue
                current_strategy = strategy
                diagnostics = {}
                if prepared_frame is None:
                    # Preserve per-strategy validation diagnostics if a malformed
                    # frame ever reaches this layer; normal market-data ingestion
                    # rejects malformed candles before this point.
                    result = None
                    diagnostics.update(stage="data_validation", reason="invalid_shared_ohlcv_frame")
                else:
                    result = evaluator(diagnostics, prepared_frame)
                if strategy == "MEX":
                    diagnostics["evaluation_source"] = "completed_candle"
                self._record_strategy_evaluation(strategy, latest, result, symbol, diagnostics)
                if result:
                    self._arm_setup(symbol, strategy, result, latest)

            if self._strategy_enabled("ADVIKA") and symbol in self.configured_symbol_set:
                current_strategy = "ADVIKA"
                advika_diagnostics = {}
                btc_bars = self.candles.get("BTCUSD")
                btc_frame = pd.DataFrame([{
                    "timestamp": b.timestamp, "open": b.open, "high": b.high,
                    "low": b.low, "close": b.close, "volume": b.volume,
                } for b in btc_bars]) if btc_bars else None
                advika_result = evaluate_advika_choice(
                    frame, symbol=symbol, btc_frame=btc_frame,
                    dead_hours_ist=getattr(self.settings, "advika_dead_hours_ist", ""),
                    diagnostics=advika_diagnostics,
                )
                self._record_strategy_evaluation(
                    "ADVIKA", latest, advika_result, symbol, advika_diagnostics,
                )
                if advika_result:
                    await self._queue_completed_candle_signal(symbol, latest, advika_result)

            self.last_evaluated[symbol] = latest.timestamp
        except Exception as exc:
            stats = self.strategy_evaluations[current_strategy]
            stats["errors"] += 1
            stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            stats["last_error_at"] = datetime.now(timezone.utc).isoformat()
            # A failed indicator/strategy evaluation must be retried promptly;
            # otherwise the REST poll throttle hides the same completed bar
            # until the next interval and can miss an actionable setup.
            self.last_candle_fetch_at.pop(symbol, None)
            raise

    def _arm_setup(self, symbol: str, strategy: str, result: dict, latest: Candle) -> None:
        trigger = result.get("trigger_price")
        if trigger is None:
            trigger = latest.high if result["side"] == "LONG" else latest.low
        reference = result.get("entry_reference", latest.close)
        setup = PendingSetup(
            symbol=symbol, side=result["side"], candle_time=latest.timestamp,
            trigger_price=float(trigger), reference=float(reference),
            stop_loss=float(result["stop_loss"]),
            take_profit_1=float(result["take_profit_1"]),
            atr=float(result["atr"]), bars_remaining=self.settings.signal_max_age_bars,
            reason=result["reason"], context_15m=self.context_15m.get(symbol, "UNAVAILABLE"),
            expires_at=latest.timestamp + timedelta(minutes=5 * (1 + self.settings.signal_max_age_bars)),
            strategy=strategy, armed_at=datetime.now(timezone.utc),
        )
        try:
            if not self.store.transition_setup(
                symbol, strategy, latest.timestamp.isoformat(), None, "ARMED",
                "setup_armed", event_at=(setup.armed_at or datetime.now(timezone.utc)).isoformat(),
            ):
                current = self.store.setup_lifecycle(symbol, strategy, latest.timestamp.isoformat())
                if not current or current.get("state") != "ARMED":
                    # A terminal lifecycle is a normal deduplication fence. Never
                    # persist a pending setup behind a terminal state.
                    self._record_trigger_event(strategy, symbol, "setup_lifecycle_already_terminal")
                    log.info(
                        "Setup arming skipped by terminal lifecycle fence symbol=%s strategy=%s state=%s",
                        symbol, strategy, current.get("state") if current else "missing",
                    )
                    return
                # Another scanner pass already created the ARMED lifecycle.
                # Re-run the idempotent pending persistence step to heal a
                # possible interruption between lifecycle creation and save.
                self._record_trigger_event(strategy, symbol, "setup_already_armed")
            self._persist_pending_setup(setup)
        except Exception:
            try:
                self.store.transition_setup(
                    symbol, strategy, latest.timestamp.isoformat(), "ARMED", "INVALIDATED",
                    "setup_persistence_failed",
                )
            except Exception:
                log.exception("Unable to record setup persistence failure")
            log.exception("Unable to persist new setup; refusing to arm symbol=%s strategy=%s", symbol, strategy)
            self._record_trigger_event(strategy, symbol, "setup_persistence_failed")
            # SignalJournal accepts only evaluated/rejected/setup_armed/
            # trigger_confirmed/telegram_sent lifecycle values. Persistence
            # failure belongs in trigger diagnostics, not as a fake lifecycle.
            return
        if strategy == "TPR":
            self.pending[symbol] = setup
        else:
            self.momentum_pending[(symbol, strategy)] = setup
        try:
            self.journal.update_lifecycle(
                strategy, symbol, latest.timestamp.isoformat(), "setup_armed",
                event_at=(setup.armed_at or datetime.now(timezone.utc)).isoformat(),
            )
        except Exception:
            log.exception("Unable to update journal lifecycle for armed setup symbol=%s strategy=%s", symbol, strategy)
        log.info(
            "Setup armed symbol=%s strategy=%s side=%s candle=%s trigger=%.8f",
            symbol, strategy, result["side"], latest.timestamp.isoformat(), trigger,
        )

    async def _queue_completed_candle_signal(self, symbol: str, latest: Candle, result: dict) -> None:
        """Queue Advika's close-confirmed signal without the legacy quote trigger."""
        strategy = "ADVIKA"
        candle_time = latest.timestamp.isoformat()
        key = f"{symbol}:{strategy}:{result['side']}:{candle_time}"
        try:
            if self.store.is_sent(key):
                return
            alert = {
                "symbol": symbol, "side": result["side"], "strategy": strategy,
                "entry": float(result["entry_reference"]),
                "stop_loss": float(result["stop_loss"]),
                "take_profit_1": float(result["take_profit_1"]),
                "atr": float(result["atr"]),
                "setup_time": candle_time,
                "reason": result["reason"],
                "context_15m": "Not used by Advika Choice",
                "trigger_confirmed_at": result["candle_close_time"],
                "risk_percent_cap": 1.0,
                "position_size_formula": result["position_size_formula"],
                "partial_exit_percent": 50,
                "trailing_stop": result["trailing_stop"],
                "early_exit": result["early_exit"],
                "ema20": result["ema20"],
                "adx": result["adx"], "rsi": result["rsi"],
                "macd": result["macd"], "macd_signal": result["macd_signal"],
                "macd_histogram": result["macd_histogram"],
                "relative_volume": result["relative_volume"],
                "outcome_horizon_bars": int(getattr(self.settings, "outcome_horizon_bars", 12)),
                "outcome_fee_pct": float(getattr(self.settings, "outcome_fee_pct", 0.0)),
                "outcome_slippage_pct": float(getattr(self.settings, "outcome_slippage_pct", 0.0)),
                "outcome_tp2_r_multiple": float(getattr(self.settings, "outcome_tp2_r_multiple", 2.0)),
            }
            # Persist the notification intent before attempting network delivery.
            self.store.enqueue_alert(key, alert)
            self.journal.update_lifecycle(
                strategy, symbol, candle_time, "trigger_confirmed",
                float(result["entry_reference"]), event_at=result["candle_close_time"],
                side=result["side"], stop_loss=float(result["stop_loss"]),
                target_price=float(result["take_profit_1"]), atr=float(result["atr"]),
                horizon_bars=int(getattr(self.settings, "outcome_horizon_bars", 12)),
                fee_pct=float(getattr(self.settings, "outcome_fee_pct", 0.0)),
                slippage_pct=float(getattr(self.settings, "outcome_slippage_pct", 0.0)),
                tp2_r_multiple=float(getattr(self.settings, "outcome_tp2_r_multiple", 2.0)),
            )
            self.advika_signal_count += 1
            self.advika_last_signal_at = result["candle_close_time"]
            log.info(
                "Advika Choice close signal queued symbol=%s side=%s entry=%.10g stop=%.10g target_1_5r=%.10g",
                symbol, result["side"], result["entry_reference"],
                result["stop_loss"], result["take_profit_1"],
            )
            await self.deliver_outbox()
        except Exception:
            log.exception("Advika Choice signal queue failed symbol=%s candle=%s", symbol, candle_time)

    async def run(self) -> None:
        stream_task = None
        context_task = None
        quote_task = None
        last_validation_monotonic = 0.0
        semaphore = asyncio.Semaphore(4)
        context_semaphore = asyncio.Semaphore(1)

        async def refresh_context_one(symbol: str) -> None:
            async with context_semaphore:
                try:
                    await self.refresh_context(symbol)
                except Exception:
                    log.exception(
                        "15m context refresh failed for %s; continuing with 5m scanner",
                        symbol,
                    )
                    self.context_15m[symbol] = "UNAVAILABLE"

        async def refresh_context_batch() -> None:
            await asyncio.gather(
                *(refresh_context_one(symbol) for symbol in self.active_symbols)
            )

        async def refresh_one(symbol: str) -> bool:
            async with semaphore:
                try:
                    await self.refresh_symbol(symbol)
                    return True
                except Exception:
                    log.exception("Candle refresh failed for %s", symbol)
                    return False

        async def refresh_rest_quotes() -> None:
            """Refresh stale quote snapshots at a bounded cadence; REST never confirms triggers."""
            last_error_log = 0.0
            next_fallback_poll = 0.0
            while True:
                try:
                    received = datetime.now(timezone.utc)
                    # The WebSocket is the primary L1 feed. Avoid downloading and
                    # scanning the entire ticker universe every second while all
                    # subscribed symbols are already receiving fresh quotes.
                    fallback_after = max(
                        0.5, min(1.5, float(self.settings.quote_max_age_seconds) * 0.5)
                    )
                    stale_symbols = _stale_quote_symbols(
                        self.active_symbols, self.received_at, received, fallback_after,
                    )
                    if not stale_symbols:
                        await asyncio.sleep(1.0)
                        continue
                    now_monotonic = time.monotonic()
                    if now_monotonic < next_fallback_poll:
                        await asyncio.sleep(min(1.0, next_fallback_poll - now_monotonic))
                        continue
                    # One stale symbol must not cause a full-market REST ticker
                    # download every second. REST quotes are diagnostic/freshness
                    # snapshots only; WebSocket updates alone advance triggers.
                    next_fallback_poll = now_monotonic + 5.0
                    rows = await self.rest.tickers()
                    updated = 0
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        symbol = str(row.get("symbol") or "").upper()
                        if symbol not in stale_symbols:
                            continue
                        last_received = self.received_at.get(symbol)
                        if last_received is not None and (received - last_received).total_seconds() < 0.8:
                            continue
                        book = row.get("quotes") or {}
                        if not isinstance(book, dict):
                            continue
                        try:
                            bid = float(book.get("best_bid"))
                            ask = float(book.get("best_ask"))
                        except (TypeError, ValueError):
                            continue
                        if (not math.isfinite(bid) or not math.isfinite(ask)
                                or bid <= 0 or ask <= 0 or ask < bid):
                            continue
                        quote = Quote(
                            price=(bid + ask) / 2.0, received_at=received,
                            bid=bid, ask=ask, source="rest",
                        )
                        await self.on_quote(symbol, quote, received)
                        updated += 1
                    self.rest_quote_polls += 1
                    self.rest_quote_updates += updated
                    self.last_rest_quote_poll_at = received.isoformat()
                    self.last_rest_quote_error = None
                    if self.rest_quote_polls == 1 or self.rest_quote_polls % 30 == 0:
                        log.info(
                            "Delta REST ticker fallback poll=%s updated=%s active=%s",
                            self.rest_quote_polls, updated, len(self.active_symbols),
                        )
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.last_rest_quote_error = f"{type(exc).__name__}: {str(exc)[:180]}"
                    now_mono = time.monotonic()
                    if now_mono - last_error_log >= 15.0:
                        log.warning(
                            "Delta REST ticker fallback failed error_type=%s",
                            type(exc).__name__,
                        )
                        last_error_log = now_mono
                    await asyncio.sleep(2.0)

        try:
            while True:
                if not self.active_symbols and time.monotonic() - last_validation_monotonic >= 60.0:
                    last_validation_monotonic = time.monotonic()
                    await self.validate_symbol_universe()
                if self.active_symbols and stream_task is None:
                    stream_task = asyncio.create_task(self.stream.run_forever(), name="delta-public-ws")
                if self.active_symbols and quote_task is None:
                    quote_task = asyncio.create_task(refresh_rest_quotes(), name="delta-rest-ticker-fallback")
                if self.active_symbols:
                    if context_task is None or context_task.done():
                        context_task = asyncio.create_task(
                            refresh_context_batch(), name="delta-15m-context-refresh"
                        )
                    cycle_started = time.monotonic()
                    if self.scan_cycles == 0 or self.scan_cycles % 6 == 0:
                        log.info(
                            "Scanner cycle starting cycle=%s active_symbols=%s",
                            self.scan_cycles + 1, len(self.active_symbols),
                        )
                    if self._strategy_enabled("ADVIKA") and "BTCUSD" in self.active_symbols:
                        # Refresh BTC's closed 5m bar first so the alt-long EMA50
                        # gate compares the same candle timestamp across symbols.
                        btc_ok = await refresh_one("BTCUSD")
                        other_results = await asyncio.gather(
                            *(refresh_one(symbol) for symbol in self.active_symbols if symbol != "BTCUSD")
                        )
                        results = [btc_ok, *other_results]
                    else:
                        results = await asyncio.gather(
                            *(refresh_one(symbol) for symbol in self.active_symbols)
                        )
                    self.scan_cycles += 1
                    self.last_scan_cycle_at = datetime.now(timezone.utc).isoformat()
                    self.last_scan_cycle_duration_seconds = round(time.monotonic() - cycle_started, 3)
                    self.last_scan_cycle_errors = sum(1 for result in results if not result)
                    if self.scan_cycles == 1 or self.scan_cycles % 6 == 0:
                        evaluations = {
                            name: stats["evaluated"]
                            for name, stats in self.strategy_evaluations.items()
                        }
                        armed = {
                            name: stats["setups_armed"]
                            for name, stats in self.strategy_evaluations.items()
                        }
                        rejection_top = {
                            name: sorted(
                                self.strategy_diagnostics[name]["rejections_by_reason"].items(),
                                key=lambda item: item[1], reverse=True,
                            )[:3]
                            for name in self.strategy_diagnostics
                        }
                        # Emit the individual MEX gates and a tiny bounded sample
                        # so production diagnosis does not depend on manually
                        # fetching the /status payload. This is observability only.
                        gate_counts = {
                            name: {
                                "failed": dict(self.strategy_diagnostics[name].get("detail_counts", {})),
                                "mex_gate_outcomes": dict(self.strategy_diagnostics[name].get("mex_gate_counts", {})),
                            }
                            for name in self.strategy_diagnostics
                        }
                        rejection_samples = {}
                        for name, summary in self.strategy_diagnostics.items():
                            # Keep one latest sample per rejection reason so high-volume
                            # flat-candle rejects cannot hide TPR regime diagnostics.
                            selected = []
                            seen_reasons = set()
                            reason_samples = list(summary.get("last_rejection_by_reason", {}).values())
                            if not reason_samples:
                                reason_samples = summary.get("recent_rejections", [])
                            for sample in reversed(reason_samples):
                                reason = sample.get("reason", "unknown")
                                if reason in seen_reasons:
                                    continue
                                seen_reasons.add(reason)
                                selected.append(sample)
                                if len(selected) >= 6:
                                    break
                            if selected:
                                rejection_samples[name] = selected
                        log.info(
                            "Scanner cycle complete cycle=%s active_symbols=%s candle_ready=%s "
                            "cycle_errors=%s duration_seconds=%.3f websocket_connected=%s "
                            "websocket_reconnects=%s evaluations=%s setups_armed=%s "
                            "top_rejections=%s gate_counts=%s rejection_samples=%s candle_volume_feed=%s",
                            self.scan_cycles, len(self.active_symbols),
                            sum(1 for item in self.candle_diagnostics.values()
                                if item.get("state") == "ready"),
                            self.last_scan_cycle_errors, self.last_scan_cycle_duration_seconds,
                            self.stream.connected, self.stream.reconnects,
                            evaluations, armed, rejection_top, gate_counts, rejection_samples,
                            self.stream.candle_volume_diagnostics(),
                        )
                await self.deliver_outbox()
                await asyncio.sleep(10)
        finally:
            for task in (stream_task, context_task, quote_task):
                if task is not None:
                    task.cancel()
            for task in (stream_task, context_task, quote_task):
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            await self.rest.close()

    def health(self) -> dict:
        now = datetime.now(timezone.utc)
        try:
            state_store_ok = self.store.health()
        except Exception:
            state_store_ok = False
        quote_ages = {
            symbol: round((now - self.received_at[symbol]).total_seconds(), 3)
            for symbol in self.received_at
        }
        # Operational readiness measures when the application last received
        # a quote update (not the exchange's last-change timestamp). The L1
        # feed can remain quiet while the book is unchanged, so allow its
        # documented cadence plus buffer. Trigger confirmation independently
        # requires a locally delivered update no older than quote_max_age_seconds.
        quote_health_max_age = max(7.0, float(self.settings.quote_max_age_seconds))
        entry_stale_quote_symbols = sorted(
            symbol for symbol, age in quote_ages.items()
            if age > self.settings.quote_max_age_seconds
        )
        heartbeat_age = (
            round((now - self.stream.last_heartbeat_at).total_seconds(), 3)
            if self.stream.last_heartbeat_at else None
        )
        module_setup_types = {"ORB": ("ORB", "RB", "TPR"), "MEX": ("MEX",), "ADVIKA": ("ADVIKA",)}
        module_setup_types = {name: types for name, types in module_setup_types.items() if self._module_enabled(name)}
        module_evaluations = {}
        module_diagnostics = {}
        module_triggers = {}
        for module, setup_types in module_setup_types.items():
            setup_stats = {name: self.strategy_evaluations[name] for name in setup_types}
            module_evaluations[module] = {
                "evaluated": sum(item["evaluated"] for item in setup_stats.values()),
                "setups_armed": sum(item["setups_armed"] for item in setup_stats.values()),
                "no_setup": sum(item["no_setup"] for item in setup_stats.values()),
                "errors": sum(item["errors"] for item in setup_stats.values()),
                "setup_types": {name: dict(item) for name, item in setup_stats.items()},
            }
            setup_diagnostics = {name: self.strategy_diagnostics[name] for name in setup_types}
            rejection_totals = {}
            for item in setup_diagnostics.values():
                for reason, count in item["rejections_by_reason"].items():
                    rejection_totals[reason] = rejection_totals.get(reason, 0) + count
            module_diagnostics[module] = {
                "rejections": sum(item["rejections"] for item in setup_diagnostics.values()),
                "rejections_by_reason": rejection_totals,
                "setup_types": {
                    name: {
                        "rejections": item["rejections"],
                        "rejections_by_reason": dict(item["rejections_by_reason"]),
                        "detail_counts": dict(item.get("detail_counts", {})),
                        "mex_gate_counts": dict(item.get("mex_gate_counts", {})),
                        "mex_evaluation_sources": {
                            source: dict(counts)
                            for source, counts in item.get("mex_evaluation_sources", {}).items()
                        },
                        "last_rejection_by_reason": dict(item.get("last_rejection_by_reason", {})),
                        "recent_rejections": list(item.get("recent_rejections", [])),
                        "last_rejection": dict(item["last_rejection"]) if item["last_rejection"] else None,
                    }
                    for name, item in setup_diagnostics.items()
                },
            }
            setup_triggers = {name: self.trigger_lifecycle[name] for name in setup_types}
            trigger_totals = {}
            for item in setup_triggers.values():
                for reason, count in item["events"].items():
                    trigger_totals[reason] = trigger_totals.get(reason, 0) + count
            module_triggers[module] = {
                "events": trigger_totals,
                "setup_types": {
                    name: {
                        "events": dict(item["events"]),
                        "last_event": dict(item["last_event"]) if item["last_event"] else None,
                    }
                    for name, item in setup_triggers.items()
                },
            }
        pending_by_setup_type = {
            "TPR": len(self.pending),
            **{
                strategy: sum(1 for _, pending_strategy in self.momentum_pending
                              if pending_strategy == strategy)
                for strategy in ("RB", "ORB", "MEX", "ADVIKA")
            },
        }
        pending_by_module = {
            "ORB": pending_by_setup_type["TPR"] + pending_by_setup_type["RB"] + pending_by_setup_type["ORB"],
            "MEX": pending_by_setup_type["MEX"],
            "ADVIKA": pending_by_setup_type["ADVIKA"],
        }
        return {
            "websocket_connected": self.stream.connected,
            "websocket_reconnects": self.stream.reconnects,
            "last_ws_connected_at": self.stream.last_connected_at.isoformat() if self.stream.last_connected_at else None,
            "last_ws_disconnect_at": self.stream.last_disconnect_at.isoformat() if self.stream.last_disconnect_at else None,
            "last_ws_disconnect_reason": self.stream.last_disconnect_reason,
            "websocket_heartbeat_age_seconds": heartbeat_age,
            "websocket_heartbeat_stale": bool(
                self.stream.connected and (heartbeat_age is None or heartbeat_age > 35.0)
            ),
            "quote_age_seconds": quote_ages,
            "quote_rejection_diagnostics": {
                "counts": dict(self.quote_rejection_diagnostics["counts"]),
                "recent": list(self.quote_rejection_diagnostics["recent"]),
                "last": self.quote_rejection_diagnostics["last"],
            },
            "quote_health_max_age_seconds": quote_health_max_age,
            "quote_stale_symbols": sorted(
                symbol for symbol, age in quote_ages.items()
                if age > quote_health_max_age
            ),
            "entry_quote_max_age_seconds": self.settings.quote_max_age_seconds,
            "entry_quote_stale_symbols": entry_stale_quote_symbols,
            "last_quote_received_at": {
                symbol: stamp.isoformat() for symbol, stamp in self.received_at.items()
            },
            "last_ws_message_at": self.stream.last_message_at.isoformat() if self.stream.last_message_at else None,
            "live_candlestick_updates": self.stream.candlestick_updates,
            "live_candlestick_volume": self.stream.candle_volume_diagnostics(),
            "last_live_candlestick_at": (
                self.stream.last_candlestick_at.isoformat() if self.stream.last_candlestick_at else None
            ),
            "live_candlestick_symbols": sorted(self.stream.last_candlestick_by_symbol),
            "last_ws_heartbeat_at": (
                self.stream.last_heartbeat_at.isoformat() if self.stream.last_heartbeat_at else None
            ),
            "websocket_heartbeats_received": self.stream.heartbeat_messages_received,
            "symbols_with_quotes": len(self.quotes),
            "rest_quote_polls": self.rest_quote_polls,
            "rest_quote_updates": self.rest_quote_updates,
            "last_rest_quote_poll_at": self.last_rest_quote_poll_at,
            "last_rest_quote_error": self.last_rest_quote_error,
            "stale_quotes": sum(
                1 for symbol in self.quotes
                if (now - self.received_at[symbol]).total_seconds() > quote_health_max_age
            ),
            "entry_stale_quotes": len(entry_stale_quote_symbols),
            "symbols_with_candles": len(self.candles),
            "configured_symbol_count": len(self.configured_symbols),
            "active_symbol_count": len(self.active_symbols),
            "unavailable_symbols": list(self.unavailable_symbols),
            "symbol_validation": dict(self.symbol_validation),
            "candle_ready_count": sum(1 for symbol in self.configured_symbols if self.candle_diagnostics.get(symbol, {}).get("state") == "ready"),
            "candle_warming_count": sum(1 for symbol in self.configured_symbols if self.candle_diagnostics.get(symbol, {}).get("state") == "warming"),
            "candle_no_data_count": sum(1 for symbol in self.configured_symbols if self.candle_diagnostics.get(symbol, {}).get("state") == "no_data"),
            "candle_error_count": sum(1 for symbol in self.configured_symbols if self.candle_diagnostics.get(symbol, {}).get("state") == "fetch_error" or self.candle_diagnostics.get(symbol, {}).get("last_attempt_status") == "error"),
            "candle_warming_symbols": [symbol for symbol in self.configured_symbols if self.candle_diagnostics.get(symbol, {}).get("state") == "warming"],
            "candle_no_data_symbols": [symbol for symbol in self.configured_symbols if self.candle_diagnostics.get(symbol, {}).get("state") == "no_data"],
            "candle_error_symbols": [symbol for symbol in self.configured_symbols if self.candle_diagnostics.get(symbol, {}).get("state") == "fetch_error" or self.candle_diagnostics.get(symbol, {}).get("last_attempt_status") == "error"],
            "candle_diagnostics": {symbol: dict(self.candle_diagnostics.get(symbol, {})) for symbol in self.configured_symbols},
            "quote_missing_symbols": [symbol for symbol in self.configured_symbols if symbol not in self.quotes],
            "candle_gap_symbols": sum(
                1 for bars in self.candles.values()
                if len(bars) >= 2 and any(
                    (current.timestamp - previous.timestamp).total_seconds() != 300
                    for previous, current in zip(bars[-60:], bars[-59:])
                )
            ),
            "candle_gap_list": [symbol for symbol, diag in self.candle_diagnostics.items() if diag.get("state") == "gap"],
            "stale_candle_symbols": sum(
                1 for symbol, bars in self.candles.items()
                if bars and (now - (bars[-1].timestamp + timedelta(minutes=5))).total_seconds() > self.settings.max_candle_lag_seconds
            ),
            "stale_candle_list": [symbol for symbol, bars in self.candles.items() if bars and (now - (bars[-1].timestamp + timedelta(minutes=5))).total_seconds() > self.settings.max_candle_lag_seconds],
            "symbols_with_15m_context": len(self.context_15m),
            "strategy_modules": list(module_setup_types),
            "strategy_evaluations": module_evaluations,
            "trigger_lifecycle": module_triggers,
            "strategy_diagnostics": module_diagnostics,
            "telegram_delivery": dict(self.telegram_delivery),
            "pending_setups": len(self.pending) + len(self.momentum_pending),
            "persisted_pending_setups": self.store.pending_setup_count() if state_store_ok else None,
            "setup_lifecycle": self.store.setup_lifecycle_health() if state_store_ok else None,
            "pending_setup_recovery": dict(self.pending_restore),
            "pending_by_module": pending_by_module,
            "pending_by_setup_type": pending_by_setup_type,
            "dedupe": self.store.backend,
            "database_backend": self.store.backend,
            "state_store_ok": state_store_ok,
            "telegram_outbox": self.store.outbox_health() if state_store_ok else None,
            "state_db_path": str(self.settings.state_db_path) if self.store.backend == "sqlite" else None,
            "max_candle_lag_seconds": self.settings.max_candle_lag_seconds,
            "scanner_cycles": self.scan_cycles,
            "last_scan_cycle_at": self.last_scan_cycle_at,
            "last_scan_cycle_duration_seconds": self.last_scan_cycle_duration_seconds,
            "last_scan_cycle_errors": self.last_scan_cycle_errors,
        }
