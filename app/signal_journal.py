"""Durable, reviewable signal-decision journal (PostgreSQL in Render; SQLite locally)."""
import csv
import io
import json
import math
import sqlite3
import secrets
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _utc_datetime(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class SignalJournal:
    """Stores one immutable decision snapshot per strategy/symbol/5m candle."""

    def __init__(self, path: str, database_url: str = "", migrate_on_startup: bool = True):
        self.database_url = (database_url or "").strip()
        self.backend = "postgresql" if self.database_url else "sqlite"
        self.migrate_on_startup = bool(migrate_on_startup)
        self.path = path
        if self.backend == "sqlite":
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        if self.backend == "postgresql":
            import psycopg
            from .database import postgres_conninfo
            return psycopg.connect(postgres_conninfo(self.database_url), connect_timeout=5)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self):
        from .migrations import apply_migrations, verify_migrations
        migrate = apply_migrations if self.migrate_on_startup else verify_migrations
        migrate(self._connect, self.backend)

    @staticmethod
    def evaluation_id(strategy: str, symbol: str, candle_time: str) -> str:
        return f"{strategy}:{symbol}:{candle_time}"

    def record_evaluation(self, strategy: str, symbol: str, candle_time: str,
                          decision: str, diagnostics: dict | None = None,
                          signal: dict | None = None):
        signal = signal or {}
        values = (
            self.evaluation_id(strategy, symbol, candle_time), strategy, symbol,
            candle_time, _now(), decision, signal.get("side"),
            (diagnostics or {}).get("reason"),
            json.dumps(diagnostics or {}, separators=(",", ":"), default=str),
            json.dumps(signal, separators=(",", ":"), default=str),
            "setup_armed" if decision == "setup" else "rejected",
            signal.get("entry_reference"), signal.get("stop_loss"),
            signal.get("take_profit_1"),
        )
        if self.backend == "sqlite":
            sql = """INSERT INTO signal_journal
              (evaluation_id,strategy,symbol,candle_time,evaluated_at,decision,side,reason,
               diagnostics_json,signal_json,lifecycle,entry_price,stop_loss,target_price)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
              ON CONFLICT(evaluation_id) DO UPDATE SET
                evaluated_at=excluded.evaluated_at, decision=excluded.decision,
                side=excluded.side, reason=excluded.reason,
                diagnostics_json=excluded.diagnostics_json, signal_json=excluded.signal_json,
                lifecycle=CASE WHEN signal_journal.lifecycle='trigger_confirmed'
                               THEN signal_journal.lifecycle ELSE excluded.lifecycle END,
                entry_price=COALESCE(signal_journal.entry_price,excluded.entry_price),
                stop_loss=COALESCE(signal_journal.stop_loss,excluded.stop_loss),
                target_price=COALESCE(signal_journal.target_price,excluded.target_price)
              WHERE signal_journal.lifecycle NOT IN ('trigger_confirmed','telegram_sent')
                AND (signal_journal.decision != 'setup' OR excluded.decision = 'setup')"""
        else:
            sql = """INSERT INTO signal_journal
              (evaluation_id,strategy,symbol,candle_time,evaluated_at,decision,side,reason,
               diagnostics_json,signal_json,lifecycle,entry_price,stop_loss,target_price)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
              ON CONFLICT (evaluation_id) DO UPDATE SET
                evaluated_at=EXCLUDED.evaluated_at, decision=EXCLUDED.decision,
                side=EXCLUDED.side, reason=EXCLUDED.reason,
                diagnostics_json=EXCLUDED.diagnostics_json, signal_json=EXCLUDED.signal_json,
                lifecycle=CASE WHEN signal_journal.lifecycle='trigger_confirmed'
                               THEN signal_journal.lifecycle ELSE EXCLUDED.lifecycle END,
                entry_price=COALESCE(signal_journal.entry_price,EXCLUDED.entry_price),
                stop_loss=COALESCE(signal_journal.stop_loss,EXCLUDED.stop_loss),
                target_price=COALESCE(signal_journal.target_price,EXCLUDED.target_price)
              WHERE signal_journal.lifecycle NOT IN ('trigger_confirmed','telegram_sent')
                AND (signal_journal.decision != 'setup' OR EXCLUDED.decision = 'setup')"""
        with self._connect() as db:
            db.execute(sql, values)

    def update_lifecycle(
        self, strategy: str, symbol: str, candle_time: str,
        lifecycle: str, entry_price: float | None = None,
        *, event_at: str | None = None, trigger_price: float | None = None,
        setup_to_trigger_seconds: float | None = None,
        entry_extension_atr: float | None = None,
        trigger_overshoot_atr: float | None = None,
        trigger_to_telegram_seconds: float | None = None,
        trigger_confirmed_at: str | None = None,
        side: str | None = None, stop_loss: float | None = None,
        target_price: float | None = None, target_price_2: float | None = None,
        atr: float | None = None, horizon_bars: int = 12,
        fee_pct: float = 0.0, slippage_pct: float = 0.0,
        tp2_r_multiple: float = 2.0,
    ):
        """Advance lifecycle and atomically seed immutable outcome context."""
        lifecycle_rank = {
            "evaluated": 0,
            "rejected": 0,
            "setup_armed": 1,
            "trigger_confirmed": 2,
            "telegram_sent": 3,
        }
        if lifecycle not in lifecycle_rank:
            raise ValueError(f"Unknown signal lifecycle: {lifecycle}")
        requested_rank = lifecycle_rank[lifecycle]
        ph = "?" if self.backend == "sqlite" else "%s"
        current_rank_sql = (
            "CASE lifecycle WHEN 'telegram_sent' THEN 3 "
            "WHEN 'trigger_confirmed' THEN 2 WHEN 'setup_armed' THEN 1 ELSE 0 END"
        )
        values_by_column = {
            "setup_armed_at": event_at if lifecycle == "setup_armed" else None,
            "trigger_confirmed_at": trigger_confirmed_at or (
                event_at if lifecycle == "trigger_confirmed" else None
            ),
            "telegram_sent_at": event_at if lifecycle == "telegram_sent" else None,
            "trigger_price": trigger_price,
            "setup_to_trigger_seconds": setup_to_trigger_seconds,
            "entry_extension_atr": entry_extension_atr,
            "trigger_overshoot_atr": trigger_overshoot_atr,
            "trigger_to_telegram_seconds": trigger_to_telegram_seconds,
        }
        assignments = [
            f"lifecycle=CASE WHEN {current_rank_sql} <= {ph} THEN {ph} ELSE lifecycle END",
            f"entry_price=COALESCE({ph},entry_price)",
        ]
        # stop_loss and target_price remain the original decision snapshot;
        # confirmed execution context is stored separately in signal_outcomes.
        values = [requested_rank, lifecycle, entry_price]
        for column, value in values_by_column.items():
            assignments.append(f"{column}=COALESCE({column},{ph})")
            values.append(value)
        evaluation_id = self.evaluation_id(strategy, symbol, candle_time)
        values.extend([evaluation_id, requested_rank])
        sql = (
            f"UPDATE signal_journal SET {', '.join(assignments)} "
            f"WHERE evaluation_id={ph} AND {current_rank_sql} <= {ph}"
        )
        with self._connect() as db:
            updated = db.execute(sql, tuple(values))
            if updated.rowcount < 1:
                return
            if lifecycle not in {"trigger_confirmed", "telegram_sent"}:
                return

            context = db.execute(
                f"SELECT side,entry_price,stop_loss,target_price,signal_json,trigger_confirmed_at "
                f"FROM signal_journal WHERE evaluation_id={ph}",
                (evaluation_id,),
            ).fetchone()
            if not context:
                return
            resolved_side = str(side or context[0] or "").upper()
            resolved_entry = float(entry_price if entry_price is not None else context[1] or 0)
            resolved_stop = float(stop_loss if stop_loss is not None else context[2] or 0)
            resolved_tp1 = float(target_price if target_price is not None else context[3] or 0)
            if resolved_side not in {"LONG", "SHORT"} or resolved_entry <= 0 or resolved_stop <= 0:
                return
            risk = resolved_entry - resolved_stop if resolved_side == "LONG" else resolved_stop - resolved_entry
            if not math.isfinite(risk) or risk <= 0:
                return
            if resolved_tp1 <= 0:
                resolved_tp1 = (
                    resolved_entry + 1.5 * risk if resolved_side == "LONG"
                    else resolved_entry - 1.5 * risk
                )
            resolved_tp2 = float(target_price_2 or (
                resolved_entry + tp2_r_multiple * risk if resolved_side == "LONG"
                else resolved_entry - tp2_r_multiple * risk
            ))
            if resolved_side == "LONG" and not (resolved_entry < resolved_tp1 < resolved_tp2):
                resolved_tp1 = resolved_entry + 1.5 * risk
                resolved_tp2 = resolved_entry + tp2_r_multiple * risk
            elif resolved_side == "SHORT" and not (resolved_entry > resolved_tp1 > resolved_tp2):
                resolved_tp1 = resolved_entry - 1.5 * risk
                resolved_tp2 = resolved_entry - tp2_r_multiple * risk
            try:
                signal = json.loads(context[4] or "{}")
            except (TypeError, ValueError):
                signal = {}
            resolved_atr = atr or signal.get("atr") or signal.get("atr_value")
            try:
                resolved_atr = float(resolved_atr) if resolved_atr is not None else None
                if resolved_atr is not None and (
                    not math.isfinite(resolved_atr) or resolved_atr <= 0
                ):
                    resolved_atr = None
            except (TypeError, ValueError):
                resolved_atr = None
            resolved_trigger_at = (
                trigger_confirmed_at or context[5] or
                (event_at if lifecycle == "trigger_confirmed" else None)
            )
            now = _now()
            db.execute(
                f"""INSERT INTO signal_outcomes
                    (evaluation_id,strategy,symbol,candle_time,side,trigger_confirmed_at,
                     entry_price,stop_loss,tp1_price,tp2_price,atr,horizon_bars,fee_pct,
                     slippage_pct,status,outcome_json,created_at,updated_at)
                    VALUES ({','.join([ph] * 18)})
                    ON CONFLICT(evaluation_id) DO NOTHING""",
                (
                    evaluation_id, strategy, symbol, candle_time, resolved_side,
                    resolved_trigger_at, resolved_entry, resolved_stop, resolved_tp1,
                    resolved_tp2, resolved_atr, max(1, min(int(horizon_bars), 288)),
                    max(0.0, float(fee_pct)), max(0.0, float(slippage_pct)),
                    "pending", "{}", now, now,
                ),
            )

    def _recover_outcome_context_from_outbox(self, limit: int = 500) -> None:
        """Recover confirmed outcome context when a crash interrupted outbox journaling."""
        ph = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT o.status,o.payload_json,o.sent_at
                    FROM alert_outbox o
                    JOIN signal_journal j
                      ON o.signal_key = j.symbol || ':' || j.strategy || ':' || j.side || ':' || j.candle_time
                    LEFT JOIN signal_outcomes so ON so.evaluation_id=j.evaluation_id
                    WHERE (o.status='sent' AND
                           (j.lifecycle != 'telegram_sent' OR so.evaluation_id IS NULL))
                       OR (o.status!='sent' AND
                           (j.lifecycle NOT IN ('trigger_confirmed','telegram_sent')
                            OR so.evaluation_id IS NULL))
                    ORDER BY o.created_at ASC LIMIT {ph}""",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        for outbox_status, payload_json, sent_at in rows:
            try:
                alert = json.loads(payload_json or "{}")
                strategy = str(alert.get("strategy") or "")
                symbol = str(alert.get("symbol") or "")
                setup_time = str(alert.get("setup_time") or "")
                side = str(alert.get("side") or "").upper()
                entry = float(alert.get("entry") or 0)
                stop = float(alert.get("stop_loss") or 0)
                tp1 = float(alert.get("take_profit_1") or 0)
                if not all((strategy, symbol, setup_time)) or side not in {"LONG", "SHORT"}:
                    continue
                if not all(math.isfinite(value) and value > 0 for value in (entry, stop, tp1)):
                    continue
                trigger_at = alert.get("trigger_confirmed_at") or setup_time
                is_sent = outbox_status == "sent"
                event_stamp = (sent_at or _now()) if is_sent else trigger_at
                try:
                    delay = max(
                        0.0,
                        (_utc_datetime(event_stamp) - _utc_datetime(trigger_at)).total_seconds(),
                    ) if is_sent else None
                except Exception:
                    delay = None
                lifecycle = "telegram_sent" if is_sent else "trigger_confirmed"
                self.update_lifecycle(
                    strategy, symbol, setup_time, lifecycle, entry,
                    event_at=event_stamp, trigger_confirmed_at=trigger_at,
                    side=side, stop_loss=stop, target_price=tp1,
                    target_price_2=alert.get("take_profit_2"),
                    atr=alert.get("atr"),
                    horizon_bars=int(alert.get("outcome_horizon_bars") or 12),
                    fee_pct=float(alert.get("outcome_fee_pct") or 0.0),
                    slippage_pct=float(alert.get("outcome_slippage_pct") or 0.0),
                    tp2_r_multiple=float(alert.get("outcome_tp2_r_multiple") or 2.0),
                    trigger_to_telegram_seconds=delay,
                )
            except (TypeError, ValueError, OverflowError):
                continue

    def market_outcome_counts(self) -> dict:
        """Aggregate market-outcome lifecycle counts without mixing human feedback."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT status,COUNT(*) FROM signal_outcomes GROUP BY status"
            ).fetchall()
        by_status = {str(status): int(count) for status, count in rows}
        return {"total": sum(by_status.values()), "by_status": by_status}

    def pending_market_outcomes(self, limit: int = 50):
        """Return durable confirmed-signal outcomes still pending or retryable."""
        self._recover_outcome_context_from_outbox()
        ph = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT o.evaluation_id,o.strategy,o.symbol,o.candle_time,o.side,
                           o.entry_price,o.stop_loss,o.tp1_price,o.tp2_price,o.atr,
                           o.horizon_bars,o.fee_pct,o.slippage_pct,o.trigger_confirmed_at,
                           j.signal_json
                    FROM signal_outcomes o
                    JOIN signal_journal j ON j.evaluation_id=o.evaluation_id
                    WHERE j.lifecycle IN ('trigger_confirmed','telegram_sent') AND o.status != 'complete'
                    ORDER BY o.trigger_confirmed_at ASC LIMIT {ph}""",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        keys = [
            "evaluation_id","strategy","symbol","candle_time","side","entry_price",
            "stop_loss","tp1_price","tp2_price","atr","horizon_bars","fee_pct",
            "slippage_pct","trigger_confirmed_at","signal_json",
        ]
        result = []
        for row in rows:
            item = dict(zip(keys, row))
            try:
                item["signal"] = json.loads(item.pop("signal_json") or "{}")
            except (TypeError, ValueError):
                item["signal"] = {}
                item.pop("signal_json", None)
            result.append(item)
        return result

    def list_market_outcomes(self, limit: int = 100, status: str | None = None):
        """Read normalized persisted outcomes for protected operator verification."""
        if status and status not in {"pending", "complete", "error"}:
            raise ValueError(f"Unknown market outcome status: {status}")
        ph = "?" if self.backend == "sqlite" else "%s"
        where = f" WHERE status={ph}" if status else ""
        params = [status] if status else []
        params.append(max(1, min(int(limit), 500)))
        columns = [
            "evaluation_id","strategy","symbol","candle_time","side",
            "trigger_confirmed_at","entry_price","stop_loss","tp1_price","tp2_price",
            "atr","horizon_bars","fee_pct","slippage_pct","status","first_hit",
            "first_hit_at","final_result","tp1_hit","tp2_hit","stop_hit",
            "same_candle_ambiguous","bars_to_first_hit","bars_to_resolution",
            "bars_observed","elapsed_seconds","mfe_r","mae_r","mfe_pct","mae_pct",
            "mfe_atr","mae_atr","exit_price","gross_return_pct","cost_pct",
            "net_return_pct","resolution_at","outcome_json","created_at","updated_at",
        ]
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {','.join(columns)} FROM signal_outcomes{where} "
                f"ORDER BY trigger_confirmed_at DESC LIMIT {ph}",
                tuple(params),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            try:
                item["outcome"] = json.loads(item.pop("outcome_json") or "{}")
            except (TypeError, ValueError):
                item["outcome"] = {}
                item.pop("outcome_json", None)
            for field in ("tp1_hit","tp2_hit","stop_hit","same_candle_ambiguous"):
                item[field] = bool(item[field])
            result.append(item)
        return result

    def save_market_outcome(self, evaluation_id: str, status: str, outcome: dict) -> bool:
        """Persist normalized outcome fields and JSON; complete rows are immutable."""
        if status not in {"pending", "complete", "error"}:
            raise ValueError(f"Unknown market outcome status: {status}")
        outcome = dict(outcome or {})
        final_result = outcome.get("final_result") or outcome.get("result")
        if status == "complete" and final_result not in {"win", "loss", "ambiguous", "expired"}:
            raise ValueError("completed market outcome requires a final result")
        if final_result is not None:
            outcome["final_result"] = final_result
        ph = "?" if self.backend == "sqlite" else "%s"
        fields = (
            "first_hit", "first_hit_at", "final_result", "tp1_hit", "tp2_hit",
            "stop_hit", "same_candle_ambiguous", "bars_to_first_hit",
            "bars_to_resolution", "bars_observed", "elapsed_seconds", "mfe_r",
            "mae_r", "mfe_pct", "mae_pct", "mfe_atr", "mae_atr", "exit_price",
            "gross_return_pct", "cost_pct", "net_return_pct", "resolution_at",
        )
        assignments = ["status={}".format(ph)]
        params = [status]
        for field in fields:
            assignments.append(f"{field}={ph}")
            value = outcome.get(field)
            if field in {"tp1_hit", "tp2_hit", "stop_hit", "same_candle_ambiguous"}:
                value = int(bool(value))
            elif field == "bars_observed" and value is None:
                value = 0
            params.append(value)
        assignments.extend([f"outcome_json={ph}", f"updated_at={ph}"])
        params.extend([
            json.dumps(outcome or {}, separators=(",", ":"), default=str),
            _now(), evaluation_id,
        ])
        with self._connect() as db:
            updated = db.execute(
                f"UPDATE signal_outcomes SET {', '.join(assignments)} "
                f"WHERE evaluation_id={ph} AND status != 'complete'",
                tuple(params),
            )
            if updated.rowcount != 1:
                return False
            db.execute(
                f"""UPDATE signal_journal
                    SET market_outcome_status={ph},market_outcome_json={ph},
                        market_outcome_checked_at={ph}
                    WHERE evaluation_id={ph} AND market_outcome_status != 'complete'""",
                (
                    status, json.dumps(outcome or {}, separators=(",", ":"), default=str),
                    _now(), evaluation_id,
                ),
            )
            return True

    def add_feedback(self, evaluation_id: str, feedback: str, note: str = "",
                     outcome: dict | None = None, source: str = "admin_api",
                     user_id: str | None = None) -> bool:
        allowed = {"good", "bad", "late", "missed", "needs_review"}
        label = str(feedback or "").strip().lower()
        if label not in allowed:
            raise ValueError(f"Unknown feedback label: {label}")
        ph = "?" if self.backend == "sqlite" else "%s"
        sql = f"""UPDATE signal_journal SET feedback={ph}, feedback_note={ph},
                  reviewed_at={ph}, outcome_json={ph}, feedback_source={ph},
                  feedback_user_id={ph}
                  WHERE evaluation_id={ph} AND feedback IS NULL"""
        with self._connect() as db:
            cur = db.execute(sql, (
                label, str(note or "")[:2000], _now(),
                json.dumps(outcome or {}, separators=(",", ":"), default=str),
                str(source or "unknown")[:40],
                str(user_id)[:128] if user_id is not None else None,
                evaluation_id,
            ))
            return cur.rowcount > 0

    def list_ml_source_records(
        self, strategy: str | None = None, evaluation_id: str | None = None,
        limit: int = 500,
    ):
        """Return only immutable ORB/MEX signal snapshots and point-in-time trigger telemetry."""
        if strategy and strategy not in {"ORB", "MEX"}:
            raise ValueError("ML source strategy must be ORB or MEX")
        ph = "?" if self.backend == "sqlite" else "%s"
        clauses = [
            "decision='setup'",
            "lifecycle IN ('trigger_confirmed','telegram_sent')",
            "strategy IN ('ORB','MEX')",
        ]
        params = []
        if strategy:
            clauses.append(f"strategy={ph}")
            params.append(strategy)
        if evaluation_id:
            clauses.append(f"evaluation_id={ph}")
            params.append(evaluation_id)
        params.append(max(1, min(int(limit), 5000)))
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT evaluation_id,strategy,symbol,candle_time,evaluated_at,
                           decision,side,diagnostics_json,signal_json,lifecycle,
                           trigger_confirmed_at,setup_to_trigger_seconds,
                           entry_extension_atr,trigger_overshoot_atr,trigger_price,entry_price
                    FROM signal_journal WHERE {' AND '.join(clauses)}
                    ORDER BY COALESCE(trigger_confirmed_at,candle_time) ASC
                    LIMIT {ph}""",
                tuple(params),
            ).fetchall()
        keys = [
            "evaluation_id","strategy","symbol","candle_time","evaluated_at",
            "decision","side","diagnostics_json","signal_json","lifecycle",
            "trigger_confirmed_at","setup_to_trigger_seconds","entry_extension_atr",
            "trigger_overshoot_atr","trigger_price","entry_price",
        ]
        result = []
        for row in rows:
            item = dict(zip(keys, row))
            for source, target in (("diagnostics_json","diagnostics"),("signal_json","signal")):
                try:
                    item[target] = json.loads(item.pop(source) or "{}")
                except (TypeError, ValueError):
                    item[target] = {}
                    item.pop(source, None)
            item["feature_at"] = item.get("trigger_confirmed_at") or item["candle_time"]
            result.append(item)
        return result

    def get_ml_source_record(self, evaluation_id: str):
        rows = self.list_ml_source_records(evaluation_id=evaluation_id, limit=1)
        return rows[0] if rows else None

    def list_records(self, limit: int = 100, offset: int = 0,
                     strategy: str | None = None, decision: str | None = None,
                     feedback: str | None = None, lifecycle: str | None = None):
        clauses, params = [], []
        for field, value in (("strategy", strategy), ("decision", decision), ("feedback", feedback), ("lifecycle", lifecycle)):
            if value:
                clauses.append(f"{field} = {'?' if self.backend == 'sqlite' else '%s'}")
                params.append(value)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        ph = "?" if self.backend == "sqlite" else "%s"
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._connect() as db:
            rows = db.execute(
                f"SELECT evaluation_id,strategy,symbol,candle_time,evaluated_at,decision,side,reason,"
                f"diagnostics_json,signal_json,lifecycle,entry_price,stop_loss,target_price,feedback,"
                f"feedback_note,reviewed_at,outcome_json,setup_armed_at,trigger_confirmed_at,"
                f"telegram_sent_at,trigger_price,setup_to_trigger_seconds,entry_extension_atr,"
                f"trigger_overshoot_atr,trigger_to_telegram_seconds,market_outcome_status,"
                f"market_outcome_json,market_outcome_checked_at,feedback_source,feedback_user_id FROM signal_journal{where} "
                f"ORDER BY evaluated_at DESC LIMIT {ph} OFFSET {ph}", params
            ).fetchall()
        keys = ["evaluation_id","strategy","symbol","candle_time","evaluated_at","decision",
                "side","reason","diagnostics","signal","lifecycle","entry_price","stop_loss",
                "target_price","feedback","feedback_note","reviewed_at","outcome",
                "setup_armed_at","trigger_confirmed_at","telegram_sent_at","trigger_price",
                "setup_to_trigger_seconds","entry_extension_atr","trigger_overshoot_atr",
                "trigger_to_telegram_seconds","market_outcome_status","market_outcome","market_outcome_checked_at",
                "feedback_source","feedback_user_id"]
        result = []
        for row in rows:
            item = dict(zip(keys, row))
            for key in ("diagnostics", "signal", "outcome", "market_outcome"):
                try:
                    item[key] = json.loads(item[key] or "{}")
                except (TypeError, ValueError):
                    item[key] = {}
            result.append(item)
        return result

    def export_csv(self, **filters) -> str:
        records = self.list_records(limit=500, **filters)
        output = io.StringIO()
        fields = ["evaluation_id","strategy","symbol","candle_time","evaluated_at","decision",
                  "side","reason","lifecycle","entry_price","stop_loss","target_price",
                  "feedback","feedback_note","diagnostics","signal","outcome",
                  "setup_armed_at","trigger_confirmed_at","telegram_sent_at","trigger_price",
                  "setup_to_trigger_seconds","entry_extension_atr","trigger_overshoot_atr",
                  "trigger_to_telegram_seconds","market_outcome_status","market_outcome","market_outcome_checked_at",
                  "feedback_source","feedback_user_id"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(row[key], separators=(",", ":"))
                             if isinstance(row.get(key), (dict, list)) else row.get(key)
                             for key in fields})
        return output.getvalue()

    def counts(self) -> dict:
        with self._connect() as db:
            total = db.execute("SELECT COUNT(*) FROM signal_journal").fetchone()[0]
            reviewed = db.execute("SELECT COUNT(*) FROM signal_journal WHERE feedback IS NOT NULL").fetchone()[0]
            rejected = db.execute("SELECT COUNT(*) FROM signal_journal WHERE decision='rejected'").fetchone()[0]
            setups = db.execute("SELECT COUNT(*) FROM signal_journal WHERE decision='setup'").fetchone()[0]
        return {"records": int(total), "reviewed": int(reviewed),
                "rejected_evaluations": int(rejected), "setup_evaluations": int(setups)}

    def latest_record(self, decision: str | None = None, feedback: str | None = None, lifecycle: str | None = None):
        rows = self.list_records(limit=1, decision=decision, feedback=feedback, lifecycle=lifecycle)
        return rows[0] if rows else None

    def undo_feedback(self, evaluation_id: str) -> bool:
        ph = "?" if self.backend == "sqlite" else "%s"
        sql = f"""UPDATE signal_journal SET feedback=NULL, feedback_note=NULL,
                  reviewed_at=NULL, outcome_json='{{}}', feedback_source=NULL,
                  feedback_user_id=NULL
                  WHERE evaluation_id={ph} AND feedback IS NOT NULL"""
        with self._connect() as db:
            cur = db.execute(sql, (evaluation_id,))
            return cur.rowcount > 0

    def add_training_note(self, note: str, source: str = "telegram") -> int:
        clean = str(note or "").strip()[:2000]
        if not clean:
            raise ValueError("training note cannot be empty")
        ph = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            if self.backend == "sqlite":
                cur = db.execute(
                    f"INSERT INTO training_notes (created_at,note,source) VALUES ({ph},{ph},{ph})",
                    (_now(), clean, source[:40]),
                )
                return int(cur.lastrowid)
            cur = db.execute(
                f"INSERT INTO training_notes (created_at,note,source) VALUES ({ph},{ph},{ph}) RETURNING note_id",
                (_now(), clean, source[:40]),
            )
            return int(cur.fetchone()[0])

    def list_training_notes(self, limit: int = 10):
        ph = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            rows = db.execute(
                f"SELECT created_at,note,source FROM training_notes ORDER BY created_at DESC LIMIT {ph}",
                (max(1, min(int(limit), 50)),),
            ).fetchall()
        return [{"created_at": r[0], "note": r[1], "source": r[2]} for r in rows]

    def learning_report(self) -> dict:
        with self._connect() as db:
            total = db.execute("SELECT COUNT(*) FROM signal_journal").fetchone()[0]
            reviewed = db.execute("SELECT COUNT(*) FROM signal_journal WHERE feedback IS NOT NULL").fetchone()[0]
            feedback_rows = db.execute(
                "SELECT feedback, COUNT(*) FROM signal_journal WHERE feedback IS NOT NULL GROUP BY feedback"
            ).fetchall()
            strategy_rows = db.execute(
                """SELECT strategy, COUNT(*),
                   SUM(CASE WHEN feedback IS NOT NULL THEN 1 ELSE 0 END)
                   FROM signal_journal GROUP BY strategy ORDER BY strategy"""
            ).fetchall()
            note_count = db.execute("SELECT COUNT(*) FROM training_notes").fetchone()[0]
        labels = {str(label): int(count) for label, count in feedback_rows}
        reviewed_count = int(reviewed or 0)
        return {
            "total_evaluations": int(total or 0),
            "human_reviewed": reviewed_count,
            "unreviewed": max(0, int(total or 0) - reviewed_count),
            "feedback_counts": labels,
            "training_notes": int(note_count or 0),
            "by_strategy": [
                {"strategy": r[0], "evaluations": int(r[1] or 0), "reviewed": int(r[2] or 0)}
                for r in strategy_rows
            ],
            "research_status": "early_sample" if reviewed_count < 30 else "ready_for_manual_analysis",
            "minimum_reviewed_samples_for_research": 30,
            "live_strategy_modified": False,
            "automatic_learning_enabled": False,
        }

    def create_feedback_token(self, evaluation_id: str) -> str:
        """Create a high-entropy, single-use token for one Telegram review action."""
        token = secrets.token_urlsafe(12)
        ph = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            db.execute(
                f"INSERT INTO signal_feedback_tokens (token,evaluation_id,created_at) VALUES ({ph},{ph},{ph})",
                (token, evaluation_id, _now()),
            )
        return token

    def save_feedback_from_token(
        self, token: str, feedback: str, note: str = "",
        source: str = "telegram_inline", user_id: str | None = None,
    ) -> str | None:
        """Atomically save the first review and consume its token.

        A replay, a token for an already-reviewed evaluation, or an unknown
        token returns None. The token and feedback update commit together.
        """
        allowed = {"good", "bad", "late", "missed", "needs_review"}
        label = str(feedback or "").strip().lower()
        if label not in allowed:
            raise ValueError(f"Unknown feedback label: {label}")
        if not token or len(token) > 128:
            return None
        ph = "?" if self.backend == "sqlite" else "%s"
        now = _now()
        with self._connect() as db:
            lock_clause = " FOR UPDATE" if self.backend == "postgresql" else ""
            row = db.execute(
                f"SELECT evaluation_id FROM signal_feedback_tokens "
                f"WHERE token={ph} AND consumed_at IS NULL{lock_clause}",
                (token,),
            ).fetchone()
            if not row:
                return None
            evaluation_id = row[0]
            cur = db.execute(
                f"""UPDATE signal_journal SET feedback={ph}, feedback_note={ph},
                    reviewed_at={ph}, outcome_json='{{}}', feedback_source={ph},
                    feedback_user_id={ph}
                    WHERE evaluation_id={ph} AND feedback IS NULL""",
                (label, str(note or "")[:2000], now, str(source or "unknown")[:40],
                 str(user_id)[:128] if user_id is not None else None, evaluation_id),
            )
            if cur.rowcount != 1:
                # Invalidate this token too if another review already won.
                db.execute(
                    f"UPDATE signal_feedback_tokens SET consumed_at={ph} "
                    f"WHERE token={ph} AND consumed_at IS NULL",
                    (now, token),
                )
                return None
            consumed = db.execute(
                f"UPDATE signal_feedback_tokens SET consumed_at={ph} "
                f"WHERE token={ph} AND consumed_at IS NULL",
                (now, token),
            )
            if consumed.rowcount != 1:
                db.rollback()
                return None
            return evaluation_id

    def resolve_feedback_token(self, token: str):
        """Legacy read-only token lookup; new webhook handlers must consume tokens."""
        ph = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            row = db.execute(
                f"SELECT evaluation_id FROM signal_feedback_tokens WHERE token={ph}",
                (token,),
            ).fetchone()
        return row[0] if row else None
