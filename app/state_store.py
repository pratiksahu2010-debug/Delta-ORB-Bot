"""Durable signal deduplication and Telegram outbox store.

PostgreSQL is selected when DATABASE_URL is configured (Render production).
SQLite remains available for local development and tests.
"""
import json
import secrets
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SignalStore:
    def __init__(self, path: str, database_url: str | None = None, migrate_on_startup: bool = True):
        self.database_url = (database_url or "").strip()
        self.backend = "postgresql" if self.database_url else "sqlite"
        self.migrate_on_startup = bool(migrate_on_startup)
        if self.backend == "postgresql":
            self._init_postgres()
            self.path = None
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()

    def _connect_postgres(self):
        import psycopg
        from .database import postgres_conninfo
        return psycopg.connect(postgres_conninfo(self.database_url), connect_timeout=5)

    def _init_postgres(self) -> None:
        from .migrations import apply_migrations, verify_migrations
        migrate = apply_migrations if self.migrate_on_startup else verify_migrations
        migrate(self._connect_postgres, "postgresql")

    def _init_sqlite(self) -> None:
        from .migrations import apply_migrations, verify_migrations
        migrate = apply_migrations if self.migrate_on_startup else verify_migrations
        migrate(lambda: sqlite3.connect(self.path, timeout=5), "sqlite")

    def _connect(self):
        if self.backend == "postgresql":
            return self._connect_postgres()
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def is_sent(self, signal_key: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM sent_signals WHERE signal_key = ? LIMIT 1"
                if self.backend == "sqlite"
                else "SELECT 1 FROM sent_signals WHERE signal_key = %s LIMIT 1",
                (signal_key,),
            ).fetchone()
        return row is not None

    def enqueue_alert(self, signal_key: str, payload: dict) -> bool:
        """Persist the alert before network delivery; duplicate keys are idempotent."""
        now = _utc_now().isoformat()
        sql = (
            """INSERT OR IGNORE INTO alert_outbox
               (signal_key, payload_json, status, attempts, created_at, next_attempt_at, updated_at)
               VALUES (?, ?, 'pending', 0, ?, ?, ?)"""
            if self.backend == "sqlite"
            else """INSERT INTO alert_outbox
               (signal_key, payload_json, status, attempts, created_at, next_attempt_at, updated_at)
               VALUES (%s, %s, 'pending', 0, %s, %s, %s) ON CONFLICT (signal_key) DO NOTHING"""
        )
        with self._connect() as db:
            cur = db.execute(sql, (signal_key, json.dumps(payload, separators=(",", ":")), now, now, now))
            return cur.rowcount == 1

    def enqueue_advika_signal(
        self, signal_key: str, alert: dict, position: dict,
    ) -> tuple[bool, bool]:
        """Atomically persist an Advika entry alert and its paper position."""
        now = _utc_now().isoformat()
        setup_time = str(position["setup_time"])
        encoded_alert = json.dumps(alert, separators=(",", ":"))
        encoded_position = json.dumps(position, separators=(",", ":"))
        if self.backend == "sqlite":
            outbox_sql = """INSERT OR IGNORE INTO alert_outbox
                (signal_key,payload_json,status,attempts,created_at,next_attempt_at,updated_at)
                VALUES (?,?,'pending',0,?,?,?)"""
            position_sql = """INSERT OR IGNORE INTO advika_positions
                (signal_key,symbol,status,position_json,last_processed_candle,created_at,updated_at)
                VALUES (?,?,'pending',?,?,?,?)"""
        else:
            outbox_sql = """INSERT INTO alert_outbox
                (signal_key,payload_json,status,attempts,created_at,next_attempt_at,updated_at)
                VALUES (%s,%s,'pending',0,%s,%s,%s) ON CONFLICT(signal_key) DO NOTHING"""
            position_sql = """INSERT INTO advika_positions
                (signal_key,symbol,status,position_json,last_processed_candle,created_at,updated_at)
                VALUES (%s,%s,'pending',%s,%s,%s,%s) ON CONFLICT(signal_key) DO NOTHING"""
        with self._connect() as db:
            outbox_cur = db.execute(
                outbox_sql, (signal_key, encoded_alert, now, now, now),
            )
            position_cur = db.execute(
                position_sql, (
                    signal_key, str(position["symbol"]), encoded_position,
                    setup_time, now, now,
                ),
            )
            return outbox_cur.rowcount == 1, position_cur.rowcount == 1

    def due_alerts(self, limit: int = 50) -> list[tuple[str, dict]]:
        now = _utc_now().isoformat()
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT signal_key, payload_json FROM alert_outbox
                    WHERE status = 'pending' AND next_attempt_at <= {placeholder}
                    ORDER BY created_at ASC LIMIT {placeholder}""",
                (now, max(1, int(limit))),
            ).fetchall()
        return [(key, json.loads(payload)) for key, payload in rows]

    def claim_due_alerts(self, limit: int = 50, lease_seconds: int = 60) -> list[tuple[str, dict, str]]:
        """Atomically lease due alerts and return a fencing token for each claim."""
        now = _utc_now()
        now_text = now.isoformat()
        lease_text = (now + timedelta(seconds=max(10, int(lease_seconds)))).isoformat()
        batch_limit = max(1, min(int(limit), 200))
        claimed = []
        with self._connect() as db:
            if self.backend == "sqlite":
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """UPDATE alert_outbox SET status = 'pending', lease_until = NULL,
                       lease_token = NULL, next_attempt_at = ?, updated_at = ?
                       WHERE status = 'sending' AND (lease_until IS NULL OR lease_until <= ?)""",
                    (now_text, now_text, now_text),
                )
                rows = db.execute(
                    """SELECT signal_key, payload_json FROM alert_outbox
                       WHERE status = 'pending' AND next_attempt_at <= ?
                       ORDER BY created_at ASC LIMIT ?""",
                    (now_text, batch_limit),
                ).fetchall()
            else:
                db.execute(
                    """UPDATE alert_outbox SET status = 'pending', lease_until = NULL,
                       lease_token = NULL, next_attempt_at = %s, updated_at = %s
                       WHERE status = 'sending' AND (lease_until IS NULL OR lease_until <= %s)""",
                    (now_text, now_text, now_text),
                )
                rows = db.execute(
                    """SELECT signal_key, payload_json FROM alert_outbox
                       WHERE status = 'pending' AND next_attempt_at <= %s
                       ORDER BY created_at ASC LIMIT %s FOR UPDATE SKIP LOCKED""",
                    (now_text, batch_limit),
                ).fetchall()

            for key, payload_json in rows:
                lease_token = secrets.token_urlsafe(12)
                placeholder = "?" if self.backend == "sqlite" else "%s"
                updated = db.execute(
                    f"""UPDATE alert_outbox SET status = 'sending', lease_until = {placeholder},
                        lease_token = {placeholder}, attempts = attempts + 1, updated_at = {placeholder}
                        WHERE signal_key = {placeholder} AND status = 'pending'""",
                    (lease_text, lease_token, now_text, key),
                )
                if updated.rowcount == 1:
                    claimed.append((key, json.loads(payload_json), lease_token))
        return claimed

    def note_attempt(self, signal_key: str) -> None:
        now = _utc_now()
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            db.execute(
                f"""UPDATE alert_outbox SET attempts = attempts + 1,
                    next_attempt_at = {placeholder}, updated_at = {placeholder}
                    WHERE signal_key = {placeholder} AND status = 'pending'""",
                ((now + timedelta(seconds=10)).isoformat(), now.isoformat(), signal_key),
            )

    def schedule_retry(self, signal_key: str, lease_token: str, delay_seconds: float, error: str) -> bool:
        """Release only the currently owned lease; stale workers cannot reschedule a new claim."""
        now = _utc_now()
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            cur = db.execute(
                f"""UPDATE alert_outbox SET status = 'pending', next_attempt_at = {placeholder},
                    updated_at = {placeholder}, last_error = {placeholder},
                    lease_until = NULL, lease_token = NULL
                    WHERE signal_key = {placeholder} AND status = 'sending' AND lease_token = {placeholder}""",
                ((now + timedelta(seconds=max(1.0, delay_seconds))).isoformat(),
                 now.isoformat(), str(error)[:1000], signal_key, lease_token),
            )
            return cur.rowcount == 1

    def mark_sent(self, signal_key: str, payload: dict, lease_token: str) -> bool:
        """Commit sent dedupe and outbox completion only for the active lease owner."""
        now = _utc_now().isoformat()
        placeholder = "?" if self.backend == "sqlite" else "%s"
        insert_sql = (
            """INSERT OR IGNORE INTO sent_signals
               (signal_key, symbol, side, setup_time, payload_json, sent_at)
               VALUES (?, ?, ?, ?, ?, ?)"""
            if self.backend == "sqlite"
            else """INSERT INTO sent_signals
               (signal_key, symbol, side, setup_time, payload_json, sent_at)
               VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (signal_key) DO NOTHING"""
        )
        with self._connect() as db:
            updated = db.execute(
                f"""UPDATE alert_outbox SET status = 'sent', sent_at = {placeholder},
                    updated_at = {placeholder}, last_error = NULL,
                    lease_until = NULL, lease_token = NULL
                    WHERE signal_key = {placeholder} AND status = 'sending' AND lease_token = {placeholder}""",
                (now, now, signal_key, lease_token),
            )
            if updated.rowcount != 1:
                return False
            db.execute(
                insert_sql,
                (signal_key, payload["symbol"], payload["side"], payload["setup_time"],
                 json.dumps(payload, separators=(",", ":")), now),
            )
            return True

    def outbox_health(self) -> dict:
        """Return read-only delivery and recovery diagnostics without exposing payloads/errors."""
        now = _utc_now()
        now_text = now.isoformat()
        placeholder = "?" if self.backend == "sqlite" else "%s"
        sql = f"""SELECT
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'sending' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'pending' AND last_error IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'pending' AND next_attempt_at <= {placeholder} THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'pending' AND next_attempt_at > {placeholder} THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'sending' AND (lease_until IS NULL OR lease_until <= {placeholder}) THEN 1 ELSE 0 END),
            MIN(CASE WHEN status = 'pending' THEN created_at END),
            MIN(CASE WHEN status = 'sending' THEN updated_at END),
            MIN(CASE WHEN status = 'pending' AND last_error IS NOT NULL THEN updated_at END),
            MAX(CASE WHEN status = 'pending' AND last_error IS NOT NULL THEN updated_at END),
            MAX(CASE WHEN status = 'pending' AND last_error IS NOT NULL THEN attempts ELSE 0 END),
            MAX(CASE WHEN status IN ('pending', 'sending') THEN attempts ELSE 0 END)
            FROM alert_outbox"""
        with self._connect() as db:
            row = db.execute(sql, (now_text, now_text, now_text)).fetchone()

        def age_seconds(value):
            if not value:
                return None
            try:
                stamp = datetime.fromisoformat(str(value))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                return max(0, int((now - stamp.astimezone(timezone.utc)).total_seconds()))
            except (TypeError, ValueError, OverflowError):
                return None

        last_failure_at = row[10]
        expired_leases = int(row[6] or 0)
        return {
            "pending": int(row[0] or 0),
            "sending": int(row[1] or 0),
            "sent": int(row[2] or 0),
            "retrying": int(row[3] or 0),
            "due_pending": int(row[4] or 0),
            "scheduled_retries": int(row[5] or 0),
            "expired_leases": expired_leases,
            "recovery_required": expired_leases > 0,
            "oldest_pending_age_seconds": age_seconds(row[7]),
            "oldest_sending_age_seconds": age_seconds(row[8]),
            "oldest_retry_age_seconds": age_seconds(row[9]),
            "last_failure_at": last_failure_at,
            "last_failure_age_seconds": age_seconds(last_failure_at),
            "max_retry_attempts": int(row[11] or 0),
            "max_active_attempts": int(row[12] or 0),
        }

    def delivery_record(self, signal_key: str) -> dict | None:
        """Return persisted delivery metadata without exposing the signal payload."""
        placeholder = "?" if self.backend == "sqlite" else "%s"
        sql = f"""SELECT o.signal_key,o.status,o.attempts,o.created_at,o.updated_at,
                         o.next_attempt_at,o.sent_at,o.last_error,s.sent_at
                  FROM alert_outbox o
                  LEFT JOIN sent_signals s ON s.signal_key=o.signal_key
                  WHERE o.signal_key={placeholder}"""
        with self._connect() as db:
            row = db.execute(sql, (signal_key,)).fetchone()
        if row is None:
            return None
        return {
            "signal_key": row[0],
            "status": row[1],
            "attempts": int(row[2] or 0),
            "created_at": row[3],
            "updated_at": row[4],
            "next_attempt_at": row[5],
            "outbox_sent_at": row[6],
            "last_error": str(row[7])[:180] if row[7] else None,
            "sent_signal_recorded": row[8] is not None,
            "sent_signal_at": row[8],
        }

    def list_open_advika_positions(self, symbol: str) -> list[dict]:
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT signal_key,status,position_json,last_processed_candle
                    FROM advika_positions
                    WHERE symbol={placeholder} AND status IN ('open','partial')
                    ORDER BY created_at ASC""",
                (symbol,),
            ).fetchall()
        positions = []
        for signal_key, status, payload, last_candle in rows:
            try:
                position = json.loads(payload or "{}")
            except (TypeError, ValueError):
                continue
            position.update({
                "signal_key": signal_key, "status": status,
                "last_processed_candle": last_candle,
            })
            positions.append(position)
        return positions

    def update_advika_position(
        self, signal_key: str, position: dict, status: str, last_processed_candle: str,
    ) -> bool:
        """Advance one open paper position; terminal positions cannot be reopened."""
        clean_status = str(status).lower()
        if clean_status not in {"open", "partial", "closed"}:
            raise ValueError("invalid_advika_position_status")
        now = _utc_now().isoformat()
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            cur = db.execute(
                f"""UPDATE advika_positions
                    SET status={placeholder}, position_json={placeholder},
                        last_processed_candle={placeholder}, updated_at={placeholder}
                    WHERE signal_key={placeholder} AND status IN ('open','partial')""",
                (clean_status, json.dumps(position, separators=(",", ":")),
                 last_processed_candle, now, signal_key),
            )
            return cur.rowcount == 1

    def activate_delivered_advika_positions(self) -> int:
        """Activate paper positions only after Telegram accepted their entry alert."""
        with self._connect() as db:
            cur = db.execute(
                """UPDATE advika_positions SET status='open',updated_at=?
                   WHERE status='pending'
                   AND signal_key IN (SELECT signal_key FROM sent_signals)""",
                (_utc_now().isoformat(),),
            )
            return int(cur.rowcount or 0)

    def advika_position_health(self) -> dict:
        with self._connect() as db:
            rows = db.execute(
                "SELECT status,COUNT(*) FROM advika_positions GROUP BY status"
            ).fetchall()
        counts = {str(status): int(count or 0) for status, count in rows}
        return {
            "pending": counts.get("pending", 0),
            "open": counts.get("open", 0),
            "partial": counts.get("partial", 0),
            "closed": counts.get("closed", 0),
            "active": counts.get("open", 0) + counts.get("partial", 0),
        }

    def save_pending_setup(self, symbol: str, strategy: str, payload: dict) -> None:
        """Persist an armed setup; live quote-confirmation state is intentionally excluded."""
        now = _utc_now().isoformat()
        expires_at = str(payload["expires_at"])
        encoded = json.dumps(payload, separators=(",", ":"))
        if self.backend == "sqlite":
            sql = """INSERT INTO pending_setups (symbol, strategy, payload_json, expires_at, updated_at)
                     VALUES (?, ?, ?, ?, ?)
                     ON CONFLICT(symbol, strategy) DO UPDATE SET
                       payload_json = excluded.payload_json,
                       expires_at = excluded.expires_at,
                       updated_at = excluded.updated_at"""
        else:
            sql = """INSERT INTO pending_setups (symbol, strategy, payload_json, expires_at, updated_at)
                     VALUES (%s, %s, %s, %s, %s)
                     ON CONFLICT(symbol, strategy) DO UPDATE SET
                       payload_json = EXCLUDED.payload_json,
                       expires_at = EXCLUDED.expires_at,
                       updated_at = EXCLUDED.updated_at"""
        with self._connect() as db:
            db.execute(sql, (symbol, strategy, encoded, expires_at, now))

    def load_pending_setups(self) -> list[tuple[str, str, dict]]:
        """Load only unexpired setup intents and prune expired rows."""
        now = _utc_now().isoformat()
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            db.execute(f"DELETE FROM pending_setups WHERE expires_at <= {placeholder}", (now,))
            rows = db.execute(
                f"""SELECT symbol, strategy, payload_json FROM pending_setups
                    WHERE expires_at > {placeholder} ORDER BY updated_at ASC""",
                (now,),
            ).fetchall()
        return [(symbol, strategy, json.loads(payload)) for symbol, strategy, payload in rows]

    def delete_pending_setup(self, symbol: str, strategy: str) -> None:
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            db.execute(
                f"DELETE FROM pending_setups WHERE symbol = {placeholder} AND strategy = {placeholder}",
                (symbol, strategy),
            )

    def pending_setup_count(self) -> int:
        with self._connect() as db:
            row = db.execute("SELECT COUNT(*) FROM pending_setups").fetchone()
        return int(row[0] or 0)


    @staticmethod
    def _setup_id(symbol: str, strategy: str, candle_time: str) -> str:
        return f"{symbol}:{strategy}:{candle_time}"

    def setup_lifecycle(self, symbol: str, strategy: str, candle_time: str) -> dict | None:
        setup_id = self._setup_id(symbol, strategy, candle_time)
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            row = db.execute(
                f"SELECT state, version, updated_at FROM setup_lifecycle_current WHERE setup_id = {placeholder}",
                (setup_id,),
            ).fetchone()
        if row is None:
            return None
        return {"state": row[0], "version": int(row[1]), "updated_at": row[2]}

    def transition_setup(
        self, symbol: str, strategy: str, candle_time: str,
        from_state: str | None, to_state: str, reason: str,
        event_at: str | None = None,
    ) -> bool:
        """Atomically append a validated lifecycle transition with state fencing."""
        from .setup_lifecycle import is_allowed_transition

        if not is_allowed_transition(from_state, to_state):
            raise ValueError(f"Illegal setup lifecycle transition: {from_state} -> {to_state}")
        setup_id = self._setup_id(symbol, strategy, candle_time)
        now = event_at or _utc_now().isoformat()
        clean_reason = str(reason or "unspecified")[:160]
        with self._connect() as db:
            if self.backend == "sqlite":
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """INSERT OR IGNORE INTO setup_lifecycle_current
                       (setup_id, symbol, strategy, candle_time, state, version, updated_at)
                       VALUES (?, ?, ?, ?, '__NEW__', 0, ?)""",
                    (setup_id, symbol, strategy, candle_time, now),
                )
                row = db.execute(
                    "SELECT state, version FROM setup_lifecycle_current WHERE setup_id = ?",
                    (setup_id,),
                ).fetchone()
            else:
                db.execute(
                    """INSERT INTO setup_lifecycle_current
                       (setup_id, symbol, strategy, candle_time, state, version, updated_at)
                       VALUES (%s, %s, %s, %s, '__NEW__', 0, %s)
                       ON CONFLICT (setup_id) DO NOTHING""",
                    (setup_id, symbol, strategy, candle_time, now),
                )
                row = db.execute(
                    "SELECT state, version FROM setup_lifecycle_current WHERE setup_id = %s FOR UPDATE",
                    (setup_id,),
                ).fetchone()
            if row is None:
                return False
            current_state, version = row[0], int(row[1])
            if current_state == to_state:
                return True
            if current_state == "__NEW__":
                if from_state is not None:
                    db.rollback()
                    return False
                persisted_from = None
            else:
                if current_state != from_state:
                    return False
                persisted_from = current_state
            next_version = version + 1
            placeholder = "?" if self.backend == "sqlite" else "%s"
            updated = db.execute(
                f"""UPDATE setup_lifecycle_current
                    SET state = {placeholder}, version = {placeholder}, updated_at = {placeholder}
                    WHERE setup_id = {placeholder} AND version = {placeholder}""",
                (to_state, next_version, now, setup_id, version),
            )
            if updated.rowcount != 1:
                return False
            db.execute(
                f"""INSERT INTO setup_lifecycle_events
                    (setup_id, sequence, symbol, strategy, candle_time, from_state, to_state, reason, event_at)
                    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                            {placeholder}, {placeholder}, {placeholder}, {placeholder})""",
                (setup_id, next_version, symbol, strategy, candle_time,
                 persisted_from, to_state, clean_reason, now),
            )
            return True


    def setup_lifecycle_health(self) -> dict:
        """Summarize durable setup states without exposing signal payloads."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT state, COUNT(*) FROM setup_lifecycle_current GROUP BY state"
            ).fetchall()
        states = {str(state): int(count) for state, count in rows}
        active = states.get("ARMED", 0) + states.get("CONFIRMING", 0)
        return {
            "total": sum(states.values()),
            "active": active,
            "terminal": sum(states.get(state, 0) for state in ("TELEGRAM_SENT", "EXPIRED", "INVALIDATED")),
            "states": states,
        }

    def setup_lifecycle_events(self, symbol: str, strategy: str, candle_time: str) -> list[dict]:
        setup_id = self._setup_id(symbol, strategy, candle_time)
        placeholder = "?" if self.backend == "sqlite" else "%s"
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT sequence, from_state, to_state, reason, event_at
                    FROM setup_lifecycle_events WHERE setup_id = {placeholder}
                    ORDER BY sequence ASC""",
                (setup_id,),
            ).fetchall()
        return [
            {"sequence": int(row[0]), "from_state": row[1], "to_state": row[2],
             "reason": row[3], "event_at": row[4]}
            for row in rows
        ]

    def health(self) -> bool:
        """Check database connectivity and write capability without persistent side effects."""
        if self.backend == "sqlite":
            with sqlite3.connect(self.path, timeout=5) as db:
                integrity = db.execute("PRAGMA quick_check").fetchone()
                if not integrity or integrity[0] != "ok":
                    return False
                db.execute("BEGIN IMMEDIATE")
                db.execute("CREATE TABLE IF NOT EXISTS _health_probe (id INTEGER PRIMARY KEY, checked_at TEXT NOT NULL)")
                db.execute(
                    "INSERT OR REPLACE INTO _health_probe (id, checked_at) VALUES (1, ?)",
                    (_utc_now().isoformat(),),
                )
                db.rollback()
            return True

        with self._connect_postgres() as db:
            db.execute("CREATE TEMP TABLE _health_probe (id INTEGER PRIMARY KEY, checked_at TEXT NOT NULL) ON COMMIT DROP")
            db.execute("INSERT INTO _health_probe (id, checked_at) VALUES (%s, %s)",
                       (1, _utc_now().isoformat()))
            db.rollback()
        return True
