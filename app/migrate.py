"""Explicit schema migration command.

Run with `python -m app.migrate` before starting a production deployment.
"""
import json
import sqlite3
from pathlib import Path

from .config import settings
from .migrations import MIGRATIONS, apply_migrations


def main() -> None:
    settings.validate_safety()
    database_url = (settings.database_url or "").strip()
    if database_url:
        import psycopg
        from .database import postgres_conninfo

        connect = lambda: psycopg.connect(postgres_conninfo(database_url), connect_timeout=5)
        backend = "postgresql"
    else:
        path = Path(settings.state_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connect = lambda: sqlite3.connect(path, timeout=5)
        backend = "sqlite"

    applied = apply_migrations(connect, backend)
    print(json.dumps({
        "backend": backend,
        "applied_versions": applied,
        "latest_version": max(version for version, _name, _migration in MIGRATIONS),
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
