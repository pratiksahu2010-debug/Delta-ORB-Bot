import sqlite3

import pytest

from app.config import Settings
from app.state_store import SignalStore


def test_production_requires_postgres_url_and_rejects_ephemeral_sqlite():
    settings = Settings(app_env="production")
    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        settings.validate_safety()


def test_production_with_postgres_url_passes_safety_validation():
    settings = Settings(
        app_env="production",
        database_url="postgresql://bot:secret@localhost:5432/signals",
    )
    settings.validate_safety()
    assert settings.database_backend == "postgresql"


def test_development_defaults_to_sqlite():
    settings = Settings()
    assert settings.database_backend == "sqlite"


def test_store_health_checks_integrity_and_rolls_back_probe_write(tmp_path):
    path = tmp_path / "signals.sqlite3"
    store = SignalStore(str(path))
    assert store.health() is True
    with sqlite3.connect(path) as db:
        probe = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_health_probe'"
        ).fetchone()
    assert probe is None


def test_store_health_fails_when_database_is_corrupt(tmp_path):
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        SignalStore(str(path))


def test_health_endpoint_delegates_to_lightweight_readiness(monkeypatch):
    import asyncio
    import app.main as main

    expected = object()

    async def fake_ping():
        return expected

    def fail_if_diagnostics_run():
        raise AssertionError("health must not build the full diagnostics snapshot")

    monkeypatch.setattr(main, "ping", fake_ping)
    monkeypatch.setattr(main, "_operational_snapshot", fail_if_diagnostics_run)
    assert asyncio.run(main.health()) is expected
