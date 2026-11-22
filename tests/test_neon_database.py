import pytest
from psycopg.conninfo import conninfo_to_dict

from app.database import postgres_conninfo


def test_neon_pooled_uri_preserves_endpoint_and_enforces_tls():
    uri = (
        "postgresql://neon_user:secret@ep-example-pooler.us-east-2.aws.neon.tech/"
        "appdb?channel_binding=require"
    )
    params = conninfo_to_dict(postgres_conninfo(uri))
    assert params["host"] == "ep-example-pooler.us-east-2.aws.neon.tech"
    assert params["dbname"] == "appdb"
    assert params["sslmode"] == "require"
    assert params["channel_binding"] == "require"


def test_existing_strong_tls_mode_is_preserved():
    uri = "postgresql://user:secret@db.example.test/app?sslmode=verify-full"
    assert conninfo_to_dict(postgres_conninfo(uri))["sslmode"] == "verify-full"


def test_legacy_postgres_scheme_is_normalized():
    uri = "postgres://user:secret@db.example.test/app?sslmode=require"
    params = conninfo_to_dict(postgres_conninfo(uri))
    assert params["host"] == "db.example.test"
    assert params["sslmode"] == "require"


@pytest.mark.parametrize("sslmode", ["disable", "allow", "prefer"])
def test_plaintext_capable_ssl_modes_are_rejected(sslmode):
    uri = f"postgresql://user:secret@db.example.test/app?sslmode={sslmode}"
    with pytest.raises(ValueError, match="plaintext-capable"):
        postgres_conninfo(uri)


def test_invalid_database_uri_fails_with_clear_error():
    with pytest.raises(ValueError, match="valid PostgreSQL connection URI"):
        postgres_conninfo("not-a-postgresql-uri")
