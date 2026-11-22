"""PostgreSQL connection-string safety shared by durable stores.

Neon connection URIs use PostgreSQL's standard URI format. Enforce TLS even
when the dashboard URI omits an explicit sslmode parameter.
"""
from psycopg.conninfo import conninfo_to_dict, make_conninfo


_ALLOWED_TLS_MODES = {"require", "verify-ca", "verify-full"}


def postgres_conninfo(database_url: str) -> str:
    """Return a validated libpq connection string with TLS required.

    PostgreSQL providers may emit the legacy postgres:// scheme; normalize it
    to postgresql://. Refuse sslmode values that could permit plaintext.
    """
    url = (database_url or "").strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    try:
        params = conninfo_to_dict(url)
    except Exception as exc:
        raise ValueError("DATABASE_URL must be a valid PostgreSQL connection URI") from exc

    sslmode = str(params.get("sslmode", "")).lower()
    if sslmode and sslmode not in _ALLOWED_TLS_MODES:
        raise ValueError(
            "DATABASE_URL sslmode must be require, verify-ca, or verify-full; "
            "plaintext-capable SSL modes are not allowed"
        )
    params["sslmode"] = sslmode or "require"
    return make_conninfo(**params)
