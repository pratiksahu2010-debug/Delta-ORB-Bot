import json
import logging

from app.logging_config import JsonRedactingFormatter, redact_payload, redact_secrets


def test_json_formatter_emits_structured_fields_and_redacts_credentials():
    record = logging.LogRecord(
        name="app.test",
        level=logging.ERROR,
        pathname="test_logging_config.py",
        lineno=17,
        msg=(
            "request failed api_key=API_SECRET password:PASS_SECRET "
            "Authorization: Bearer BEARER_SECRET "
            "postgresql://dbuser:DB_PASSWORD@db.example.com/bot "
            "123456:TELEGRAM_BOT_TOKEN_SECRET journal_key=JOURNAL_KEY_SECRET "
            '"client_secret":"JSON_SECRET"'
        ),
        args=(),
        exc_info=None,
    )

    payload = json.loads(JsonRedactingFormatter().format(record))

    assert payload["level"] == "ERROR"
    assert payload["logger"] == "app.test"
    assert payload["source"]["line"] == 17
    assert payload["message"].count("[REDACTED]") >= 4
    for secret in (
        "API_SECRET", "PASS_SECRET", "BEARER_SECRET",
        "DB_PASSWORD", "TELEGRAM_BOT_TOKEN_SECRET", "JSON_SECRET", "JOURNAL_KEY_SECRET",
    ):
        assert secret not in payload["message"]


def test_secret_redaction_also_covers_exception_text():
    try:
        raise RuntimeError("database password=EXCEPTION_SECRET")
    except RuntimeError:
        record = logging.LogRecord(
            name="app.test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="database operation failed", args=(), exc_info=None,
        )
        record.exc_info = __import__("sys").exc_info()

    payload = json.loads(JsonRedactingFormatter().format(record))
    assert "EXCEPTION_SECRET" not in payload["exception"]
    assert "[REDACTED]" in payload["exception"]


def test_redact_secrets_is_idempotent_for_already_redacted_values():
    redacted = redact_secrets("token=[REDACTED]")
    assert redacted == "token=[REDACTED]"


def test_recursive_operational_payload_redaction_masks_sensitive_keys_and_values():
    payload = redact_payload({
        "api_key": "TOP_LEVEL_SECRET",
        "message": "request token=INLINE_SECRET failed",
        "nested": {
            "database_url": "postgresql://user:DB_SECRET@host/db",
            "state": "healthy",
        },
    })
    assert payload["api_key"] == "[REDACTED]"
    assert "INLINE_SECRET" not in payload["message"]
    assert payload["nested"]["database_url"] == "[REDACTED]"
    assert payload["nested"]["state"] == "healthy"


def test_telegram_bot_token_is_redacted_inside_httpx_request_urls():
    record = logging.LogRecord(
        name="httpx", level=logging.INFO, pathname="client.py", lineno=1,
        msg='HTTP Request: POST https://api.telegram.org/bot123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890/setWebhook "HTTP/1.1 200 OK"',
        args=(), exc_info=None,
    )
    formatted = JsonRedactingFormatter().format(record)
    assert "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890" not in formatted
    assert "[REDACTED_TELEGRAM_TOKEN]" in formatted


def test_configure_logging_suppresses_http_client_request_urls():
    from app.logging_config import configure_logging

    configure_logging(logging.INFO)
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
