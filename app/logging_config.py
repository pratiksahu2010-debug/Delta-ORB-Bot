"""JSON logging with defensive redaction for operational diagnostics."""
from datetime import datetime, timezone
import json
import logging
import re


_SECRET_PATTERNS = (
    (
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/-]+=*"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"""(?i)(["']?\b(?:api[_-]?key|access[_-]?key|private[_-]?key|journal[_-]?key|token|refresh[_-]?token|secret|webhook[_-]?secret|password|authorization|bot[_-]?token|client[_-]?secret)["']?\s*[:=]\s*["']?)([^"'\s,;}]+)"""
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^:/\s]+:)[^@/\s]+(@)"),
        r"\1[REDACTED]\2",
    ),
    (
        re.compile(r"(?i)(https?://api\.telegram\.org/bot)\d{6,}:[A-Za-z0-9_-]{20,}"),
        r"\1[REDACTED_TELEGRAM_TOKEN]",
    ),
    (
        re.compile(r"(?i)(bot)\d{6,}:[A-Za-z0-9_-]{20,}"),
        r"\1[REDACTED_TELEGRAM_TOKEN]",
    ),
    (
        re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b"),
        "[REDACTED_TELEGRAM_TOKEN]",
    ),
)


def redact_secrets(value: str) -> str:
    text = str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


_SENSITIVE_KEY = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?key|private[_-]?key|journal[_-]?key|"
    r"token|refresh[_-]?token|secret|webhook[_-]?secret|password|"
    r"authorization|bot[_-]?token|client[_-]?secret|database[_-]?url|dsn)"
)


def redact_payload(value):
    """Recursively redact credential-bearing keys and secret-like string values."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        return redact_secrets(value)
    return value


class JsonRedactingFormatter(logging.Formatter):
    """Emit one JSON object per log event and redact common credential formats."""

    def format(self, record: logging.LogRecord) -> str:
        message = redact_secrets(record.getMessage())
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "source": {
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            },
        }
        if record.exc_info:
            payload["exception"] = redact_secrets(self.formatException(record.exc_info))
        if record.stack_info:
            payload["stack"] = redact_secrets(record.stack_info)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def configure_logging(level: int) -> None:
    """Configure root logging once; preserve handlers but use structured redaction."""
    root = logging.getLogger()
    root.setLevel(level)
    formatter = JsonRedactingFormatter()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for handler in root.handlers:
        handler.setFormatter(formatter)
    # HTTPX logs full request URLs at INFO; Telegram embeds its bot token in
    # the URL path. Suppress those request logs even with defensive redaction.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
