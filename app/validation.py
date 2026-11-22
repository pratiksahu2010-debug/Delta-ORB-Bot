import math
from datetime import datetime
from .models import Quote


def validate_quote(
    quote: Quote, now: datetime, max_age_seconds: float,
    reference: float, atr: float, max_extension_atr: float,
) -> tuple[bool, str]:
    values = (quote.price, atr, reference, max_age_seconds, max_extension_atr)
    if not all(math.isfinite(value) for value in values):
        return False, "non_finite_quote_or_threshold"
    if quote.price <= 0 or atr <= 0 or max_age_seconds <= 0 or max_extension_atr < 0:
        return False, "invalid_price_or_atr"
    age = (now - quote.received_at).total_seconds()
    if age < -1 or age > max_age_seconds:
        return False, "stale_or_future_quote"
    if abs(quote.price - reference) / atr > max_extension_atr:
        return False, "entry_too_extended"
    return True, "ok"
