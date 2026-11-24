import logging
import os
import math
import json
from datetime import time as datetime_time
from zoneinfo import ZoneInfo
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "development"
    # ADVIKA is an isolated signal-only module; ALL remains backward-compatible.
    bot_module: str = "ORB"
    advika_enabled: bool = True
    advika_dead_hours_ist: str = ""
    log_level: str = "INFO"
    signal_only: bool = True
    orders_enabled: bool = False
    symbols: str = "BTCUSD,ETHUSD,SOLUSD"
    momentum_symbols_file: str = "config/momentum_symbols.txt"
    tpr_rb_symbols_file: str = "config/tpr_rb_symbols.txt"
    adx_min: float = 17.0
    mex_enabled: bool = True
    orb_enabled: bool = True
    orb_min_rvol: float = 1.10
    # JSON: session -> [IANA timezone, local HH:MM opening, active minutes].
    orb_sessions: str = '{"asia":["Asia/Tokyo","09:00",120],"london":["Europe/London","08:00",120],"new_york":["America/New_York","09:30",120]}'
    tpr_enabled: bool = True
    rb_enabled: bool = True
    rb_min_rvol: float = 1.45
    rb_range_lookback: int = 12
    mex_min_rvol: float = 1.50
    mex_min_atr_expansion: float = 1.15
    mex_live_min_eval_interval_seconds: float = 0.75
    candle_timeframe: str = "5m"
    context_timeframe: str = "15m"
    quote_max_age_seconds: float = 3.0
    signal_max_age_bars: int = 2
    max_entry_extension_atr: float = 0.8
    trigger_confirmation_seconds: float = 0.5
    trigger_confirmation_max_gap_seconds: float = 1.5
    max_candle_lag_seconds: int = 600
    candle_poll_interval_seconds: float = 30.0
    # Shadow ML is research-only and disabled in the lightweight production profile.
    shadow_ml_enabled: bool = False
    state_db_path: str = "./data/signal_state.sqlite3"
    database_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_test_key: str = ""
    journal_admin_key: str = ""
    telegram_webhook_secret: str = ""
    telegram_allowed_user_ids: str = ""
    # Outcome-only measurement settings; these never change live signal targets.
    outcome_horizon_bars: int = Field(default=12, ge=1, le=288)
    outcome_tp2_r_multiple: float = Field(default=2.0, gt=1.5, le=10.0)
    # Round-trip percentage-point costs. Configure these from the account fee tier.
    outcome_fee_pct: float = Field(default=0.0, ge=0.0, le=5.0)
    outcome_slippage_pct: float = Field(default=0.0, ge=0.0, le=5.0)
    delta_base_url: str = "https://api.india.delta.exchange"

    @staticmethod
    def _read_symbols(path: str, expected_count: int | None = None) -> list[str]:
        file_path = Path(path)
        if not file_path.is_file():
            raise ValueError(f"Configured symbol universe file does not exist: {path}")
        values = []
        seen = set()
        for line in file_path.read_text(encoding="utf-8").splitlines():
            symbol = line.split("#", 1)[0].strip().upper()
            if not symbol:
                continue
            if symbol in seen:
                raise ValueError(f"Duplicate symbol {symbol} in configured universe: {path}")
            seen.add(symbol)
            values.append(symbol)
        if not values:
            raise ValueError(f"Configured symbol universe is empty: {path}")
        if expected_count is not None and len(values) != expected_count:
            raise ValueError(
                f"Configured symbol universe must contain exactly {expected_count} unique symbols: "
                f"{path} contains {len(values)}"
            )
        return values

    @property
    def enabled_modules(self) -> list[str]:
        mode = str(self.bot_module or "ALL").strip().upper()
        if mode == "ALL":
            modules = ["ORB", "MEX"]
        elif mode in {"ORB", "MEX"}:
            modules = [mode]
        elif mode == "ADVIKA":
            modules = []
        else:
            raise ValueError("BOT_MODULE must be ALL, ORB, MEX, or ADVIKA")
        if self.advika_enabled:
            modules.append("ADVIKA")
        return modules
        raise ValueError("BOT_MODULE must be ALL, ORB, MEX, or ADVIKA")

    def module_enabled(self, module: str) -> bool:
        name = str(module).strip().upper()
        # Advika Choice is an independent strategy lane. It can run beside
        # whichever legacy BOT_MODULE is selected without replacing that module.
        if name == "ADVIKA":
            return bool(self.advika_enabled)
        return name in self.enabled_modules

    def strategy_enabled(self, strategy: str) -> bool:
        name = str(strategy).strip().upper()
        if name == "ORB":
            return self.module_enabled("ORB") and self.orb_enabled
        if name == "RB":
            return self.module_enabled("ORB") and self.rb_enabled
        if name == "TPR":
            return self.module_enabled("ORB") and self.tpr_enabled
        if name == "MEX":
            return self.module_enabled("MEX") and self.mex_enabled
        if name == "ADVIKA":
            return self.module_enabled("ADVIKA") and self.advika_enabled
        return False

    @property
    def mex_core_symbol_list(self) -> list[str]:
        """The 60-symbol MEX seed list, expanded to the complete scanner universe below."""
        return self._read_symbols(self.momentum_symbols_file, expected_count=60)

    @property
    def tpr_rb_symbol_list(self) -> list[str]:
        """The 60-symbol subset eligible for the RB and TPR setup types."""
        return self._read_symbols(self.tpr_rb_symbols_file, expected_count=60)

    @property
    def symbol_list(self) -> list[str]:
        # One fixed 120-symbol crypto universe is shared by both top-level modules.
        mex_core = self.mex_core_symbol_list
        tpr_rb = self.tpr_rb_symbol_list
        overlap = sorted(set(mex_core) & set(tpr_rb))
        if overlap:
            raise ValueError(f"MEX core and TPR+RB symbol lists overlap: {overlap}")
        return tpr_rb + mex_core

    @property
    def momentum_symbol_list(self) -> list[str]:
        # MEX uses the same complete scanner universe as the active deployment.
        return self.symbol_list

    @property
    def telegram_feedback_user_ids(self) -> list[str]:
        """Return the explicit feedback allowlist, or the private-chat owner ID.

        Group chats fail closed unless TELEGRAM_ALLOWED_USER_IDS is configured.
        """
        configured = list(dict.fromkeys(
            value.strip() for value in self.telegram_allowed_user_ids.split(",")
            if value.strip()
        ))
        if configured:
            return configured
        chat_id = str(self.telegram_chat_id or "").strip()
        if chat_id.isdigit() and int(chat_id) > 0:
            return [chat_id]
        return []

    @property
    def uses_default_state_db_path(self) -> bool:
        return Path(self.state_db_path) == Path("./data/signal_state.sqlite3")

    @property
    def database_backend(self) -> str:
        return "postgresql" if self.database_url.strip() else "sqlite"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"} or bool(os.getenv("RENDER_SERVICE_ID"))


    @property
    def orb_session_config(self) -> dict[str, tuple[str, str, int]]:
        try:
            raw = json.loads(self.orb_sessions)
            if not isinstance(raw, dict) or not raw:
                raise ValueError
            sessions = {}
            for name, values in raw.items():
                if name not in {"asia", "london", "new_york"} or not isinstance(values, list) or len(values) != 3:
                    raise ValueError
                zone_name, open_time, active_minutes = values
                ZoneInfo(str(zone_name))
                parsed_open = datetime_time.fromisoformat(str(open_time))
                if parsed_open.tzinfo is not None or parsed_open.isoformat(timespec="minutes") != str(open_time):
                    raise ValueError
                if isinstance(active_minutes, bool) or int(active_minutes) != active_minutes or int(active_minutes) < 1:
                    raise ValueError
                sessions[name] = (str(zone_name), str(open_time), int(active_minutes))
            return sessions
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            raise ValueError("ORB_SESSIONS must be a JSON object of supported sessions with valid IANA timezones, HH:MM openings and positive active minutes") from None

    def validate_safety(self) -> None:
        _ = self.enabled_modules  # validates BOT_MODULE before the engine starts
        _ = self.orb_session_config  # validates ORB session settings
        from .advika_choice import parse_dead_hours
        parse_dead_hours(self.advika_dead_hours_ist)
        if not self.signal_only or self.orders_enabled:
            raise ValueError("Unsafe configuration: signal-only must be true and order execution must be disabled")
        float_thresholds = (
            self.adx_min, self.orb_min_rvol,
            self.rb_min_rvol, self.mex_min_rvol, self.mex_min_atr_expansion,
            self.mex_live_min_eval_interval_seconds, self.quote_max_age_seconds,
            self.max_entry_extension_atr, self.trigger_confirmation_seconds,
            self.trigger_confirmation_max_gap_seconds, self.candle_poll_interval_seconds,
        )
        if (not all(math.isfinite(value) for value in float_thresholds)
                or self.adx_min < 0 or self.orb_min_rvol <= 0
                or self.rb_min_rvol <= 0 or self.rb_range_lookback < 8
                or self.mex_min_rvol <= 0 or self.mex_min_atr_expansion <= 0
                or self.mex_live_min_eval_interval_seconds < 0
                or self.quote_max_age_seconds <= 0 or self.max_entry_extension_atr <= 0
                or self.signal_max_age_bars < 1
                or self.max_candle_lag_seconds < 1 or self.candle_poll_interval_seconds <= 0
                or self.trigger_confirmation_seconds < 0
                or self.trigger_confirmation_max_gap_seconds <= 0
                or self.trigger_confirmation_max_gap_seconds < self.trigger_confirmation_seconds):
            raise ValueError("Invalid strategy thresholds")
        if self.is_production:
            database_url = self.database_url.strip().lower()
            if not database_url:
                raise ValueError("DATABASE_URL is required in production; refusing ephemeral SQLite state storage")
            if not database_url.startswith(("postgres://", "postgresql://")):
                raise ValueError("Production DATABASE_URL must point to PostgreSQL/Neon; SQLite is not permitted")


settings = Settings()
