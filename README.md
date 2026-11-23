# Delta Exchange India Signal Bot — ORB

This is an isolated single-module build from the supplied combined bot source.

## Strategy scope
Only the ORB module: Session ORB plus its nested Range Breakout (RB) and Trend Pullback Rejection (TPR) setup types. The MEX detector and live-forming-candle evaluator are removed from this package.

- Fixed market-data universe: 120 symbols.
- 5-minute completed candles and live quote confirmation are retained.
- ORB defaults: Tokyo 09:00, London 08:00 and New York 09:30 in each session's local timezone. The first completed 5-minute candle defines the opening range; breakouts are eligible for the next 120 minutes. IANA timezones handle daylight-saving changes. Configure with `ORB_SESSIONS`.
- Existing indicator calculations, strategy thresholds, trigger confirmation, structural stop validation, signal journal, Telegram delivery and market-data handling are retained.
- 15-minute context remains informational; this split does not add a new 15-minute blocking gate.
- Signal-only: `SIGNAL_ONLY=true`, `ORDERS_ENABLED=false`. Order execution is not implemented/enabled.

## Run locally
1. Use Python 3.11.9.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Fill Delta/Telegram/database settings. Keep `SIGNAL_ONLY=true` and `ORDERS_ENABLED=false`.
5. Run: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

## Render deployment
- Create a separate Render Web Service for this ZIP/repository.
- Use `render.yaml` or the commands above.
- Set a dedicated PostgreSQL `DATABASE_URL` for this bot. **Do not point the MEX and ORB bots at the same database**; this keeps setup lifecycle, journal and Telegram outbox state isolated.
- Configure its own Telegram bot token/chat ID if you want separate alert channels.
- The blueprint uses the free web-service plan; the database is provisioned separately.

## Environment isolation
`BOT_MODULE=ORB` identifies this single-module distribution. MEX detector code and live MEX candle evaluation are not shipped here.

## Checks
Run the targeted isolation/health tests:
`pytest -q tests/test_symbol_universes.py tests/test_render_config.py tests/test_operational_endpoints.py`

The full test suite should be run in an environment with all `requirements.txt` dependencies installed, including `psycopg[binary]`.
