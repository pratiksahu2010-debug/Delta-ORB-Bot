from pathlib import Path


def test_render_blueprint_enforces_production_database_safety():
    render_config = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(
        encoding="utf-8"
    )
    assert "      - key: APP_ENV\n        value: production\n" in render_config
    # The database URL must remain a dashboard-managed secret, not a committed value.
    assert "DATABASE_URL" not in render_config


def test_render_blueprint_keeps_order_execution_disabled():
    render_config = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(
        encoding="utf-8"
    )
    assert '      - key: SIGNAL_ONLY\n        value: "true"\n' in render_config
    assert '      - key: ORDERS_ENABLED\n        value: "false"\n' in render_config
    assert "    healthCheckPath: /ready\n" in render_config


def test_render_blueprint_matches_the_configured_free_service_plan():
    render_config = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(
        encoding="utf-8"
    )
    assert "    plan: free\n" in render_config
    assert "    plan: starter\n" not in render_config
