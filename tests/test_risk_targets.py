import pandas as pd
import pytest

from app.momentum import _risk
from app.risk import LIVE_TARGET_R_MULTIPLE


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_shared_structural_risk_uses_two_r_live_target(side):
    frame = pd.DataFrame({
        "low": [99.0, 99.2, 99.1, 99.3, 99.0],
        "high": [101.0, 101.2, 101.1, 101.3, 101.0],
    })
    result = _risk(frame, side, 100.0, 1.0)
    assert result is not None
    risk = (100.0 - result["stop_loss"]) if side == "LONG" else (result["stop_loss"] - 100.0)
    reward = (result["take_profit_1"] - 100.0) if side == "LONG" else (100.0 - result["take_profit_1"])
    assert LIVE_TARGET_R_MULTIPLE == 2.0
    assert reward / risk == pytest.approx(2.0)
