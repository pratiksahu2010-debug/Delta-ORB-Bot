import numpy as np
import pandas as pd

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((loss == 0) & (gain > 0), 100)
    out = out.mask((gain == 0) & (loss > 0), 0)
    return out

def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = frame.close.shift(1)
    tr = pd.concat([(frame.high-frame.low), (frame.high-prev).abs(), (frame.low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def adx(frame: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = frame.high.diff()
    down = -frame.low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev = frame.close.shift(1)
    tr = pd.concat([(frame.high-frame.low), (frame.high-prev).abs(), (frame.low-prev).abs()], axis=1).max(axis=1)
    smooth_tr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / smooth_tr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / smooth_tr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean(), plus_di, minus_di

def vwap(frame: pd.DataFrame) -> pd.Series:
    typical = (frame.high + frame.low + frame.close) / 3
    # Caller must pass one exchange-session slice; reset at session boundary.
    vol = frame.volume.fillna(0)
    total = vol.cumsum()
    return (typical * vol).cumsum() / total.replace(0, np.nan)
