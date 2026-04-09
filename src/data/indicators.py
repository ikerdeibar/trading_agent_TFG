from __future__ import annotations
import pandas as pd


def _bars_to_df(bars: list) -> pd.DataFrame:
    rows = [{"close": float(b.close), "high": float(b.high),
             "low": float(b.low), "volume": int(b.volume)} for b in bars]
    return pd.DataFrame(rows)


def compute_indicators(bars: list) -> dict:
    """
    Given a list of Alpaca Bar objects (oldest→newest),
    return a dict of TA indicators for the most recent bar.
    Requires at least 50 bars for all indicators to be valid.
    """
    if len(bars) < 2:
        return {"rsi": None, "macd": None, "macd_signal": None,
                "ma20": None, "ma50": None, "bars_available": len(bars)}

    df = _bars_to_df(bars)
    close = df["close"]

    ma20 = round(float(close.rolling(20).mean().iloc[-1]), 4) if len(df) >= 20 else None
    ma50 = round(float(close.rolling(50).mean().iloc[-1]), 4) if len(df) >= 50 else None

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    rsi = round(float(rsi_series.iloc[-1]), 2) if not rsi_series.empty else None

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd        = round(float(macd_line.iloc[-1]), 4)
    macd_signal = round(float(signal_line.iloc[-1]), 4)

    return {
        "rsi":         rsi,
        "macd":        macd,
        "macd_signal": macd_signal,
        "macd_hist":   round(macd - macd_signal, 4),
        "ma20":        ma20,
        "ma50":        ma50,
        "bars_available": len(df),
    }
