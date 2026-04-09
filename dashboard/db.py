"""
db.py — single source of truth for all DB queries and metric computations.

Schema matches SQLAlchemy models in src/storage/models.py (PostgreSQL, snake_case):
  portfolio_snapshots   : arm_id, cash, equity, total_value, captured_at, …
  trades                : id, arm_id, symbol, side, qty, price, status, created_at, …
  agent_decision_logs   : arm_id, cycle_ts, orders_proposed, llm_tokens_used, …
  market_snapshots      : symbol, price, captured_at, …  (price exposed as mid in loads)
  sentiment_snapshots   : cycle_ts, symbol, label, score, confidence, created_at
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st

ARM_ORDER  = ["A", "B", "C", "D"]
# Must match configs/experiments.yaml (ARM_ID env per process).
ARM_LABELS = {
    "A": "Arm A · Qwen 235B · Monolithic",
    "B": "Arm B · Qwen 235B · Council",
    "C": "Arm C · GPT-4.1 · Monolithic",
    "D": "Arm D · GPT-4.1 · Council",
}
# A/B: cyan vs indigo so both OpenRouter arms read clearly on dark charts (C/D unchanged).
ARM_COLORS = {
    "A": "#22d3ee",
    "B": "#818cf8",
    "C": "#fdab43",
    "D": "#a86fdf",
}
FILL_COLORS = {
    "A": "rgba(34,211,238,0.20)",
    "B": "rgba(129,140,248,0.20)",
    "C": "rgba(253,171,67,0.18)",
    "D": "rgba(168,111,223,0.18)",
}
ARM_PROVIDER = {"A": "OpenRouter", "B": "OpenRouter", "C": "OpenAI", "D": "OpenAI"}
# Blended $/1M total tokens for rough “corrected” cost on Home (see src/agents/llm.py rates).
COST_RATE_PER_M = {"A": 0.2105, "B": 0.2105, "C": 3.204, "D": 3.204}
START_EQUITY = 100_000.0
BENCHMARK = "SPY"
DATA_START = "2026-03-15"

PLOT_LAYOUT = dict(
    plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
    font=dict(family="sans-serif", color="#cdccca", size=12),
)

def _conn():
    return st.connection("postgresql", type="sql")

@st.cache_data(ttl=300)
def _q(sql: str) -> pd.DataFrame:
    return _conn().query(sql, ttl="5m")

@st.cache_data(ttl=300)
def load_portfolio_snapshots() -> pd.DataFrame:
    df = _q(f"""
        SELECT arm_id, cash, equity, total_value, captured_at
        FROM portfolio_snapshots
        WHERE captured_at >= '{DATA_START}'
        ORDER BY arm_id, captured_at
    """)
    if df.empty:
        return df
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df["date"]        = df["captured_at"].dt.tz_convert("America/New_York").dt.date
    return df

@st.cache_data(ttl=300)
def load_trades() -> pd.DataFrame:
    df = _q(f"""
        SELECT id, arm_id, symbol, side, qty, price, status, created_at
        FROM trades
        WHERE created_at >= '{DATA_START}'
        ORDER BY arm_id, created_at
    """)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["date"]       = df["created_at"].dt.tz_convert("America/New_York").dt.date
    df["notional"]   = (df["qty"] * df["price"]).round(2)
    # Broker persists Alpaca-style BUY/SELL; dashboard filters use buy/sell.
    df["side"] = df["side"].astype(str).str.strip().str.lower()
    return df

@st.cache_data(ttl=300)
def load_agent_decisions() -> pd.DataFrame:
    df = _q(f"""
        SELECT id, arm_id, cycle_ts, agent, reasoning,
               orders_proposed, orders_executed, orders_blocked,
               memory_packet_in, sentiment_per_symbol,
               llm_tokens_used, llm_cost_usd,
               post_mortem, created_at
        FROM agent_decision_logs
        WHERE cycle_ts >= '{DATA_START}'
        ORDER BY arm_id, cycle_ts
    """)
    if df.empty:
        return df
    df["cycle_ts"]   = pd.to_datetime(df["cycle_ts"],   utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["date"]       = df["cycle_ts"].dt.tz_convert("America/New_York").dt.date
    df["llm_tokens_used"] = pd.to_numeric(df["llm_tokens_used"], errors="coerce").fillna(0)
    df["llm_cost_usd"]    = pd.to_numeric(df["llm_cost_usd"],    errors="coerce").fillna(0)
    df["cost_corrected"] = df.apply(
        lambda r: r["llm_tokens_used"] * COST_RATE_PER_M[r["arm_id"]] / 1_000_000, axis=1
    )
    return df

@st.cache_data(ttl=300)
def load_market_snapshots() -> pd.DataFrame:
    df = _q(f"""
        SELECT symbol, price AS mid, captured_at
        FROM market_snapshots
        WHERE captured_at >= '{DATA_START}'
        ORDER BY symbol, captured_at
    """)
    if df.empty:
        return df
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df["date"]        = df["captured_at"].dt.tz_convert("America/New_York").dt.date
    return df

@st.cache_data(ttl=300)
def load_sentiment_snapshots() -> pd.DataFrame:
    df = _q(f"""
        SELECT cycle_ts, symbol, label, score, confidence, created_at
        FROM sentiment_snapshots
        WHERE cycle_ts >= '{DATA_START}'
        ORDER BY cycle_ts, symbol
    """)
    if df.empty:
        return df
    df["cycle_ts"]   = pd.to_datetime(df["cycle_ts"],   utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["date"]       = df["cycle_ts"].dt.tz_convert("America/New_York").dt.date
    df["score"]      = pd.to_numeric(df["score"],       errors="coerce").fillna(0)
    df["confidence"] = pd.to_numeric(df["confidence"],  errors="coerce").fillna(0)
    return df

def compute_daily_equity(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_df.empty:
        return pd.DataFrame(columns=["arm_id", "date", "equity"])
    return (
        portfolio_df.sort_values("captured_at")
        .groupby(["arm_id", "date"])["equity"]
        .last()
        .reset_index()
    )

def compute_returns(daily_eq: pd.DataFrame) -> pd.DataFrame:
    if daily_eq.empty:
        return pd.DataFrame(columns=["arm_id","date","equity","cum_return_pct","daily_return_pct"])
    rows = []
    for armid, grp in daily_eq.groupby("arm_id"):
        g = grp.sort_values("date").copy()
        g["cum_return_pct"]   = (g["equity"] / START_EQUITY - 1) * 100
        g["daily_return_pct"] = g["equity"].pct_change() * 100
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def compute_sharpe(returns_df: pd.DataFrame, trading_days: int = 252) -> dict:
    result = {}
    for armid, grp in returns_df.groupby("arm_id"):
        dr = grp["daily_return_pct"].dropna()
        if len(dr) < 2 or dr.std() == 0:
            result[armid] = None
        else:
            result[armid] = round((dr.mean() / dr.std()) * np.sqrt(trading_days), 3)
    return result

def compute_max_drawdown(returns_df: pd.DataFrame) -> dict:
    result = {}
    for armid, grp in returns_df.groupby("arm_id"):
        eq = grp.sort_values("date")["equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        result[armid] = round(dd.min(), 2)
    return result

def compute_block_rate(decisions_df: pd.DataFrame) -> pd.DataFrame:
    import json
    rows = []
    for armid, grp in decisions_df.groupby("arm_id"):
        proposed = executed = blocked = 0
        for _, row in grp.iterrows():
            for col, counter in [("orders_proposed","proposed"),
                                  ("orders_executed","executed"),
                                  ("orders_blocked","blocked")]:
                try:
                    v = row.get(col)
                    d = json.loads(v) if isinstance(v, str) else v
                    n = len(d) if isinstance(d, list) else 0
                except Exception:
                    n = 0
                if counter == "proposed": proposed += n
                elif counter == "executed": executed += n
                else: blocked += n
        rows.append({"arm_id": armid, "proposed": proposed,
                     "executed": executed, "blocked": blocked})
    return pd.DataFrame(rows)


# ── Thesis §3.1.4.2 — Return on Intelligence helpers ─────────────────────

def compute_roi(returns_df: pd.DataFrame, decisions_df: pd.DataFrame) -> dict:
    """ROI = Net Return ($) / Total Operational Cost ($) per arm."""
    result = {}
    for arm in ARM_ORDER:
        arm_r = returns_df[returns_df["arm_id"] == arm]
        arm_d = decisions_df[decisions_df["arm_id"] == arm]
        net_return = arm_r["equity"].iloc[-1] - START_EQUITY if not arm_r.empty else 0.0
        total_cost = arm_d["llm_cost_usd"].sum()
        result[arm] = round(net_return / total_cost, 2) if total_cost > 0 else None
    return result


def compute_cost_per_bp(returns_df: pd.DataFrame, decisions_df: pd.DataFrame) -> dict:
    """Cost per basis point of return = Total Cost / abs(cum return in bps)."""
    result = {}
    for arm in ARM_ORDER:
        arm_r = returns_df[returns_df["arm_id"] == arm]
        arm_d = decisions_df[decisions_df["arm_id"] == arm]
        cum_ret_pct = arm_r["cum_return_pct"].iloc[-1] if not arm_r.empty else 0.0
        bps = abs(cum_ret_pct) * 100
        total_cost = arm_d["llm_cost_usd"].sum()
        result[arm] = round(total_cost / bps, 4) if bps > 0 else None
    return result


def compute_action_rate(decisions_df: pd.DataFrame) -> dict:
    """Action rate (α) = cycles with ≥1 BUY/SELL / total cycles per arm."""
    import json
    result = {}
    for arm in ARM_ORDER:
        grp = decisions_df[decisions_df["arm_id"] == arm]
        total_cycles = len(grp)
        active_cycles = 0
        for _, row in grp.iterrows():
            try:
                v = row.get("orders_executed")
                d = json.loads(v) if isinstance(v, str) else v
                if isinstance(d, list) and len(d) > 0:
                    active_cycles += 1
            except Exception:
                pass
        result[arm] = round(active_cycles / total_cycles * 100, 1) if total_cycles > 0 else 0.0
    return result


# ── SPY buy-and-hold benchmark (§3.1.2) ──────────────────────────────────

def _alpaca_keys() -> tuple[str, str]:
    """Resolve Alpaca API credentials from env vars or Streamlit secrets."""
    import os
    key = secret = ""
    for k in ("ALPACA_API_KEY", "APCA_API_KEY_ID", "ALPACA_KEY_A"):
        key = os.environ.get(k, "")
        if key:
            break
    for s in ("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_SECRET_A"):
        secret = os.environ.get(s, "")
        if secret:
            break
    if not key:
        try:
            key = st.secrets["ALPACA_API_KEY"]
        except Exception:
            pass
    if not secret:
        try:
            secret = st.secrets["ALPACA_SECRET_KEY"]
        except Exception:
            pass
    return key or "", secret or ""


@st.cache_data(ttl=3600)
def load_spy_benchmark(start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """Fetch daily SPY closes from Alpaca and compute buy-and-hold returns
    normalised to START_EQUITY.  Dates passed as ISO strings for cache safety."""
    import datetime as dt
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        api_key, api_secret = _alpaca_keys()
        if not api_key or not api_secret:
            return pd.DataFrame()

        start = dt.datetime.fromisoformat(start_date_str)
        end   = dt.datetime.fromisoformat(end_date_str)

        client = StockHistoricalDataClient(api_key, api_secret)
        req = StockBarsRequest(
            symbol_or_symbols=[BENCHMARK],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(req)
        rows = [{"date": b.timestamp.date(), "close": float(b.close)} for b in bars[BENCHMARK]]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
        p0 = df["close"].iloc[0]
        df["equity"]         = START_EQUITY * df["close"] / p0
        df["cum_return_pct"] = (df["close"] / p0 - 1) * 100
        return df
    except Exception:
        return pd.DataFrame()