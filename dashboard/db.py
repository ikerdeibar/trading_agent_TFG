"""
db.py — single source of truth for all dashboard data + metrics.

Two source modes, picked by the env var ``DASHBOARD_SOURCE``:

* ``snapshot`` (DEFAULT) — read frozen Parquet files from ``dashboard/data/``.
  This is what the deployed app uses. No DB credentials needed.
* ``live`` — read directly from PostgreSQL via ``st.connection``. Useful for
  local development against the running container or Cloud SQL.

Schema matches SQLAlchemy models in ``src/storage/models.py``:
  portfolio_snapshots   : arm_id, cash, equity, total_value, captured_at
  trades                : id, arm_id, symbol, side, qty, price, status, created_at
  agent_decision_logs   : arm_id, cycle_ts, orders_proposed, llm_tokens_used, …
  market_snapshots      : symbol, price (renamed to mid), captured_at
  sentiment_snapshots   : cycle_ts, symbol, label, score, confidence, created_at
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants — visual identity & business config
# ─────────────────────────────────────────────────────────────────────────────
ARM_ORDER  = ["A", "B", "C", "D"]
ARM_LABELS = {
    "A": "Arm A · Qwen 235B · Monolithic",
    "B": "Arm B · Qwen 235B · Council",
    "C": "Arm C · GPT-4.1 · Monolithic",
    "D": "Arm D · GPT-4.1 · Council",
}
ARM_SHORT = {
    "A": "A · Qwen Mono",
    "B": "B · Qwen Council",
    "C": "C · GPT Mono",
    "D": "D · GPT Council",
}
ARM_COLORS = {
    "A": "#22d3ee",   # cyan
    "B": "#818cf8",   # indigo
    "C": "#fdab43",   # amber
    "D": "#a86fdf",   # purple
}
FILL_COLORS = {
    "A": "rgba(34,211,238,0.20)",
    "B": "rgba(129,140,248,0.20)",
    "C": "rgba(253,171,67,0.18)",
    "D": "rgba(168,111,223,0.18)",
}
ARM_PROVIDER    = {"A": "OpenRouter", "B": "OpenRouter", "C": "OpenAI", "D": "OpenAI"}
COST_RATE_PER_M = {"A": 0.2105, "B": 0.2105, "C": 3.204, "D": 3.204}

START_EQUITY = 100_000.0
BENCHMARK    = "SPY"
DATA_START   = "2026-03-15"

PLOT_LAYOUT = dict(
    plot_bgcolor="#1c1b19",
    paper_bgcolor="#171614",
    font=dict(family="Inter, system-ui, sans-serif", color="#cdccca", size=12),
)

# Shared axis / legend styling — re-exported so pages can mix-and-match.
AXIS_STYLE   = dict(gridcolor="#262523", zerolinecolor="#393836")
LEGEND_STYLE = dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)")

# Semantic colours (executed/positive vs blocked/negative)
COLOR_POS = "#6daa45"
COLOR_NEG = "#d163a7"


# ─────────────────────────────────────────────────────────────────────────────
# Source resolution — Parquet snapshot vs live Postgres
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR      = DASHBOARD_DIR / "data"


def _source_mode() -> str:
    """Default to snapshot; flip with ``DASHBOARD_SOURCE=live`` for dev."""
    return os.environ.get("DASHBOARD_SOURCE", "snapshot").strip().lower()


def get_snapshot_meta() -> dict:
    """Return the manifest written by ``scripts/snapshot_db.py``, if any."""
    p = DATA_DIR / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Generic loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _load_parquet(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(ttl=300)
def _load_live(sql: str) -> pd.DataFrame:
    return st.connection("postgresql", type="sql").query(sql, ttl="5m")


_LIVE_SQL = {
    "portfolio_snapshots": f"""
        SELECT arm_id, cash, equity, total_value, captured_at
        FROM portfolio_snapshots WHERE captured_at >= '{DATA_START}'
        ORDER BY arm_id, captured_at""",
    "trades": f"""
        SELECT id, arm_id, symbol, side, qty, price, status, created_at
        FROM trades WHERE created_at >= '{DATA_START}'
        ORDER BY arm_id, created_at""",
    "agent_decision_logs": f"""
        SELECT id, arm_id, cycle_ts, agent, reasoning,
               orders_proposed, orders_executed, orders_blocked,
               memory_packet_in, sentiment_per_symbol,
               llm_tokens_used, llm_cost_usd,
               post_mortem, created_at
        FROM agent_decision_logs WHERE cycle_ts >= '{DATA_START}'
        ORDER BY arm_id, cycle_ts""",
    "market_snapshots": f"""
        SELECT symbol, price AS mid, captured_at
        FROM market_snapshots WHERE captured_at >= '{DATA_START}'
        ORDER BY symbol, captured_at""",
    "sentiment_snapshots": f"""
        SELECT cycle_ts, symbol, label, score, confidence, created_at
        FROM sentiment_snapshots WHERE cycle_ts >= '{DATA_START}'
        ORDER BY cycle_ts, symbol""",
}


def _load_table(name: str) -> pd.DataFrame:
    """Pick snapshot or live based on env."""
    if _source_mode() == "live":
        return _load_live(_LIVE_SQL[name])
    df = _load_parquet(name)
    # Live SQL renames `price → mid` for market_snapshots; harmonise here.
    if name == "market_snapshots" and not df.empty and "price" in df.columns and "mid" not in df.columns:
        df = df.rename(columns={"price": "mid"})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders — same return shape as before
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_portfolio_snapshots() -> pd.DataFrame:
    df = _load_table("portfolio_snapshots")
    if df.empty:
        return df
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df["date"]        = df["captured_at"].dt.tz_convert("America/New_York").dt.date
    return df


@st.cache_data
def load_trades() -> pd.DataFrame:
    df = _load_table("trades")
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["date"]       = df["created_at"].dt.tz_convert("America/New_York").dt.date
    df["notional"]   = (df["qty"] * df["price"]).round(2)
    df["side"]       = df["side"].astype(str).str.strip().str.lower()
    return df


@st.cache_data
def load_agent_decisions() -> pd.DataFrame:
    df = _load_table("agent_decision_logs")
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


@st.cache_data
def load_market_snapshots() -> pd.DataFrame:
    df = _load_table("market_snapshots")
    if df.empty:
        return df
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df["date"]        = df["captured_at"].dt.tz_convert("America/New_York").dt.date
    return df


@st.cache_data
def load_sentiment_snapshots() -> pd.DataFrame:
    df = _load_table("sentiment_snapshots")
    if df.empty:
        return df
    df["cycle_ts"]   = pd.to_datetime(df["cycle_ts"],   utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["date"]       = df["cycle_ts"].dt.tz_convert("America/New_York").dt.date
    df["score"]      = pd.to_numeric(df["score"],      errors="coerce").fillna(0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SPY benchmark — snapshot Parquet first, optional live Alpaca fallback
# ─────────────────────────────────────────────────────────────────────────────
def _alpaca_keys() -> tuple[str, str]:
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


@st.cache_data
def load_spy_benchmark(start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """SPY buy-and-hold normalised to ``START_EQUITY``.

    Order of preference:
      1. Parquet snapshot at ``dashboard/data/spy_benchmark.parquet`` (fast, deterministic).
      2. Live fetch from Alpaca if credentials are available (used in dev/live mode).

    Rows are clipped to ``[start_date_str, end_date_str]`` (calendar dates) so the benchmark
    never runs past the portfolio validation window.
    """
    start_d = pd.to_datetime(start_date_str).date()
    end_d = pd.to_datetime(end_date_str).date()
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    df = _load_parquet("spy_benchmark")
    if not df.empty:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start_d) & (df["date"] <= end_d)].sort_values("date")
        if df.empty:
            return df
        p0 = float(df["close"].iloc[0])
        df["equity"] = START_EQUITY * df["close"] / p0
        df["cum_return_pct"] = (df["close"] / p0 - 1) * 100
        return df

    if _source_mode() != "live":
        return pd.DataFrame()

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests   import StockBarsRequest
        from alpaca.data.timeframe  import TimeFrame

        api_key, api_secret = _alpaca_keys()
        if not api_key or not api_secret:
            return pd.DataFrame()
        client = StockHistoricalDataClient(api_key, api_secret)
        req = StockBarsRequest(
            symbol_or_symbols=[BENCHMARK],
            timeframe=TimeFrame.Day,
            start=dt.datetime.fromisoformat(start_date_str),
            end=dt.datetime.fromisoformat(end_date_str),
        )
        bars = client.get_stock_bars(req)
        rows = [{"date": b.timestamp.date(), "close": float(b.close)}
                for b in bars[BENCHMARK]]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
        df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]
        if df.empty:
            return pd.DataFrame()
        p0 = df["close"].iloc[0]
        df["equity"]         = START_EQUITY * df["close"] / p0
        df["cum_return_pct"] = (df["close"] / p0 - 1) * 100
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Computations — unchanged behaviour
# ─────────────────────────────────────────────────────────────────────────────
def compute_daily_equity(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_df.empty:
        return pd.DataFrame(columns=["arm_id", "date", "equity"])
    return (
        portfolio_df.sort_values("captured_at")
        .groupby(["arm_id", "date"])["equity"].last()
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


def _coerce_list(v) -> list:
    """Normalise a JSON column value (str / list / ndarray / None) → list."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            j = json.loads(v)
            return j if isinstance(j, list) else []
        except Exception:
            return []
    if isinstance(v, list):
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    try:
        return list(v)
    except Exception:
        return []


def compute_block_rate(decisions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for armid, grp in decisions_df.groupby("arm_id"):
        proposed = executed = blocked = 0
        for _, row in grp.iterrows():
            proposed += len(_coerce_list(row.get("orders_proposed")))
            executed += len(_coerce_list(row.get("orders_executed")))
            blocked  += len(_coerce_list(row.get("orders_blocked")))
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
    result = {}
    for arm in ARM_ORDER:
        grp = decisions_df[decisions_df["arm_id"] == arm]
        total_cycles = len(grp)
        active_cycles = sum(
            1 for _, row in grp.iterrows()
            if len(_coerce_list(row.get("orders_executed"))) > 0
        )
        result[arm] = round(active_cycles / total_cycles * 100, 1) if total_cycles > 0 else 0.0
    return result


def apply_daily_returns_x_range(fig: Any, returns_df: pd.DataFrame) -> None:
    """Clamp Plotly x-axis to portfolio trading dates (daily equity / return curves)."""
    if returns_df.empty:
        return
    d = returns_df["date"].dropna()
    if d.empty:
        return
    fig.update_xaxes(range=[str(d.min()), str(d.max())], autorange=False)


def apply_datetime_x_range(fig: Any, ts_low: Any, ts_high: Any) -> None:
    """Clamp Plotly x-axis to explicit timestamps (intraday, cumulative cost, etc.)."""
    if ts_low is None or ts_high is None:
        return
    if pd.isna(ts_low) or pd.isna(ts_high):
        return
    fig.update_xaxes(range=[ts_low, ts_high], autorange=False)


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI helpers (used across pages)
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar_about() -> None:
    """Persistent sidebar block: title, source mode, snapshot date."""
    meta = get_snapshot_meta()
    mode = _source_mode()
    if mode == "snapshot" and meta:
        gen = meta.get("generated_at", "")[:10]
        window = f'{meta.get("window_start", "?")} → {meta.get("window_end", "?")}'
        badge = (
            f"<span style='background:#1c1b19;color:#22d3ee;border:1px solid #262523;"
            f"padding:.15rem .55rem;border-radius:6px;font-size:.7rem;"
            f"font-family:monospace'>FROZEN · {gen}</span>"
        )
    else:
        window = "live (5-min cache)"
        badge = (
            f"<span style='background:#1c1b19;color:#fdab43;border:1px solid #262523;"
            f"padding:.15rem .55rem;border-radius:6px;font-size:.7rem;"
            f"font-family:monospace'>LIVE</span>"
        )
    st.sidebar.markdown(
        f"""
        <div style='font-size:.78rem;color:#7a7974;line-height:1.55'>
          <div style='color:#cdccca;font-weight:600;margin-bottom:.25rem'>About this dashboard</div>
          A 2×2 factorial study of LLM-driven trading agents.
          <span style='color:#22d3ee'>Model</span>
          (Qwen 235B vs GPT-4.1) ×
          <span style='color:#a86fdf'>Architecture</span>
          (Monolithic vs Council).
          <div style='margin-top:.6rem'>{badge}</div>
          <div style='margin-top:.4rem;font-family:monospace;font-size:.7rem;color:#7a7974'>
            Window: {window}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_takeaway(text: str, *, title: str = "Key takeaway") -> None:
    """Bottom-of-page summary card for the jury."""
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#1c1b19,#22211f);
                    border:1px solid #393836;border-left:4px solid #22d3ee;
                    border-radius:10px;padding:1rem 1.2rem;margin-top:1.4rem'>
          <div style='font-size:.7rem;color:#22d3ee;text-transform:uppercase;
                      letter-spacing:.1em;font-weight:600;margin-bottom:.35rem'>{title}</div>
          <div style='color:#cdccca;font-size:.95rem;line-height:1.55'>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
