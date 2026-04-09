"""Performance Metrics — Sharpe, drawdown, win rate, architectural effects."""
from __future__ import annotations

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import (load_portfolio_snapshots, load_trades, load_agent_decisions,
                compute_daily_equity, compute_returns, compute_sharpe,
                compute_max_drawdown, ARM_LABELS, ARM_COLORS, ARM_ORDER,
                PLOT_LAYOUT, FILL_COLORS)

st.set_page_config(page_title="Performance Metrics", page_icon="📊", layout="wide")
st.markdown("## Performance Metrics")
st.caption("Risk-adjusted returns, drawdown, and win rate per arm.")

portfolio_df = load_portfolio_snapshots()
if portfolio_df.empty:
    st.warning("No portfolio data found.")
    st.stop()

trades_df    = load_trades()
decisions_df = load_agent_decisions()
daily_eq     = compute_daily_equity(portfolio_df)
returns_df   = compute_returns(daily_eq)
sharpe_dict  = compute_sharpe(returns_df)
drawdown_dict = compute_max_drawdown(returns_df)

# ── Metrics table ─────────────────────────────────────────────────────────
rows = []
for arm in ARM_ORDER:
    arm_ret    = returns_df[returns_df["arm_id"] == arm]
    arm_trades = trades_df[trades_df["arm_id"] == arm]
    cum_r      = arm_ret["cum_return_pct"].iloc[-1] if not arm_ret.empty else 0.0
    sharpe     = sharpe_dict.get(arm)
    mdd        = drawdown_dict.get(arm, 0.0)
    n_trades   = len(arm_trades)

    wins = 0
    total_pm = 0
    for _, row in decisions_df[decisions_df["arm_id"] == arm].iterrows():
        try:
            pm = row["post_mortem"] if isinstance(row["post_mortem"], dict) else json.loads(row["post_mortem"] or "{}")
            for sym, val in (pm or {}).items():
                pnl = val.get("pnl_pct", val.get("pnl", 0)) if isinstance(val, dict) else 0
                total_pm += 1
                if pnl > 0:
                    wins += 1
        except Exception:
            pass
    win_rate = f"{wins/total_pm*100:.1f}%" if total_pm > 0 else "—"

    rows.append({
        "Arm": ARM_LABELS[arm],
        "Return (%)": round(cum_r, 3),
        "Sharpe Ratio": sharpe if sharpe is not None else None,
        "Max Drawdown (%)": mdd,
        "Trades Executed": n_trades,
        "Win Rate": win_rate,
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

_AXIS = dict(gridcolor="#262523")
_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")

# ── Bar charts: model effect & architecture effect ─────────────────────────
st.markdown("### Factorial Effects: Model & Architecture")
col1, col2 = st.columns(2)

model_data = {
    "Qwen (A+B avg)": (
        (returns_df[returns_df["arm_id"]=="A"]["cum_return_pct"].iloc[-1] if not returns_df[returns_df["arm_id"]=="A"].empty else 0) +
        (returns_df[returns_df["arm_id"]=="B"]["cum_return_pct"].iloc[-1] if not returns_df[returns_df["arm_id"]=="B"].empty else 0)
    ) / 2,
    "GPT-4.1 (C+D avg)": (
        (returns_df[returns_df["arm_id"]=="C"]["cum_return_pct"].iloc[-1] if not returns_df[returns_df["arm_id"]=="C"].empty else 0) +
        (returns_df[returns_df["arm_id"]=="D"]["cum_return_pct"].iloc[-1] if not returns_df[returns_df["arm_id"]=="D"].empty else 0)
    ) / 2,
}
arch_data = {
    "Monolithic (A+C avg)": (
        (returns_df[returns_df["arm_id"]=="A"]["cum_return_pct"].iloc[-1] if not returns_df[returns_df["arm_id"]=="A"].empty else 0) +
        (returns_df[returns_df["arm_id"]=="C"]["cum_return_pct"].iloc[-1] if not returns_df[returns_df["arm_id"]=="C"].empty else 0)
    ) / 2,
    "Council (B+D avg)": (
        (returns_df[returns_df["arm_id"]=="B"]["cum_return_pct"].iloc[-1] if not returns_df[returns_df["arm_id"]=="B"].empty else 0) +
        (returns_df[returns_df["arm_id"]=="D"]["cum_return_pct"].iloc[-1] if not returns_df[returns_df["arm_id"]=="D"].empty else 0)
    ) / 2,
}

with col1:
    fig = go.Figure(go.Bar(
        x=list(model_data.keys()), y=list(model_data.values()),
        marker_color=[ARM_COLORS["A"], ARM_COLORS["C"]],
        text=[f"{v:.2f}%" for v in model_data.values()], textposition="outside",
        textfont=dict(color="#cdccca"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Model Effect on Return", font=dict(color="#cdccca", size=14)),
        height=280, margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(title="Avg Cumulative Return (%)", title_font=dict(color="#7a7974"), **_AXIS),
        xaxis=dict(tickangle=-12, **_AXIS),
        legend=_LEGEND,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure(go.Bar(
        x=list(arch_data.keys()), y=list(arch_data.values()),
        marker_color=[ARM_COLORS["C"], ARM_COLORS["D"]],
        text=[f"{v:.2f}%" for v in arch_data.values()], textposition="outside",
        textfont=dict(color="#cdccca"),
    ))
    fig2.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Architecture Effect on Return", font=dict(color="#cdccca", size=14)),
        height=280, margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(title="Avg Cumulative Return (%)", title_font=dict(color="#7a7974"), **_AXIS),
        xaxis=dict(tickangle=-12, **_AXIS),
        legend=_LEGEND,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Drawdown timeline ──────────────────────────────────────────────────────
st.markdown("### Drawdown Over Time")
st.caption(
    "Drawdown is the % distance from the arm's running peak equity. "
    "0% means at a new high; -3% means equity is 3% below its previous peak."
)
fig3 = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].sort_values("date")
    if grp.empty:
        continue
    eq = grp["equity"]
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    fig3.add_trace(go.Scatter(
        x=grp["date"], y=dd,
        mode="lines", fill="tozeroy",
        name=ARM_LABELS[arm],
        line=dict(color=ARM_COLORS[arm], width=1.5),
        fillcolor=FILL_COLORS[arm],
    ))
fig3.update_layout(
    **PLOT_LAYOUT,
    height=300, margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title="Drawdown (%)", title_font=dict(color="#7a7974"), **_AXIS),
    xaxis=_AXIS,
    legend=_LEGEND,
)
st.plotly_chart(fig3, use_container_width=True)
st.info(
    "Interpretation guide: focus on (1) **depth** of valleys (capital pain), "
    "(2) **duration** until recovery (time under water), and "
    "(3) whether drawdowns are isolated shocks or persistent deterioration."
)
