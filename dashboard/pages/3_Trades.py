"""Trades — execution log, block rate, symbol breakdown."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import (load_trades, load_agent_decisions, compute_block_rate,
                ARM_LABELS, ARM_COLORS, ARM_ORDER)

st.set_page_config(page_title="Trades", page_icon="🔄", layout="wide")
st.markdown("## Trade Execution")
st.caption("All executed and blocked orders across arms.")

trades_df    = load_trades()
decisions_df = load_agent_decisions()
block_df     = compute_block_rate(decisions_df)

st.markdown("### Execution vs. Risk Guard Blocks")
if not block_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Executed", x=block_df["arm_id"],
                         y=block_df["executed"], marker_color="#437a22"))
    fig.add_trace(go.Bar(name="Blocked",  x=block_df["arm_id"],
                         y=block_df["blocked"],  marker_color="#a12c7b"))
    fig.update_layout(
        barmode="group", height=300, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
        font=dict(family="sans-serif", color="#cdccca", size=12),
        xaxis=dict(tickvals=ARM_ORDER,
                   ticktext=[ARM_LABELS[a] for a in ARM_ORDER]),
        yaxis=dict(title="Orders", gridcolor="#262523"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    block_df["block_rate_pct"] = (
        block_df["blocked"] / block_df["proposed"].clip(lower=1) * 100
    ).round(1)
    st.dataframe(
        block_df.rename(columns={
            "arm_id": "Arm", "proposed": "Proposed",
            "executed": "Executed", "blocked": "Blocked",
            "block_rate_pct": "Block Rate (%)"
        }),
        use_container_width=True, hide_index=True
    )

st.markdown("### Trades by Symbol")
if not trades_df.empty:
    sym_counts = trades_df.groupby(["arm_id","symbol"]).size().reset_index(name="count")
    arms_sel   = st.multiselect("Filter arms", ARM_ORDER, default=ARM_ORDER)
    sym_filt   = sym_counts[sym_counts["arm_id"].isin(arms_sel)]
    fig2 = go.Figure()
    for arm in arms_sel:
        grp = sym_filt[sym_filt["arm_id"] == arm]
        fig2.add_trace(go.Bar(name=ARM_LABELS[arm], x=grp["symbol"],
                              y=grp["count"], marker_color=ARM_COLORS[arm]))
    fig2.update_layout(
        barmode="group", height=300, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
        font=dict(family="sans-serif", color="#cdccca", size=12),
        yaxis=dict(title="Trade Count", gridcolor="#262523"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Full Trade Log")
col1, col2, col3 = st.columns(3)
arm_f  = col1.multiselect("Arm",    ARM_ORDER, default=ARM_ORDER)
side_f = col2.multiselect("Side",   ["buy","sell"], default=["buy","sell"])
sym_all= sorted(trades_df["symbol"].unique()) if not trades_df.empty else []
sym_f  = col3.multiselect("Symbol", sym_all, default=sym_all)

filt = trades_df[
    trades_df["arm_id"].isin(arm_f) &
    trades_df["side"].isin(side_f) &
    trades_df["symbol"].isin(sym_f)
].copy()
filt["created_at"] = filt["created_at"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M ET")
st.dataframe(
    filt[["arm_id","symbol","side","qty","price","notional","status","created_at"]]
      .rename(columns={"arm_id":"Arm","symbol":"Symbol","side":"Side",
                        "qty":"Qty","price":"Price","notional":"Notional",
                        "status":"Status","created_at":"Time (ET)"}),
    use_container_width=True, hide_index=True
)