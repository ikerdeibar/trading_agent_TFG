"""LLM Cost & Latency — cost per cycle, token usage, ROI analysis."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import (load_agent_decisions, load_trades, ARM_LABELS, ARM_COLORS, ARM_ORDER)

st.set_page_config(page_title="LLM Cost", page_icon="💰", layout="wide")
st.markdown("## LLM Cost & Token Usage")

decisions_df = load_agent_decisions()
trades_df    = load_trades()

if decisions_df.empty:
    st.warning("No decision data found.")
    st.stop()

ab = decisions_df[decisions_df["arm_id"].isin(["A", "B"])]
cd = decisions_df[decisions_df["arm_id"].isin(["C", "D"])]
st.caption(
    f"Logged per cycle at inference (internal $/1M estimates — invoices may differ slightly). "
    f"**OpenRouter (arms A+B, Qwen):** ${ab['llm_cost_usd'].sum():.4f}, {int(ab['llm_tokens_used'].sum()):,} tokens. "
    f"**OpenAI (arms C+D, GPT-4.1):** ${cd['llm_cost_usd'].sum():.4f}, {int(cd['llm_tokens_used'].sum()):,} tokens."
)

# ── KPI cards ─────────────────────────────────────────────────────────────
cols = st.columns(4)
for i, arm in enumerate(ARM_ORDER):
    grp        = decisions_df[decisions_df["arm_id"] == arm]
    total_cost = grp["llm_cost_usd"].sum()
    avg_cost   = grp["llm_cost_usd"].mean()
    total_tok  = int(grp["llm_tokens_used"].sum())
    with cols[i]:
        st.metric(ARM_LABELS[arm], f"${total_cost:.4f}",
                  f"${avg_cost:.5f}/cycle")
        st.caption(f"Total tokens: {total_tok:,}")

st.markdown("---")

# ── Cumulative cost over time ──────────────────────────────────────────────
st.markdown("### Cumulative LLM Cost Over Time")
fig = go.Figure()
for arm in ARM_ORDER:
    grp = decisions_df[decisions_df["arm_id"] == arm].sort_values("cycle_ts").copy()
    if grp.empty:
        continue
    grp["cum_cost"] = grp["llm_cost_usd"].cumsum()
    fig.add_trace(go.Scatter(
        x=grp["cycle_ts"], y=grp["cum_cost"],
        mode="lines+markers", name=ARM_LABELS[arm],
        line=dict(color=ARM_COLORS[arm], width=2.5),
        marker=dict(size=4),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:.4f}<extra></extra>",
    ))
fig.update_layout(
    height=320, margin=dict(l=0,r=0,t=10,b=0),
    plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
    font=dict(family="sans-serif", color="#cdccca", size=12),
    yaxis=dict(tickprefix="$", gridcolor="#262523"),
    xaxis=dict(gridcolor="#262523"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Avg cost per cycle & avg tokens per cycle ──────────────────────────────
st.markdown("### Avg Cost & Tokens per Cycle")
col1, col2 = st.columns(2)
summary = []
for arm in ARM_ORDER:
    grp = decisions_df[decisions_df["arm_id"] == arm]
    summary.append({
        "arm": arm, "label": ARM_LABELS[arm],
        "avg_cost":   grp["llm_cost_usd"].mean(),
        "avg_tokens": grp["llm_tokens_used"].mean(),
    })
sdf    = pd.DataFrame(summary)
labels = sdf["label"].tolist()
colors = [ARM_COLORS[a] for a in sdf["arm"]]

with col1:
    f = go.Figure(go.Bar(
        x=labels, y=sdf["avg_cost"],
        marker_color=colors,
        text=[f"${v:.5f}" for v in sdf["avg_cost"]], textposition="outside",
    ))
    f.update_layout(title="Avg Cost / Cycle (USD)", height=300,
                    margin=dict(l=0,r=0,t=40,b=0),
                    plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
                    font=dict(family="sans-serif", color="#cdccca", size=11),
                    yaxis=dict(tickprefix="$", gridcolor="#262523"),
                    showlegend=False, xaxis=dict(tickangle=-20))
    st.plotly_chart(f, use_container_width=True)

with col2:
    f2 = go.Figure(go.Bar(
        x=labels, y=sdf["avg_tokens"],
        marker_color=colors,
        text=[f"{int(v):,}" for v in sdf["avg_tokens"]], textposition="outside",
    ))
    f2.update_layout(title="Avg Tokens / Cycle", height=300,
                     margin=dict(l=0,r=0,t=40,b=0),
                     plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
                     font=dict(family="sans-serif", color="#cdccca", size=11),
                     yaxis=dict(title="Tokens", gridcolor="#262523"),
                     showlegend=False, xaxis=dict(tickangle=-20))
    st.plotly_chart(f2, use_container_width=True)

# ── Cost efficiency: cost per trade ───────────────────────────────────────
st.markdown("### Cost Efficiency: $/Trade Executed")
eff_rows = []
for arm in ARM_ORDER:
    grp_d = decisions_df[decisions_df["arm_id"] == arm]
    grp_t = trades_df[trades_df["arm_id"] == arm]
    total    = grp_d["llm_cost_usd"].sum()
    n_tr     = len(grp_t)
    cpt      = total / n_tr if n_tr > 0 else None
    avg_cost = grp_d["llm_cost_usd"].mean()
    avg_tok  = grp_d["llm_tokens_used"].mean()
    eff_rows.append({
        "Arm":                ARM_LABELS[arm],
        "Total LLM Cost":     f"${total:.4f}",
        "Trades Executed":    n_tr,
        "Cost / Trade":       f"${cpt:.5f}" if cpt else "—",
        "Avg Cost/Cycle ($)": f"${avg_cost:.6f}",
        "Avg Tokens/Cycle":   f"{int(avg_tok):,}",
    })
st.dataframe(pd.DataFrame(eff_rows), use_container_width=True, hide_index=True)
st.info(
    "Cost/trade is your primary ROI metric for the thesis. "
    "Monolithic arms (A, C) should use fewer tokens per cycle than council arms (B, D). "
    "Each live runner must set **ARM_ID** to A–D so rows land in the right bucket (default in code is `C` if unset)."
)