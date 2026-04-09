"""Equity Curves — cumulative portfolio value per arm over time."""
import streamlit as st
import plotly.graph_objects as go
from db import (load_portfolio_snapshots, compute_daily_equity, compute_returns,
                load_spy_benchmark, ARM_LABELS, ARM_COLORS, ARM_ORDER,
                PLOT_LAYOUT, START_EQUITY)

st.set_page_config(page_title="Equity Curves", page_icon="📈", layout="wide")
st.markdown("## Equity Curves")
st.caption("Portfolio value per arm over the validation period. Baseline: $100,000.")

portfolio_df = load_portfolio_snapshots()
if portfolio_df.empty:
    st.warning("No portfolio data found.")
    st.stop()

daily_eq     = compute_daily_equity(portfolio_df)
returns_df   = compute_returns(daily_eq)

# ── Toggle: show returns % or absolute equity ──────────────────────────────
view = st.radio("View", ["Absolute ($)", "Cumulative Return (%)"], horizontal=True)

fig = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].sort_values("date")
    if grp.empty:
        continue
    y_col = "cum_return_pct" if "Return" in view else "equity"
    prefix = "" if "Return" in view else "$"
    suffix = "%" if "Return" in view else ""
    fig.add_trace(go.Scatter(
        x=grp["date"], y=grp[y_col],
        mode="lines+markers",
        name=ARM_LABELS[arm],
        line=dict(color=ARM_COLORS[arm], width=2.5),
        marker=dict(size=5),
        hovertemplate=f"%{{x}}<br>{prefix}%{{y:.2f}}{suffix}<extra></extra>",
    ))

# SPY buy-and-hold benchmark (§3.1.2)
all_dates = returns_df["date"].dropna()
if not all_dates.empty:
    spy = load_spy_benchmark(str(all_dates.min()), str(all_dates.max()))
    if not spy.empty:
        y_spy = spy["cum_return_pct"] if "Return" in view else spy["equity"]
        fig.add_trace(go.Scatter(
            x=spy["date"], y=y_spy,
            mode="lines", name="SPY Buy-&-Hold",
            line=dict(color="#5a5957", width=2, dash="dash"),
            hovertemplate=f"%{{x}}<br>{'%{y:.2f}%' if 'Return' in view else '$%{y:,.0f}'}<extra></extra>",
        ))

if "Return" in view:
    fig.add_hline(y=0, line_dash="dot", line_color="#393836",
                  annotation_text="0% baseline")
else:
    fig.add_hline(y=START_EQUITY, line_dash="dot", line_color="#393836",
                  annotation_text="$100k baseline")

fig.update_layout(
    **PLOT_LAYOUT,
    height=420, margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="#262523", tickprefix="" if "Return" in view else "$",
               ticksuffix="%" if "Return" in view else ""),
    xaxis=dict(gridcolor="#262523"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Daily return distribution ─────────────────────────────────────────────
st.markdown("### Daily Return Distribution")
fig2 = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].dropna(subset=["daily_return_pct"])
    if grp.empty:
        continue
    fig2.add_trace(go.Box(
        y=grp["daily_return_pct"],
        name=ARM_LABELS[arm],
        marker_color=ARM_COLORS[arm],
        boxmean=True,
    ))
fig2.update_layout(
    **PLOT_LAYOUT,
    height=320, margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title="Daily Return (%)", gridcolor="#262523"),
    xaxis=dict(gridcolor="#262523"),
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)
