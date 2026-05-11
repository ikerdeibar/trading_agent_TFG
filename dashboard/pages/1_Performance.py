"""Performance — risk-adjusted returns for all four arms."""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db import (
    load_portfolio_snapshots, load_trades, load_agent_decisions,
    load_spy_benchmark,
    compute_daily_equity, compute_returns, compute_sharpe,
    compute_max_drawdown,
    ARM_LABELS, ARM_SHORT, ARM_COLORS, ARM_ORDER,
    PLOT_LAYOUT, AXIS_STYLE, LEGEND_STYLE, FILL_COLORS,
    START_EQUITY,
    render_sidebar_about, render_takeaway,
    apply_daily_returns_x_range,
)

st.set_page_config(page_title="Performance", page_icon="📈", layout="wide")
render_sidebar_about()

st.markdown("## Performance")
st.caption(
    "Equity curves, drawdowns, and risk-adjusted returns over the 16-session "
    "validation window (2026-03-16 → 2026-04-06). Baseline: $100,000 per arm."
)

portfolio_df = load_portfolio_snapshots()
if portfolio_df.empty:
    st.warning("No portfolio data found.")
    st.stop()

trades_df    = load_trades()
decisions_df = load_agent_decisions()
daily_eq     = compute_daily_equity(portfolio_df)
returns_df   = compute_returns(daily_eq)
sharpe_d     = compute_sharpe(returns_df)
mdd_d        = compute_max_drawdown(returns_df)


# ═══════════════════════════════════════════════════════════════════════════
# 1 — Equity curves
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Equity curves vs SPY")

view = st.radio(
    "View",
    ["Cumulative Return (%)", "Absolute Equity ($)"],
    horizontal=True,
    label_visibility="visible",
    key="perf_view",
)
use_pct = "Return" in view

fig1 = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].sort_values("date")
    if grp.empty:
        continue
    y = grp["cum_return_pct"] if use_pct else grp["equity"]
    fig1.add_trace(go.Scatter(
        x=grp["date"].astype(str), y=y,
        mode="lines+markers", name=ARM_SHORT[arm],
        line=dict(color=ARM_COLORS[arm], width=2.6),
        marker=dict(size=5),
        hovertemplate=("<b>" + ARM_LABELS[arm] + "</b><br>%{x}<br>"
                       + ("%{y:+.3f}%" if use_pct else "$%{y:,.0f}")
                       + "<extra></extra>"),
    ))

all_dates = returns_df["date"].dropna()
if not all_dates.empty:
    spy = load_spy_benchmark(str(all_dates.min()), str(all_dates.max()))
    if not spy.empty:
        y_spy = spy["cum_return_pct"] if use_pct else spy["equity"]
        fig1.add_trace(go.Scatter(
            x=spy["date"].astype(str), y=y_spy,
            mode="lines", name="SPY Buy-&-Hold",
            line=dict(color="#5a5957", width=2, dash="dash"),
        ))

fig1.add_hline(y=0 if use_pct else START_EQUITY, line_dash="dot", line_color="#393836")
fig1.update_layout(
    **PLOT_LAYOUT, height=440, margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(
        tickprefix="" if use_pct else "$",
        ticksuffix="%" if use_pct else "",
        title="Return (%)" if use_pct else "Equity (USD)",
        title_font=dict(color="#7a7974"), **AXIS_STYLE,
    ),
    xaxis=dict(**AXIS_STYLE),
    legend=LEGEND_STYLE,
)
apply_daily_returns_x_range(fig1, returns_df)
st.plotly_chart(fig1, use_container_width=True)

with st.expander("How to read this"):
    st.markdown(
        "- Each colored line is one arm's portfolio value, indexed to $100,000 on the first session.\n"
        "- The dashed grey line is **SPY buy-and-hold** over the same window — the passive benchmark.\n"
        "- **Lines that finish above the dotted baseline made money**; those below lost it.\n"
        "- The shape matters: smooth ascents mean steady alpha, jagged ones mean risk concentration."
    )

st.info(
    "**Insight (§4.2)** — Only **Arm D (GPT-4.1 + Council)** finishes meaningfully above SPY. "
    "Arm A (Qwen + Monolithic) is the only loss-making arm; the joint-shock phase around "
    "30 March hurt every arm but Council arms recovered fastest."
)


# ═══════════════════════════════════════════════════════════════════════════
# 2 — Drawdowns
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Drawdown over time")
st.caption(
    "Drawdown = % distance from each arm's running peak equity. 0% means at a fresh high; "
    "−2% means equity sits 2% below the previous peak. Shallower valleys = less capital pain."
)

fig2 = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].sort_values("date")
    if grp.empty:
        continue
    eq = grp["equity"]
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    fig2.add_trace(go.Scatter(
        x=grp["date"].astype(str), y=dd,
        mode="lines", fill="tozeroy",
        name=ARM_SHORT[arm],
        line=dict(color=ARM_COLORS[arm], width=1.6),
        fillcolor=FILL_COLORS[arm],
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))
fig2.add_hline(y=0, line_dash="dot", line_color="#393836")
fig2.update_layout(
    **PLOT_LAYOUT, height=320, margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title="Drawdown (%)", ticksuffix="%",
               title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(**AXIS_STYLE),
    legend=LEGEND_STYLE,
)
apply_daily_returns_x_range(fig2, returns_df)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("How to read this"):
    st.markdown(
        "- Look at three things: **depth** (how far below zero), **duration** (time under water), "
        "and whether drawdowns are isolated shocks or persistent deterioration.\n"
        "- The bottoms cluster around 30 March 2026 — a market-wide shock, not arm-specific."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3 — Daily return distribution
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Daily return distribution")
st.caption(
    "One box per arm. The box shows the inter-quartile range; the line is the median, "
    "the diamond is the mean (`boxmean`). Whiskers and outliers reveal tail risk."
)

fig3 = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].dropna(subset=["daily_return_pct"])
    if grp.empty:
        continue
    fig3.add_trace(go.Box(
        y=grp["daily_return_pct"],
        name=ARM_SHORT[arm],
        marker_color=ARM_COLORS[arm],
        boxmean=True,
    ))
fig3.add_hline(y=0, line_dash="dot", line_color="#393836")
fig3.update_layout(
    **PLOT_LAYOUT, height=360, margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title="Daily Return (%)", ticksuffix="%",
               title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(**AXIS_STYLE, tickangle=-25),
    showlegend=False,
)
st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4 — Metrics table
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Risk-adjusted summary")
rows = []
for arm in ARM_ORDER:
    arm_ret    = returns_df[returns_df["arm_id"] == arm]
    arm_trades = trades_df[trades_df["arm_id"] == arm]
    cum_r      = arm_ret["cum_return_pct"].iloc[-1] if not arm_ret.empty else 0.0
    sharpe     = sharpe_d.get(arm)
    mdd        = mdd_d.get(arm, 0.0)
    n_trades   = len(arm_trades)

    wins = total_pm = 0
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
        "Arm":              ARM_LABELS[arm],
        "Return (%)":       f"{cum_r:+.3f}",
        "Sharpe":           f"{sharpe:.3f}" if sharpe is not None else "—",
        "Max Drawdown (%)": f"{mdd:.2f}",
        "Trades":           n_trades,
        "Win rate":         win_rate,
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

render_takeaway(
    "<b>Council arms (B, D)</b> dominate on risk-adjusted measures — "
    "Arm D's annualised Sharpe is the only one comfortably above zero. "
    "Maximum drawdowns are tight (&lt;2.5%) across all arms because of the 5% position cap "
    "in the Judge layer (<i>see Methodology</i>), but only Council arms recovered fully from the joint shock."
)
