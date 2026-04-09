"""Thesis Export — publication-ready figures for thesis document.

Temporary page: collects all key charts in one place so you can
use the Plotly camera icon (📷) to export each as a PNG.
Delete this file once exports are complete.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from db import (
    load_portfolio_snapshots, load_trades, load_agent_decisions,
    load_market_snapshots, load_sentiment_snapshots,
    load_spy_benchmark,
    compute_daily_equity, compute_returns, compute_sharpe,
    compute_max_drawdown, compute_roi, compute_cost_per_bp,
    compute_action_rate, compute_block_rate,
    ARM_LABELS, ARM_COLORS, ARM_ORDER, PLOT_LAYOUT,
    FILL_COLORS, START_EQUITY,
)

st.set_page_config(page_title="Thesis Export", page_icon="📄", layout="wide")
st.markdown("## 📄 Thesis Figure Export")
st.caption(
    "Publication-ready figures for the thesis document. "
    "Use the Plotly camera icon (top-right of each chart) to download PNGs. "
    "Delete `pages/99_Thesis_Export.py` when done."
)

# ── Load all data ────────────────────────────────────────────────────────
portfolio_df  = load_portfolio_snapshots()
trades_df     = load_trades()
decisions_df  = load_agent_decisions()
market_df     = load_market_snapshots()
sentiment_df  = load_sentiment_snapshots()

daily_eq      = compute_daily_equity(portfolio_df)
returns_df    = compute_returns(daily_eq)
sharpe_d      = compute_sharpe(returns_df)
mdd_d         = compute_max_drawdown(returns_df)
roi_d         = compute_roi(returns_df, decisions_df)
cpbp_d        = compute_cost_per_bp(returns_df, decisions_df)
alpha_d       = compute_action_rate(decisions_df)
block_df      = compute_block_rate(decisions_df)

_AXIS   = dict(gridcolor="#262523")
_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Equity Curves + SPY Benchmark (Cumulative Return %)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — Equity Curves: Cumulative Return (%) with SPY Benchmark")

fig1 = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].sort_values("date")
    if grp.empty:
        continue
    fig1.add_trace(go.Scatter(
        x=grp["date"], y=grp["cum_return_pct"],
        mode="lines+markers",
        name=ARM_LABELS[arm],
        line=dict(color=ARM_COLORS[arm], width=2.5),
        marker=dict(size=5),
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))

all_dates = returns_df["date"].dropna()
if not all_dates.empty:
    spy = load_spy_benchmark(str(all_dates.min()), str(all_dates.max()))
    if not spy.empty:
        fig1.add_trace(go.Scatter(
            x=spy["date"], y=spy["cum_return_pct"],
            mode="lines", name="SPY Buy-&-Hold",
            line=dict(color="#5a5957", width=2, dash="dash"),
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        ))

fig1.add_hline(y=0, line_dash="dot", line_color="#393836",
               annotation_text="0% baseline")
fig1.update_layout(
    **PLOT_LAYOUT,
    height=480, margin=dict(l=10, r=10, t=30, b=10),
    legend=_LEGEND,
    yaxis=dict(title="Cumulative Return (%)", title_font=dict(color="#7a7974"),
               ticksuffix="%", **_AXIS),
    xaxis=dict(title="Date", title_font=dict(color="#7a7974"), **_AXIS),
)
st.plotly_chart(fig1, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Daily Return Distribution (Box Plot)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — Daily Return Distribution")

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
    height=400, margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="Daily Return (%)", title_font=dict(color="#7a7974"), **_AXIS),
    xaxis=dict(**_AXIS),
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — 2×2 Factorial Heatmap (Cumulative Return %)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — 2×2 Factorial Heatmap: Cumulative Return (%)")


def _last_ret(arm):
    g = returns_df[returns_df["arm_id"] == arm]
    return g["cum_return_pct"].iloc[-1] if not g.empty else 0.0


grid_vals = np.array([
    [_last_ret("A"), _last_ret("B")],
    [_last_ret("C"), _last_ret("D")],
])

model_eff = (grid_vals[1, 0] + grid_vals[1, 1]) / 2 - (grid_vals[0, 0] + grid_vals[0, 1]) / 2
arch_eff  = (grid_vals[0, 1] + grid_vals[1, 1]) / 2 - (grid_vals[0, 0] + grid_vals[1, 0]) / 2
interact  = (grid_vals[0, 0] - grid_vals[0, 1] - grid_vals[1, 0] + grid_vals[1, 1]) / 2

row_labels = ["Qwen 235B<br>(OpenRouter)", "GPT-4.1<br>(OpenAI)"]
col_labels = ["Monolithic", "Council"]
arm_grid   = [["A", "B"], ["C", "D"]]

text_vals = []
for i in range(2):
    row_text = []
    for j in range(2):
        v = grid_vals[i, j]
        sign = "+" if v >= 0 else ""
        arm = arm_grid[i][j]
        row_text.append(f"<b>{sign}{v:.3f}%</b><br><span style='font-size:11px'>Arm {arm}</span>")
    text_vals.append(row_text)

fig3 = go.Figure(data=go.Heatmap(
    z=grid_vals,
    x=col_labels,
    y=row_labels,
    text=text_vals,
    texttemplate="%{text}",
    textfont=dict(size=18, color="white"),
    colorscale=[
        [0.0, "#922b21"],
        [0.35, "#4a4947"],
        [1.0, "#1e8449"],
    ],
    zmin=-1.0, zmax=2.0,
    showscale=False,
    hovertemplate="Model: %{y}<br>Architecture: %{x}<br>Return: %{z:.3f}%<extra></extra>",
    xgap=4, ygap=4,
))

# Marginal annotations
mono_mean    = (grid_vals[0, 0] + grid_vals[1, 0]) / 2
council_mean = (grid_vals[0, 1] + grid_vals[1, 1]) / 2
qwen_mean    = (grid_vals[0, 0] + grid_vals[0, 1]) / 2
gpt_mean     = (grid_vals[1, 0] + grid_vals[1, 1]) / 2

for j, (label, val) in enumerate([(col_labels[0], mono_mean), (col_labels[1], council_mean)]):
    sign = "+" if val >= 0 else ""
    fig3.add_annotation(
        x=label, y=-0.22, yref="paper",
        text=f"<b>μ = {sign}{val:.3f}%</b>",
        showarrow=False, font=dict(size=12, color="#999999"),
    )

for i, (label, val) in enumerate([(row_labels[0], qwen_mean), (row_labels[1], gpt_mean)]):
    sign = "+" if val >= 0 else ""
    fig3.add_annotation(
        x=1.15, xref="paper", y=label,
        text=f"<b>μ = {sign}{val:.3f}%</b>",
        showarrow=False, font=dict(size=12, color="#999999"),
    )

# Effect annotations below chart
effects_text = (
    f"<span style='color:#fdab43'><b>Model Effect: +{model_eff:.3f}%</b></span>"
    f"  ·  "
    f"<span style='color:#818cf8'><b>Architecture Effect: +{arch_eff:.3f}%</b></span>"
    f"  ·  "
    f"<span style='color:#a86fdf'><b>Interaction: +{interact:.3f}%</b></span>"
)

fig3.update_layout(
    **PLOT_LAYOUT,
    height=420, margin=dict(l=10, r=80, t=30, b=70),
    xaxis=dict(side="top", tickfont=dict(size=15, color="#e0dfdd"),
               gridcolor="rgba(0,0,0,0)"),
    yaxis=dict(tickfont=dict(size=13, color="#e0dfdd"), autorange="reversed",
               gridcolor="rgba(0,0,0,0)"),
)

fig3.add_annotation(
    x=0.5, y=-0.18, xref="paper", yref="paper",
    text=effects_text,
    showarrow=False, font=dict(size=13), align="center",
)

st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Drawdown Over Time
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — Drawdown Over Time")

fig4 = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].sort_values("date")
    if grp.empty:
        continue
    eq = grp["equity"]
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    fig4.add_trace(go.Scatter(
        x=grp["date"], y=dd,
        mode="lines", fill="tozeroy",
        name=ARM_LABELS[arm],
        line=dict(color=ARM_COLORS[arm], width=1.5),
        fillcolor=FILL_COLORS[arm],
    ))
fig4.update_layout(
    **PLOT_LAYOUT,
    height=380, margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="Drawdown (%)", title_font=dict(color="#7a7974"),
               ticksuffix="%", **_AXIS),
    xaxis=dict(title="Date", title_font=dict(color="#7a7974"), **_AXIS),
    legend=_LEGEND,
)
st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — ROI by Arm
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — Return on Intelligence (ROI) by Arm")

roi_vals = [roi_d.get(a, 0) or 0 for a in ARM_ORDER]
fig5 = go.Figure(go.Bar(
    x=[ARM_LABELS[a] for a in ARM_ORDER], y=roi_vals,
    marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
    text=[f"{v:,.1f}×" for v in roi_vals], textposition="outside",
    textfont=dict(color="#cdccca", size=14),
))
fig5.add_hline(y=0, line_dash="dot", line_color="#393836")
fig5.update_layout(
    **PLOT_LAYOUT,
    height=400, margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="ROI (× net return / $ cost)", title_font=dict(color="#7a7974"), **_AXIS),
    xaxis=dict(tickangle=-12, **_AXIS),
)
st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Block Rate & Action Rate (side-by-side bars)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — Block Rate & Action Rate by Arm")

block_rates = []
for arm in ARM_ORDER:
    row = block_df[block_df["arm_id"] == arm]
    if not row.empty:
        p = row["proposed"].values[0]
        b = row["blocked"].values[0]
        block_rates.append(b / p * 100 if p > 0 else 0)
    else:
        block_rates.append(0)
action_rates = [alpha_d.get(a, 0) for a in ARM_ORDER]

fig6 = go.Figure()
fig6.add_trace(go.Bar(
    x=[ARM_LABELS[a] for a in ARM_ORDER], y=block_rates,
    name="Block Rate (%)",
    marker_color="#d163a7",
    text=[f"{v:.0f}%" for v in block_rates], textposition="outside",
    textfont=dict(color="#cdccca"),
))
fig6.add_trace(go.Bar(
    x=[ARM_LABELS[a] for a in ARM_ORDER], y=action_rates,
    name="Action Rate α (%)",
    marker_color="#6daa45",
    text=[f"{v:.0f}%" for v in action_rates], textposition="outside",
    textfont=dict(color="#cdccca"),
))
fig6.update_layout(
    **PLOT_LAYOUT,
    barmode="group",
    height=400, margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="Rate (%)", title_font=dict(color="#7a7974"),
               range=[0, 110], ticksuffix="%", **_AXIS),
    xaxis=dict(tickangle=-12, **_AXIS),
    legend=_LEGEND,
)
st.plotly_chart(fig6, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — LLM Cost Comparison
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — LLM Inference Cost by Arm")

costs = []
for arm in ARM_ORDER:
    ad = decisions_df[decisions_df["arm_id"] == arm]
    costs.append(ad["llm_cost_usd"].sum())

fig7 = go.Figure(go.Bar(
    x=[ARM_LABELS[a] for a in ARM_ORDER], y=costs,
    marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
    text=[f"${v:.4f}" for v in costs], textposition="outside",
    textfont=dict(color="#cdccca", size=13),
))
fig7.update_layout(
    **PLOT_LAYOUT,
    height=400, margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="Total LLM Cost ($)", tickprefix="$",
               title_font=dict(color="#7a7974"), **_AXIS),
    xaxis=dict(tickangle=-12, **_AXIS),
)
st.plotly_chart(fig7, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Sentiment Correlation (bar chart)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — Sentiment → Next-Day Return Correlation (Pearson r)")

if not sentiment_df.empty and not market_df.empty:
    px_d = (market_df.sort_values("captured_at")
            .groupby(["date", "symbol"])["mid"].last()
            .reset_index().sort_values(["symbol", "date"]))
    px_d["next_mid"] = px_d.groupby("symbol")["mid"].shift(-1)
    px_d["next_ret"] = (px_d["next_mid"] / px_d["mid"] - 1) * 100
    sent_daily = (sentiment_df.groupby(["date", "symbol"])["score"]
                  .mean().reset_index(name="sent_score"))
    merged = (sent_daily.merge(px_d[["date", "symbol", "next_ret"]],
                               on=["date", "symbol"], how="inner").dropna())

    corr_data = []
    for sym in sorted(merged["symbol"].unique()):
        g = merged[merged["symbol"] == sym]
        n = len(g)
        r = g["sent_score"].corr(g["next_ret"]) if n >= 3 else None
        if r is not None and not np.isnan(r):
            corr_data.append({"symbol": sym, "r": r, "n": n})

    if corr_data:
        cdf = pd.DataFrame(corr_data).sort_values("r")
        colors = ["#6daa45" if v >= 0 else "#d163a7" for v in cdf["r"]]
        fig8 = go.Figure(go.Bar(
            x=cdf["symbol"], y=cdf["r"],
            marker_color=colors,
            text=[f"{v:.3f}" for v in cdf["r"]], textposition="outside",
            textfont=dict(color="#cdccca"),
        ))
        fig8.add_hline(y=0, line_dash="dot", line_color="#393836")
        fig8.update_layout(
            **PLOT_LAYOUT,
            height=400, margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Pearson r", title_font=dict(color="#7a7974"),
                       range=[-1, 1], **_AXIS),
            xaxis=dict(title="Ticker", title_font=dict(color="#7a7974"), **_AXIS),
        )
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Not enough data to compute sentiment correlations.")
else:
    st.info("Sentiment or market data not available.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Factorial Effects Bar Charts (Model & Architecture side-by-side)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — Factorial Effects: Model & Architecture")

model_data = {
    "Qwen 235B\n(A+B avg)": (grid_vals[0, 0] + grid_vals[0, 1]) / 2,
    "GPT-4.1\n(C+D avg)":   (grid_vals[1, 0] + grid_vals[1, 1]) / 2,
}
arch_data = {
    "Monolithic\n(A+C avg)": (grid_vals[0, 0] + grid_vals[1, 0]) / 2,
    "Council\n(B+D avg)":    (grid_vals[0, 1] + grid_vals[1, 1]) / 2,
}

col_l, col_r = st.columns(2)

with col_l:
    fig9a = go.Figure(go.Bar(
        x=list(model_data.keys()), y=list(model_data.values()),
        marker_color=[ARM_COLORS["A"], ARM_COLORS["C"]],
        text=[f"{v:+.3f}%" for v in model_data.values()], textposition="outside",
        textfont=dict(color="#cdccca", size=13),
    ))
    fig9a.add_hline(y=0, line_dash="dot", line_color="#393836")
    fig9a.update_layout(
        **PLOT_LAYOUT,
        height=350, margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Model Effect on Return", font=dict(color="#cdccca", size=14)),
        yaxis=dict(title="Avg Cumulative Return (%)", ticksuffix="%",
                   title_font=dict(color="#7a7974"), **_AXIS),
        xaxis=dict(**_AXIS),
    )
    st.plotly_chart(fig9a, use_container_width=True)

with col_r:
    fig9b = go.Figure(go.Bar(
        x=list(arch_data.keys()), y=list(arch_data.values()),
        marker_color=[ARM_COLORS["C"], ARM_COLORS["D"]],
        text=[f"{v:+.3f}%" for v in arch_data.values()], textposition="outside",
        textfont=dict(color="#cdccca", size=13),
    ))
    fig9b.add_hline(y=0, line_dash="dot", line_color="#393836")
    fig9b.update_layout(
        **PLOT_LAYOUT,
        height=350, margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Architecture Effect on Return", font=dict(color="#cdccca", size=14)),
        yaxis=dict(title="Avg Cumulative Return (%)", ticksuffix="%",
                   title_font=dict(color="#7a7974"), **_AXIS),
        xaxis=dict(**_AXIS),
    )
    st.plotly_chart(fig9b, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Cost per Basis Point
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Figure — Cost per Basis Point of Return")

cpbp_vals = [cpbp_d.get(a, 0) or 0 for a in ARM_ORDER]
fig10 = go.Figure(go.Bar(
    x=[ARM_LABELS[a] for a in ARM_ORDER], y=cpbp_vals,
    marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
    text=[f"${v:.4f}" for v in cpbp_vals], textposition="outside",
    textfont=dict(color="#cdccca", size=13),
))
fig10.update_layout(
    **PLOT_LAYOUT,
    height=400, margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="$ per Basis Point", tickprefix="$",
               title_font=dict(color="#7a7974"), **_AXIS),
    xaxis=dict(tickangle=-12, **_AXIS),
)
st.plotly_chart(fig10, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABLE — Win Rate and Implied Payoff Asymmetry
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Table — Win Rate and Implied Payoff Asymmetry")

payoff_rows = []
for arm in ARM_ORDER:
    arm_ret = returns_df[returns_df["arm_id"] == arm]
    net_return = arm_ret["equity"].iloc[-1] - START_EQUITY if not arm_ret.empty else 0.0

    wins = losses = 0
    for _, row in decisions_df[decisions_df["arm_id"] == arm].iterrows():
        try:
            pm = (row["post_mortem"] if isinstance(row["post_mortem"], dict)
                  else json.loads(row["post_mortem"] or "{}"))
            for _sym, val in (pm or {}).items():
                pnl = val.get("pnl_pct", 0) if isinstance(val, dict) else 0
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
        except Exception:
            pass

    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0.0

    if net_return > 0 and wins > 0:
        implied_avg_win = net_return / wins
        payoff_rows.append({
            "Arm": ARM_LABELS[arm],
            "Total Trades": total,
            "Wins": wins,
            "Losses": losses,
            "Win Rate (%)": f"{win_rate:.1f}",
            "Net Return ($)": f"{net_return:+,.2f}",
            "Implied Avg Win ($)": f"+{implied_avg_win:,.2f}",
        })
    else:
        payoff_rows.append({
            "Arm": ARM_LABELS[arm],
            "Total Trades": total,
            "Wins": wins,
            "Losses": losses,
            "Win Rate (%)": f"{win_rate:.1f}",
            "Net Return ($)": f"{net_return:+,.2f}",
            "Implied Avg Win ($)": "— ¹",
        })

st.dataframe(pd.DataFrame(payoff_rows), use_container_width=True, hide_index=True)
st.caption(
    "¹ Arm A's net return is negative; implied average win cannot be derived "
    "consistently from summary-level data. Average loss per trade cannot be "
    "reliably computed from summary-level data without individual trade-level "
    "P&L records."
)

st.markdown("---")
st.info(
    "**Export instructions**: hover over any chart and click the 📷 camera icon "
    "(top-right toolbar) to download as PNG. For higher resolution, right-click → "
    "'Save image as' or use browser developer tools to capture at 2× scale."
)
