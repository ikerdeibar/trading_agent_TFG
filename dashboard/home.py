"""Headline — TL;DR landing page for the AI Trading Agent thesis."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db import (
    load_portfolio_snapshots, load_trades, load_agent_decisions,
    load_spy_benchmark,
    compute_daily_equity, compute_returns, compute_sharpe,
    compute_max_drawdown, compute_roi,
    ARM_LABELS, ARM_SHORT, ARM_COLORS, ARM_ORDER,
    PLOT_LAYOUT, AXIS_STYLE, LEGEND_STYLE,
    COLOR_POS, COLOR_NEG, START_EQUITY,
    render_sidebar_about, render_takeaway, get_snapshot_meta,
    apply_daily_returns_x_range,
)

st.set_page_config(
    page_title="AI Trading Agent · TFG Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Final-year thesis dashboard · live results from 16-session paper trading on Alpaca."},
)
render_sidebar_about()

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='padding:.4rem 0 1rem 0'>
      <div style='font-size:.78rem;color:#22d3ee;text-transform:uppercase;
                  letter-spacing:.18em;font-weight:600'>
        TFG · AI TRADING AGENT
      </div>
      <h1 style='margin:.1rem 0 .25rem 0;font-weight:700;color:#cdccca'>
        Architecture is the lever, not the model.
      </h1>
      <div style='color:#7a7974;font-size:1rem;line-height:1.5;max-width:880px'>
        A 2&times;2 factorial study of LLM-driven trading agents
        (Qwen 235B vs GPT-4.1 &times; Monolithic vs Council),
        live paper-traded on NYSE for 16 sessions starting 16 March 2026
        with $100k per arm.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading frozen data…"):
    portfolio_df = load_portfolio_snapshots()
    trades_df    = load_trades()
    decisions_df = load_agent_decisions()

if portfolio_df.empty:
    st.error("No portfolio data found. Snapshot files appear missing — re-run "
             "`python scripts/snapshot_db.py` and redeploy.")
    st.stop()

daily_eq   = compute_daily_equity(portfolio_df)
returns_df = compute_returns(daily_eq)
sharpe_d   = compute_sharpe(returns_df)
mdd_d      = compute_max_drawdown(returns_df)
roi_d      = compute_roi(returns_df, decisions_df)


def _last_ret(arm: str) -> float:
    g = returns_df[returns_df["arm_id"] == arm]
    return float(g["cum_return_pct"].iloc[-1]) if not g.empty else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-arm KPI cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Final standings")
kpi_cols = st.columns(4)
for i, arm in enumerate(ARM_ORDER):
    arm_r = returns_df[returns_df["arm_id"] == arm]
    arm_t = trades_df[trades_df["arm_id"] == arm]
    arm_d = decisions_df[decisions_df["arm_id"] == arm]
    eq_now  = arm_r["equity"].iloc[-1] if not arm_r.empty else START_EQUITY
    cum_ret = arm_r["cum_return_pct"].iloc[-1] if not arm_r.empty else 0.0
    sharpe  = sharpe_d.get(arm)
    mdd     = mdd_d.get(arm, 0.0)
    roi     = roi_d.get(arm)
    sign    = "+" if cum_ret >= 0 else ""
    pc      = COLOR_POS if cum_ret >= 0 else COLOR_NEG
    c       = ARM_COLORS[arm]
    with kpi_cols[i]:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;border-radius:12px;
                    padding:1.1rem 1.3rem;height:100%">
          <div style="font-size:.66rem;color:{c};text-transform:uppercase;
                      letter-spacing:.1em;font-weight:600;margin-bottom:.45rem">{ARM_LABELS[arm]}</div>
          <div style="font-size:1.55rem;font-weight:700;color:#cdccca;
                      font-variant-numeric:tabular-nums">${eq_now:,.0f}</div>
          <div style="font-size:.95rem;font-weight:600;color:{pc};margin-top:.05rem">
            {sign}{cum_ret:.3f}%
          </div>
          <div style="display:flex;gap:.5rem;margin-top:.6rem;flex-wrap:wrap">
            <span style="background:#22211f;border-radius:6px;padding:.12rem .5rem;
                         font-size:.7rem;color:{c}">Sharpe {f"{sharpe:.2f}" if sharpe is not None else "—"}</span>
            <span style="background:#22211f;border-radius:6px;padding:.12rem .5rem;
                         font-size:.7rem;color:#7a7974">MDD {mdd:.1f}%</span>
            <span style="background:#22211f;border-radius:6px;padding:.12rem .5rem;
                         font-size:.7rem;color:{c}">ROI {f"{roi:,.0f}×" if roi is not None and abs(roi) < 1e4 else "n/a"}</span>
          </div>
          <div style="font-size:.7rem;color:#5a5957;margin-top:.55rem">
            {len(arm_d):,} cycles · {len(arm_t):,} trades
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Equity curves (single chart)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Cumulative return — all four arms vs SPY buy-and-hold")
view = st.radio(
    "View",
    ["Cumulative Return (%)", "Absolute Equity ($)"],
    horizontal=True,
    label_visibility="visible",
)
use_pct = "Return" in view

fig = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].sort_values("date")
    if grp.empty:
        continue
    y = grp["cum_return_pct"] if use_pct else grp["equity"]
    fig.add_trace(go.Scatter(
        x=grp["date"].astype(str), y=y,
        mode="lines+markers", name=ARM_SHORT[arm],
        line=dict(color=ARM_COLORS[arm], width=2.6),
        marker=dict(size=5),
        hovertemplate=(f"<b>{ARM_LABELS[arm]}</b><br>%{{x}}<br>"
                       + ("%{y:+.3f}%" if use_pct else "$%{y:,.0f}")
                       + "<extra></extra>"),
    ))

all_dates = returns_df["date"].dropna()
if not all_dates.empty:
    spy = load_spy_benchmark(str(all_dates.min()), str(all_dates.max()))
    if not spy.empty:
        y_spy = spy["cum_return_pct"] if use_pct else spy["equity"]
        fig.add_trace(go.Scatter(
            x=spy["date"].astype(str), y=y_spy,
            mode="lines", name="SPY Buy-&-Hold",
            line=dict(color="#5a5957", width=2, dash="dash"),
            hovertemplate=("SPY · %{x}<br>"
                           + ("%{y:+.3f}%" if use_pct else "$%{y:,.0f}")
                           + "<extra></extra>"),
        ))

fig.add_hline(y=0 if use_pct else START_EQUITY, line_dash="dot", line_color="#393836")
fig.update_layout(
    **PLOT_LAYOUT, height=440, margin=dict(l=0, r=0, t=20, b=0),
    yaxis=dict(
        tickprefix="" if use_pct else "$",
        ticksuffix="%" if use_pct else "",
        title="Return (%)" if use_pct else "Equity (USD)",
        title_font=dict(color="#7a7974"), **AXIS_STYLE,
    ),
    xaxis=dict(title=None, **AXIS_STYLE),
    legend=LEGEND_STYLE,
)
apply_daily_returns_x_range(fig, returns_df)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Factorial-effect cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Two questions, three numbers")
st.caption(
    "Marginal means from the 2×2 design. *Model effect* averages out architecture; "
    "*architecture effect* averages out model; *interaction* shows whether the two factors compound."
)

model_effect = (_last_ret("C") + _last_ret("D")) / 2 - (_last_ret("A") + _last_ret("B")) / 2
arch_effect  = (_last_ret("B") + _last_ret("D")) / 2 - (_last_ret("A") + _last_ret("C")) / 2
interact     = (_last_ret("A") - _last_ret("B") - _last_ret("C") + _last_ret("D")) / 2

col1, col2, col3 = st.columns(3)
for col, label, val, accent, note in [
    (col1, "Model effect (GPT − Qwen)",          model_effect, "#fdab43",
     "Averages over both architectures"),
    (col2, "Architecture effect (Council − Mono)", arch_effect, "#a86fdf",
     "Averages over both models"),
    (col3, "Interaction (A − B − C + D) / 2",    interact,    "#22d3ee",
     "GPT-4.1 benefits more from Council than Qwen does"),
]:
    sgn = "+" if val >= 0 else ""
    pc  = COLOR_POS if val >= 0 else COLOR_NEG
    with col:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;
                    border-top:3px solid {accent};border-radius:10px;
                    padding:1rem 1.2rem;text-align:center;height:100%">
          <div style="font-size:.7rem;color:#7a7974;margin-bottom:.4rem">{label}</div>
          <div style="font-size:1.95rem;font-weight:700;color:{pc};
                      font-variant-numeric:tabular-nums">{sgn}{val:.3f}%</div>
          <div style="font-size:.72rem;color:#5a5957;margin-top:.35rem">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Detailed table
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Full per-arm summary table"):
    rows = []
    for arm in ARM_ORDER:
        arm_r = returns_df[returns_df["arm_id"] == arm]
        arm_t = trades_df[trades_df["arm_id"] == arm]
        arm_d = decisions_df[decisions_df["arm_id"] == arm]
        eq  = arm_r["equity"].iloc[-1] if not arm_r.empty else START_EQUITY
        cr  = arm_r["cum_return_pct"].iloc[-1] if not arm_r.empty else 0.0
        sh  = sharpe_d.get(arm)
        mdd = mdd_d.get(arm, 0.0)
        rows.append({
            "Arm":          ARM_LABELS[arm],
            "Final Equity": f"${eq:,.0f}",
            "Return (%)":   f"{cr:+.3f}",
            "Sharpe":       f"{sh:.3f}" if sh is not None else "—",
            "MDD (%)":      f"{mdd:.2f}",
            "Cycles":       len(arm_d),
            "Trades":       len(arm_t),
            "LLM cost ($)": f"${arm_d['llm_cost_usd'].sum():.4f}",
            "ROI (×)":      f"{roi_d.get(arm):,.1f}" if roi_d.get(arm) is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# What to read next
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Where to go next")
nav_cols = st.columns(3)
items = [
    ("📈 Performance",       "1_Performance",
     "Equity curves, drawdowns, daily-return distribution, risk metrics."),
    ("🧪 Factorial Results", "2_Factorial_Results",
     "2×2 heatmap, model effect, architecture effect, interaction term."),
    ("🔄 Trades & Risk",     "3_Trades_and_Risk",
     "Executed vs blocked orders, judge layer behaviour, ticker concentration."),
    ("💰 Cost & ROI",        "4_Cost_and_ROI",
     "Token spend, $/trade, $/bp, ROI multipliers, factorial decomposition of cost-efficiency."),
    ("🧠 Sentiment",         "5_Sentiment",
     "FinBERT signal per ticker and its correlation with next-day returns."),
    ("📅 Session Explorer",  "6_Session_Explorer",
     "Drill into any single trading day across all four arms."),
]
for i, (title, page, desc) in enumerate(items):
    with nav_cols[i % 3]:
        st.markdown(
            f"<div style='background:#1c1b19;border:1px solid #262523;border-radius:10px;"
            f"padding:.85rem 1rem;margin-bottom:.6rem;height:96px'>"
            f"<div style='font-weight:600;color:#cdccca;font-size:.95rem;margin-bottom:.2rem'>{title}</div>"
            f"<div style='color:#7a7974;font-size:.78rem;line-height:1.4'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Takeaway + footer
# ─────────────────────────────────────────────────────────────────────────────
render_takeaway(
    "Across 16 NYSE sessions, only the <b>Council architecture</b> + a strong base model "
    f"(<b>Arm D · GPT-4.1 · Council</b>, +{_last_ret('D'):.3f}%) "
    "produced compelling risk-adjusted returns. The architecture effect "
    f"(+{arch_effect:.3f}%) is the dominant lever — comparable in magnitude to the model effect "
    f"(+{model_effect:.3f}%). Use the sidebar to walk through the supporting evidence."
)

# Footer
meta = get_snapshot_meta()
gen  = meta.get("generated_at", "")[:10] if meta else ""
window = (f'{meta.get("window_start", "?")} → {meta.get("window_end", "?")}'
          if meta else "snapshot manifest missing")
st.markdown(
    f"""
    <div style='margin-top:3rem;padding-top:1rem;border-top:1px solid #262523;
                color:#5a5957;font-size:.75rem;line-height:1.6'>
      <div>
        <b style='color:#7a7974'>AI Trading Agent · Final-year thesis (TFG)</b>
        &nbsp;·&nbsp; Iker Sánchez Pereira
        &nbsp;·&nbsp; Universidad Pontificia Comillas — ICADE-ICAI
      </div>
      <div>
        Validation window: {window}
        &nbsp;·&nbsp; Snapshot frozen: {gen or 'n/a'}
        &nbsp;·&nbsp; Data source: Alpaca paper-trading API
        &nbsp;·&nbsp; <a href='https://github.com/ikerdeibar/trading_agent_TFG' style='color:#22d3ee;text-decoration:none'>repository</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
