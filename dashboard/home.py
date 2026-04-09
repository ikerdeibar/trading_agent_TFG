"""Home — Overview dashboard for the TFG AI Trading Agent experiment."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from db import (load_portfolio_snapshots, load_trades, load_agent_decisions,
                compute_daily_equity, compute_returns, compute_sharpe,
                compute_max_drawdown, compute_roi, load_spy_benchmark,
                ARM_LABELS, ARM_COLORS, ARM_ORDER, FILL_COLORS, PLOT_LAYOUT,
                START_EQUITY)

st.set_page_config(
    page_title="AI Trading Agent · TFG Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("## 🤖 AI Trading Agent — Experiment Dashboard")
st.caption("2×2 factorial design: **Model** (GPT-4.1 vs Qwen) × **Architecture** (Monolithic vs Council). "
           "Paper trading on NYSE · Validation window: Mar–Apr 2026 · $100k start per arm.")

with st.spinner("Loading data…"):
    portfolio_df = load_portfolio_snapshots()
    trades_df    = load_trades()
    decisions_df = load_agent_decisions()

if portfolio_df.empty:
    st.error("No portfolio data found. Check your secrets.toml DB credentials.")
    st.stop()

daily_eq   = compute_daily_equity(portfolio_df)
returns_df = compute_returns(daily_eq)
sharpe_d   = compute_sharpe(returns_df)
mdd_d      = compute_max_drawdown(returns_df)
roi_d      = compute_roi(returns_df, decisions_df)

st.markdown("### Arm Performance Summary")
kpi_cols = st.columns(4)
for i, arm in enumerate(ARM_ORDER):
    arm_r = returns_df[returns_df["arm_id"] == arm]
    arm_t = trades_df[trades_df["arm_id"] == arm]
    arm_d = decisions_df[decisions_df["arm_id"] == arm]
    eq_now  = arm_r["equity"].iloc[-1]        if not arm_r.empty else 100_000.0
    cum_ret = arm_r["cum_return_pct"].iloc[-1] if not arm_r.empty else 0.0
    sharpe  = sharpe_d.get(arm)
    mdd     = mdd_d.get(arm, 0.0)
    roi     = roi_d.get(arm)
    sign    = "+" if cum_ret >= 0 else ""
    pc      = "#6daa45" if cum_ret >= 0 else "#d163a7"
    c       = ARM_COLORS[arm]
    with kpi_cols[i]:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;border-radius:12px;
                    padding:1.2rem 1.4rem;box-shadow:0 1px 4px rgba(0,0,0,.3)">
          <div style="font-size:.68rem;color:#7a7974;text-transform:uppercase;
                      letter-spacing:.09em;margin-bottom:.4rem">{ARM_LABELS[arm]}</div>
          <div style="font-size:1.6rem;font-weight:700;font-variant-numeric:tabular-nums;
                      color:#cdccca">${eq_now:,.0f}</div>
          <div style="font-size:.9rem;font-weight:600;color:{pc}">{sign}{cum_ret:.3f}%</div>
          <div style="display:flex;gap:.6rem;margin-top:.5rem;flex-wrap:wrap">
            <span style="background:#22211f;border-radius:6px;padding:.1rem .5rem;
                         font-size:.7rem;color:{c}">
              Sharpe: {f"{sharpe:.2f}" if sharpe is not None else "—"}
            </span>
            <span style="background:#22211f;border-radius:6px;padding:.1rem .5rem;
                         font-size:.7rem;color:#7a7974">
              MDD: {mdd:.1f}%
            </span>
            <span style="background:#22211f;border-radius:6px;padding:.1rem .5rem;
                         font-size:.7rem;color:{c}">
              ROI: {f"{roi:,.1f}×" if roi is not None else "—"}
            </span>
          </div>
          <div style="font-size:.7rem;color:#5a5957;margin-top:.4rem">
            {len(arm_d)} cycles · {len(arm_t)} trades · ${arm_d['cost_corrected'].sum():.3f} LLM
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Equity Curves")
col_a, _ = st.columns([3,1])
with col_a:
    view    = st.radio("", ["Absolute ($)", "Cumulative Return (%)"], horizontal=True, label_visibility="collapsed")
use_pct = "Return" in view

fig = go.Figure()
for arm in ARM_ORDER:
    grp = returns_df[returns_df["arm_id"] == arm].sort_values("date")
    if grp.empty:
        continue
    y = grp["cum_return_pct"] if use_pct else grp["equity"]
    fig.add_trace(go.Scatter(
        x=grp["date"].astype(str), y=y,
        mode="lines+markers", name=ARM_LABELS[arm],
        line=dict(color=ARM_COLORS[arm], width=2.5), marker=dict(size=5),
        hovertemplate="%{x}<br>" + ("%{y:.2f}%" if use_pct else "$%{y:,.0f}") + "<extra></extra>",
    ))
# SPY buy-and-hold benchmark (§3.1.2)
all_dates = returns_df["date"].dropna()
if not all_dates.empty:
    spy = load_spy_benchmark(str(all_dates.min()), str(all_dates.max()))
    if not spy.empty:
        y_spy = spy["cum_return_pct"] if use_pct else spy["equity"]
        fig.add_trace(go.Scatter(
            x=spy["date"].astype(str), y=y_spy,
            mode="lines", name="SPY Buy-&-Hold",
            line=dict(color="#5a5957", width=2, dash="dash"),
            hovertemplate="%{x}<br>" + ("%{y:.2f}%" if use_pct else "$%{y:,.0f}") + "<extra></extra>",
        ))
fig.add_hline(y=0 if use_pct else START_EQUITY, line_dash="dot", line_color="#393836",
              annotation_text="$100k start" if not use_pct else "0% baseline")
fig.update_layout(
    **PLOT_LAYOUT, height=380, margin=dict(l=0,r=0,t=10,b=0),
    yaxis=dict(gridcolor="#262523", tickprefix="" if use_pct else "$", ticksuffix="%" if use_pct else ""),
    xaxis=dict(gridcolor="#262523"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Detailed Summary")
rows = []
for arm in ARM_ORDER:
    arm_r = returns_df[returns_df["arm_id"] == arm]
    arm_t = trades_df[trades_df["arm_id"] == arm]
    arm_d = decisions_df[decisions_df["arm_id"] == arm]
    eq  = arm_r["equity"].iloc[-1]        if not arm_r.empty else 100_000.0
    cr  = arm_r["cum_return_pct"].iloc[-1] if not arm_r.empty else 0.0
    sh  = sharpe_d.get(arm)
    mdd = mdd_d.get(arm, 0.0)
    rows.append({
        "Arm":              ARM_LABELS[arm],
        "Final Equity":     f"${eq:,.0f}",
        "Return (%)":       f"{cr:+.3f}",
        "Sharpe":           f"{sh:.3f}" if sh is not None else "—",
        "MDD (%)":          f"{mdd:.2f}",
        "Cycles":           len(arm_d),
        "Trades":           len(arm_t),
        "LLM Cost ($)":     f"${arm_d['llm_cost_usd'].sum():.4f}",
        "ROI (×)":          f"{roi_d.get(arm):,.1f}" if roi_d.get(arm) is not None else "—",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("### Factorial Effects")
def last_ret(arm):
    g = returns_df[returns_df["arm_id"] == arm]
    return g["cum_return_pct"].iloc[-1] if not g.empty else 0.0

col1, col2, col3 = st.columns(3)
for col, label, val, note in [
    (col1, "Model Effect (GPT − Qwen)",
     (last_ret("C")+last_ret("D"))/2 - (last_ret("A")+last_ret("B"))/2,
     "Positive → GPT arms (C,D) beat Qwen arms (A,B)"),
    (col2, "Architecture Effect (Council − Mono)",
     (last_ret("B")+last_ret("D"))/2 - (last_ret("A")+last_ret("C"))/2,
     "Positive → Council beats Monolithic"),
    (col3, "Interaction (A−B−C+D) / 2",
     (last_ret("A")-last_ret("B")-last_ret("C")+last_ret("D"))/2,
     "Positive → GPT gains more from Council than Qwen does"),
]:
    sgn = "+" if val >= 0 else ""
    pc  = "#6daa45" if val >= 0 else "#d163a7"
    with col:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;border-radius:10px;
                    padding:1rem 1.2rem;text-align:center">
          <div style="font-size:.7rem;color:#7a7974;margin-bottom:.3rem">{label}</div>
          <div style="font-size:1.8rem;font-weight:700;color:{pc}">{sgn}{val:.3f}%</div>
          <div style="font-size:.7rem;color:#5a5957;margin-top:.3rem">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Navigate using the sidebar → Equity Curves, Performance Metrics, Trades, LLM Cost, Sentiment, Daily Session Explorer.")