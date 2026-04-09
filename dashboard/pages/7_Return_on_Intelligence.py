"""Return on Intelligence — thesis §3.1.4.2 composite metric.

ROI = Net Return ÷ Total Operational Cost  (Inference + Infrastructure)
Cost/bp = Total Cost ÷ |Cumulative Return in bps|
Action Rate (α) = cycles with ≥1 executed order / total cycles
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import (
    load_portfolio_snapshots, load_trades, load_agent_decisions,
    compute_daily_equity, compute_returns, compute_sharpe,
    compute_max_drawdown, compute_roi, compute_cost_per_bp,
    compute_action_rate, compute_block_rate,
    ARM_LABELS, ARM_COLORS, ARM_ORDER, PLOT_LAYOUT, START_EQUITY,
)

st.set_page_config(page_title="Return on Intelligence", page_icon="🧠", layout="wide")
st.markdown("## Return on Intelligence (ROI)")
st.caption(
    "Composite measure from the thesis (§3.1.4.2): *performance* is only meaningful "
    "when assessed against the *total cost* of producing it. "
    "ROI synthesises both research questions into one business metric."
)

with st.expander("Metric definitions used on this page", expanded=False):
    st.markdown(
        "| Metric | Formula | Interpretation |\n"
        "|---|---|---|\n"
        "| **ROI** (Return on Intelligence) | Net Return ($) ÷ Total LLM Cost ($) | "
        "How many dollars of return each dollar of inference cost produced. "
        "ROI = 100× means $100 gained per $1 spent. Negative means the arm lost money. |\n"
        "| **Basis point (bp)** | 1 bp = 0.01% of return | "
        "Standard financial unit for small return differences. "
        "A cumulative return of +1.476% = 147.6 bp. |\n"
        "| **$/bp** (Cost per Basis Point) | Total LLM Cost ($) ÷ |Cumulative Return in bp| | "
        "How much inference cost it took to produce each basis point of absolute return. "
        "Lower = more cost-efficient. Unstable when return is near zero. |\n"
        "| **α** (Action Rate) | Cycles with ≥1 executed order ÷ Total cycles × 100 | "
        "How often the agent actually traded. "
        "Low α = conservative/selective; high α = frequent trading. |\n"
        "| **Sharpe Ratio** | (mean daily return ÷ std daily return) × √252 | "
        "Risk-adjusted return, annualised. Higher = better return per unit of risk. |"
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
roi_d        = compute_roi(returns_df, decisions_df)
cpbp_d       = compute_cost_per_bp(returns_df, decisions_df)
alpha_d      = compute_action_rate(decisions_df)
block_df     = compute_block_rate(decisions_df)

_AXIS   = dict(gridcolor="#262523")
_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")

# ── 2×2 Factorial ROI Table ───────────────────────────────────────────────
st.markdown("### 2×2 Factorial — ROI Summary")
st.caption(
    "Each cell shows the key metrics for one arm. "
    "Read **left→right** to isolate architecture effect; **top→bottom** to isolate model effect."
)

header_cols = st.columns([1, 2, 2])
header_cols[0].markdown("")
header_cols[1].markdown(
    "<div style='text-align:center;font-size:.75rem;color:#7a7974;"
    "text-transform:uppercase;letter-spacing:.08em'>Monolithic</div>",
    unsafe_allow_html=True,
)
header_cols[2].markdown(
    "<div style='text-align:center;font-size:.75rem;color:#7a7974;"
    "text-transform:uppercase;letter-spacing:.08em'>Multi-Agent (Council)</div>",
    unsafe_allow_html=True,
)

for model_label, arms in [("Qwen 235B (A, B)", ["A", "B"]), ("GPT-4.1 (C, D)", ["C", "D"])]:
    row_cols = st.columns([1, 2, 2])
    row_cols[0].markdown(
        f"<div style='padding-top:.8rem;font-size:.75rem;color:#7a7974;"
        f"text-transform:uppercase;letter-spacing:.06em'>{model_label}</div>",
        unsafe_allow_html=True,
    )
    for idx, arm in enumerate(arms):
        arm_r = returns_df[returns_df["arm_id"] == arm]
        arm_d = decisions_df[decisions_df["arm_id"] == arm]
        eq  = arm_r["equity"].iloc[-1] if not arm_r.empty else START_EQUITY
        cr  = arm_r["cum_return_pct"].iloc[-1] if not arm_r.empty else 0.0
        cost = arm_d["llm_cost_usd"].sum()
        roi  = roi_d.get(arm)
        cpbp = cpbp_d.get(arm)
        sh   = sharpe_d.get(arm)
        act  = alpha_d.get(arm, 0)
        c    = ARM_COLORS[arm]
        sign = "+" if cr >= 0 else ""
        pc   = "#6daa45" if cr >= 0 else "#d163a7"

        with row_cols[idx + 1]:
            st.markdown(f"""
            <div style="background:#1c1b19;border:1px solid #262523;border-radius:10px;
                        padding:1rem 1.2rem;margin-bottom:.6rem">
              <div style="font-size:.68rem;color:{c};text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:.3rem">{ARM_LABELS[arm]}</div>
              <div style="font-size:1.3rem;font-weight:700;color:#cdccca">
                ROI: {f"{roi:,.1f}×" if roi is not None else "—"}</div>
              <div style="font-size:.82rem;color:{pc};font-weight:500">{sign}{cr:.3f}%
                <span style="color:#5a5957;font-weight:400"> · </span>
                ${cost:.4f} cost</div>
              <div style="display:flex;gap:.5rem;margin-top:.4rem;flex-wrap:wrap">
                <span style="background:#22211f;border-radius:5px;padding:.1rem .45rem;
                             font-size:.68rem;color:#7a7974">
                  Cost/bp: {f"${cpbp:.4f}" if cpbp else "—"}</span>
                <span style="background:#22211f;border-radius:5px;padding:.1rem .45rem;
                             font-size:.68rem;color:#7a7974">
                  Sharpe: {f"{sh:.2f}" if sh is not None else "—"}</span>
                <span style="background:#22211f;border-radius:5px;padding:.1rem .45rem;
                             font-size:.68rem;color:#7a7974">
                  α: {act:.0f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── ROI Bar Chart ─────────────────────────────────────────────────────────
st.markdown("### ROI by Arm")
st.caption("Net return per dollar of LLM inference cost. Higher = more efficient use of intelligence spend.")
with st.expander("How to interpret ROI multipliers"):
    st.markdown(
        "- **ROI = 10.0×** means the arm generated about **$10 net return per $1 of LLM cost**.\n"
        "- **ROI = 1.0×** is breakeven on intelligence spend (before non-LLM operating costs).\n"
        "- **ROI between 0 and 1** means positive return, but weak cost-efficiency.\n"
        "- **ROI < 0** means the arm lost money while still consuming LLM cost.\n"
        "- Compare ROI together with **Sharpe**, **MDD**, and **$/bp** to avoid overvaluing one metric."
    )
roi_vals = [roi_d.get(a, 0) or 0 for a in ARM_ORDER]
fig_roi = go.Figure(go.Bar(
    x=[ARM_LABELS[a] for a in ARM_ORDER], y=roi_vals,
    marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
    text=[f"{v:,.1f}×" for v in roi_vals], textposition="outside",
    textfont=dict(color="#cdccca"),
))
fig_roi.add_hline(y=0, line_dash="dot", line_color="#393836")
fig_roi.update_layout(
    **PLOT_LAYOUT, height=320, margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title="ROI (× return / $ cost)", title_font=dict(color="#7a7974"), **_AXIS),
    xaxis=dict(tickangle=-12, **_AXIS),
)
st.plotly_chart(fig_roi, use_container_width=True)

# ── Cost per Basis Point ──────────────────────────────────────────────────
st.markdown("### Cost per Basis Point of Return")
st.caption(
    "A **basis point (bp)** is 0.01% of return — the standard financial unit for measuring "
    "small performance differences. This chart shows how many dollars of LLM inference cost "
    "it took to produce each basis point of absolute return. **Lower = more cost-efficient.**"
)
with st.expander("How to read $/bp"):
    st.markdown(
        "- **Formula**: $/bp = Total LLM Cost ÷ |Cumulative Return in basis points|\n"
        "- **Example**: If an arm returned +1.476% (= 147.6 bp) and cost $7.43, then $/bp = $7.43 ÷ 147.6 = **$0.0503/bp**.\n"
        "- An arm with $/bp = $0.02 is more cost-efficient than one with $/bp = $0.12 — "
        "it produces the same 1 bp of return for less inference spend.\n"
        "- **Caveat**: this metric becomes unstable when cumulative return is very close to zero, "
        "because the denominator shrinks toward zero."
    )
cpbp_vals = [cpbp_d.get(a, 0) or 0 for a in ARM_ORDER]
fig_cpbp = go.Figure(go.Bar(
    x=[ARM_LABELS[a] for a in ARM_ORDER], y=cpbp_vals,
    marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
    text=[f"${v:.4f}" for v in cpbp_vals], textposition="outside",
    textfont=dict(color="#cdccca"),
))
fig_cpbp.update_layout(
    **PLOT_LAYOUT, height=320, margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title="$ per Basis Point", tickprefix="$", title_font=dict(color="#7a7974"), **_AXIS),
    xaxis=dict(tickangle=-12, **_AXIS),
)
st.plotly_chart(fig_cpbp, use_container_width=True)

# ── Action Rate (α) ──────────────────────────────────────────────────────
st.markdown("### Action Rate (α)")
st.caption(
    "Percentage of OODA cycles that produced ≥1 executed order. "
    "Low α → conservative/selective; high α → frequent trading."
)
col1, col2 = st.columns(2)
alpha_vals = [alpha_d.get(a, 0) for a in ARM_ORDER]
with col1:
    fig_alpha = go.Figure(go.Bar(
        x=[ARM_LABELS[a] for a in ARM_ORDER], y=alpha_vals,
        marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
        text=[f"{v:.0f}%" for v in alpha_vals], textposition="outside",
        textfont=dict(color="#cdccca"),
    ))
    fig_alpha.update_layout(
        **PLOT_LAYOUT, height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Action Rate (%)", range=[0, max(alpha_vals or [10]) * 1.3],
                   title_font=dict(color="#7a7974"), **_AXIS),
        xaxis=dict(tickangle=-12, **_AXIS),
    )
    st.plotly_chart(fig_alpha, use_container_width=True)

with col2:
    if not block_df.empty:
        block_df_m = block_df.set_index("arm_id")
        summary = []
        for arm in ARM_ORDER:
            arm_d = decisions_df[decisions_df["arm_id"] == arm]
            blk = block_df_m.loc[arm] if arm in block_df_m.index else {}
            summary.append({
                "Arm":           ARM_LABELS[arm],
                "Total Cycles":  len(arm_d),
                "Active Cycles": int(len(arm_d) * alpha_d.get(arm, 0) / 100),
                "α (%)":         alpha_d.get(arm, 0),
                "Proposed":      int(blk.get("proposed", 0)),
                "Executed":      int(blk.get("executed", 0)),
                "Blocked":       int(blk.get("blocked", 0)),
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ── Factorial Decomposition of ROI ────────────────────────────────────────
st.markdown("### Factorial Decomposition")

def _roi(arm):
    return roi_d.get(arm) or 0

col_a, col_b, col_c = st.columns(3)
for col, label, val, note in [
    (col_a, "Model Effect on ROI (GPT − Qwen)",
     (_roi("C") + _roi("D")) / 2 - (_roi("A") + _roi("B")) / 2,
     "Positive → GPT yields more return per $ of cost"),
    (col_b, "Architecture Effect on ROI (Council − Mono)",
     (_roi("B") + _roi("D")) / 2 - (_roi("A") + _roi("C")) / 2,
     "Positive → council is more cost-efficient"),
    (col_c, "Interaction (A−B−C+D) / 2",
     (_roi("A") - _roi("B") - _roi("C") + _roi("D")) / 2,
     "Positive → GPT benefits more from Council than Qwen does"),
]:
    sgn = "+" if val >= 0 else ""
    pc  = "#6daa45" if val >= 0 else "#d163a7"
    with col:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;border-radius:10px;
                    padding:1rem 1.2rem;text-align:center">
          <div style="font-size:.7rem;color:#7a7974;margin-bottom:.3rem">{label}</div>
          <div style="font-size:1.6rem;font-weight:700;color:{pc}">{sgn}{val:,.1f}×</div>
          <div style="font-size:.7rem;color:#5a5957;margin-top:.3rem">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Full comparison table ─────────────────────────────────────────────────
st.markdown("### Complete Arm Comparison")
rows = []
for arm in ARM_ORDER:
    arm_r = returns_df[returns_df["arm_id"] == arm]
    arm_d = decisions_df[decisions_df["arm_id"] == arm]
    arm_t = trades_df[trades_df["arm_id"] == arm]
    eq  = arm_r["equity"].iloc[-1] if not arm_r.empty else START_EQUITY
    cr  = arm_r["cum_return_pct"].iloc[-1] if not arm_r.empty else 0.0
    rows.append({
        "Arm":              ARM_LABELS[arm],
        "Return (%)":       f"{cr:+.3f}",
        "Sharpe":           f"{sharpe_d.get(arm):.3f}" if sharpe_d.get(arm) is not None else "—",
        "MDD (%)":          f"{mdd_d.get(arm, 0):.2f}",
        "LLM Cost ($)":     f"${arm_d['llm_cost_usd'].sum():.4f}",
        "ROI (×)":          f"{roi_d.get(arm):,.1f}" if roi_d.get(arm) is not None else "—",
        "Cost/bp ($)":      f"${cpbp_d.get(arm):.4f}" if cpbp_d.get(arm) else "—",
        "α (%)":            f"{alpha_d.get(arm, 0):.0f}",
        "Trades":           len(arm_t),
        "Cycles":           len(arm_d),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
