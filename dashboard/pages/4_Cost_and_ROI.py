"""Cost & ROI — token spend, $/bp, ROI multipliers, factorial decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db import (
    load_portfolio_snapshots, load_trades, load_agent_decisions,
    compute_daily_equity, compute_returns, compute_sharpe,
    compute_max_drawdown, compute_roi, compute_cost_per_bp,
    compute_action_rate, compute_block_rate,
    ARM_LABELS, ARM_SHORT, ARM_COLORS, ARM_ORDER,
    PLOT_LAYOUT, AXIS_STYLE, LEGEND_STYLE,
    COLOR_POS, COLOR_NEG, START_EQUITY,
    render_sidebar_about, render_takeaway,
    apply_datetime_x_range,
)

st.set_page_config(page_title="Cost & ROI", page_icon="💰", layout="wide")
render_sidebar_about()

st.markdown("## Cost & Return on Intelligence")
st.caption(
    "How much LLM inference each arm consumed, and how productively that spend translated into "
    "portfolio return. ROI = Net Return ($) ÷ LLM Cost ($) — the thesis's headline business metric (§3.1.4.2)."
)

with st.expander("Metric definitions", expanded=False):
    st.markdown(
        "| Metric | Formula | Plain English |\n"
        "|---|---|---|\n"
        "| **ROI** | Net Return ÷ Total LLM Cost | $ of return per $ of inference. ROI = 100× → $100 made per $1 spent. |\n"
        "| **Basis point (bp)** | 1 bp = 0.01% return | Standard small-return unit. +1.476% = 147.6 bp. |\n"
        "| **$/bp** | LLM Cost ÷ \\|Cumulative Return in bp\\| | Cost to produce 1 bp of *absolute* return. Lower = better. |\n"
        "| **α (Action rate)** | Cycles with ≥1 trade ÷ Total cycles | How often the agent actually acts. |\n"
        "| **Sharpe** | (mean dr ÷ std dr) × √252 | Annualised risk-adjusted return. |"
    )

portfolio_df = load_portfolio_snapshots()
if portfolio_df.empty:
    st.warning("No portfolio data found.")
    st.stop()

trades_df    = load_trades()
decisions_df = load_agent_decisions()
returns_df   = compute_returns(compute_daily_equity(portfolio_df))
sharpe_d     = compute_sharpe(returns_df)
roi_d        = compute_roi(returns_df, decisions_df)
cpbp_d       = compute_cost_per_bp(returns_df, decisions_df)
alpha_d      = compute_action_rate(decisions_df)
block_df     = compute_block_rate(decisions_df).set_index("arm_id")


# ═══════════════════════════════════════════════════════════════════════════
# KPI strip
# ═══════════════════════════════════════════════════════════════════════════
ab = decisions_df[decisions_df["arm_id"].isin(["A", "B"])]
cd = decisions_df[decisions_df["arm_id"].isin(["C", "D"])]
st.markdown("### Cost summary")
k = st.columns(4)
k[0].metric("OpenRouter (Qwen, A+B)", f"${ab['llm_cost_usd'].sum():.4f}",
            f"{int(ab['llm_tokens_used'].sum()):,} tokens")
k[1].metric("OpenAI (GPT, C+D)",      f"${cd['llm_cost_usd'].sum():.4f}",
            f"{int(cd['llm_tokens_used'].sum()):,} tokens")
prov_ratio = (cd['llm_cost_usd'].sum() / max(ab['llm_cost_usd'].sum(), 1e-9))
k[2].metric("Provider cost gap (GPT ÷ Qwen)", f"{prov_ratio:.1f}×")
total_cost = decisions_df["llm_cost_usd"].sum()
k[3].metric("Total LLM cost · 16 sessions", f"${total_cost:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 1 — Cumulative cost over time
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Cumulative LLM cost over time")
fig_cum = go.Figure()
for arm in ARM_ORDER:
    grp = decisions_df[decisions_df["arm_id"] == arm].sort_values("cycle_ts").copy()
    if grp.empty:
        continue
    grp["cum_cost"] = grp["llm_cost_usd"].cumsum()
    fig_cum.add_trace(go.Scatter(
        x=grp["cycle_ts"], y=grp["cum_cost"],
        mode="lines+markers", name=ARM_SHORT[arm],
        line=dict(color=ARM_COLORS[arm], width=2.6),
        marker=dict(size=4),
        hovertemplate=f"<b>{ARM_LABELS[arm]}</b><br>%{{x|%Y-%m-%d %H:%M}}<br>$%{{y:.4f}}<extra></extra>",
    ))
fig_cum.update_layout(
    **PLOT_LAYOUT, height=360, margin=dict(l=10, r=10, t=20, b=0),
    yaxis=dict(title="Cumulative cost (USD)", tickprefix="$",
               title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(**AXIS_STYLE),
    legend=LEGEND_STYLE,
)
if not decisions_df.empty:
    apply_datetime_x_range(fig_cum, decisions_df["cycle_ts"].min(), decisions_df["cycle_ts"].max())
st.plotly_chart(fig_cum, use_container_width=True)
with st.expander("How to read this"):
    st.markdown(
        "- Steeper lines = faster cost burn. The slope step-changes when the market opens each session.\n"
        "- Compare lines at the same x-tick: at session 16, **D ≈ $7.43** vs **A ≈ $0.024** — "
        "a ~310× cost spread for the *same* 16-session schedule."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2 — Total cost & token volume bars
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Total cost & token volume per arm")
costs  = [decisions_df[decisions_df["arm_id"] == a]["llm_cost_usd"].sum() for a in ARM_ORDER]
tokens = [decisions_df[decisions_df["arm_id"] == a]["llm_tokens_used"].sum() for a in ARM_ORDER]

c1, c2 = st.columns(2)
with c1:
    fig_c = go.Figure(go.Bar(
        x=[ARM_SHORT[a] for a in ARM_ORDER], y=costs,
        marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
        text=[f"${v:.4f}" for v in costs], textposition="outside",
        textfont=dict(color="#cdccca", size=12),
    ))
    fig_c.update_layout(
        **PLOT_LAYOUT, height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Total LLM cost ($)", tickprefix="$",
                   title_font=dict(color="#7a7974"), **AXIS_STYLE),
        xaxis=dict(**AXIS_STYLE),
        showlegend=False,
    )
    st.plotly_chart(fig_c, use_container_width=True)
with c2:
    fig_tk = go.Figure(go.Bar(
        x=[ARM_SHORT[a] for a in ARM_ORDER], y=[t/1_000_000 for t in tokens],
        marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
        text=[f"{t/1_000_000:.2f}M" for t in tokens], textposition="outside",
        textfont=dict(color="#cdccca", size=12),
    ))
    fig_tk.update_layout(
        **PLOT_LAYOUT, height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Tokens consumed (millions)",
                   title_font=dict(color="#7a7974"), **AXIS_STYLE),
        xaxis=dict(**AXIS_STYLE),
        showlegend=False,
    )
    st.plotly_chart(fig_tk, use_container_width=True)

st.info(
    "**Insight (§4.4.1)** — Token volume is similar across providers (≈0.3M for Mono, ≈2.5M for Council), "
    "but **dollar cost diverges by ~38×** because of OpenAI's price-per-token. The cost gap is a "
    "*provider* effect, not an architecture effect."
)


# ═══════════════════════════════════════════════════════════════════════════
# 3 — ROI multiplier
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Return on Intelligence (ROI ×)")

show_a = st.checkbox(
    "Include Arm A (denominator artefact)",
    value=False,
    help=("Arm A's $0.024 LLM cost is so small that ROI explodes to ≈ −24,000× and dominates the "
          "axis. Off by default for readability; check to see the raw number."),
)

roi_log_y = st.checkbox(
    "Compress extreme ROI on Y-axis (signed log₁₀)",
    value=False,
    help=("When Arm A is included, ROI spans orders of magnitude. This applies "
          "sign(x)·log₁₀(|x|+1) so all arms stay visible while preserving sign."),
)

if show_a:
    plot_arms = ARM_ORDER
else:
    plot_arms = [a for a in ARM_ORDER if a != "A"]

roi_vals = [roi_d.get(a, 0) or 0 for a in plot_arms]

if roi_log_y:
    y_roi = [np.sign(v) * np.log10(abs(v) + 1.0) for v in roi_vals]
    y_title = "Signed log₁₀(|ROI| + 1)"
    text_outer = [f"{v:,.1f}×" for v in roi_vals]
else:
    y_roi = roi_vals
    y_title = "ROI (× net return / $ cost)"
    text_outer = [f"{v:,.1f}×" for v in roi_vals]

bar_kw: dict = dict(
    x=[ARM_SHORT[a] for a in plot_arms],
    y=y_roi,
    marker_color=[ARM_COLORS[a] for a in plot_arms],
    text=text_outer,
    textposition="outside",
    textfont=dict(color="#cdccca", size=14),
)
if roi_log_y:
    bar_kw["customdata"] = roi_vals
    bar_kw["hovertemplate"] = "%{x}<br>ROI: %{customdata:,.2f}×<extra></extra>"
else:
    bar_kw["hovertemplate"] = "%{x}<br>ROI: %{y:,.2f}×<extra></extra>"

fig_roi = go.Figure(go.Bar(**bar_kw))
fig_roi.add_hline(y=0, line_dash="dot", line_color="#393836")
fig_roi.update_layout(
    **PLOT_LAYOUT, height=380, margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(title=y_title,
               title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(**AXIS_STYLE),
    showlegend=False,
)
st.plotly_chart(fig_roi, use_container_width=True)

if roi_log_y:
    st.caption(
        "Y-axis uses a **signed logarithmic transform** so Arm A's extreme multiplier "
        "does not flatten the other bars. Hover shows the **true ROI ×**."
    )

if not show_a:
    st.caption(
        f"Arm A's nominal ROI is **{roi_d.get('A', 0):,.0f}×** — a denominator artefact "
        "($0.024 inference cost vs a small negative return), not a meaningful efficiency signal. "
        "See §4.4.2."
    )

with st.expander("How to read ROI multipliers"):
    st.markdown(
        "- **ROI = 10×** → about $10 net return per $1 of LLM cost.\n"
        "- **ROI = 1×** → break-even on inference spend (before non-LLM operating costs).\n"
        "- **0 < ROI < 1** → positive return but weak cost-efficiency.\n"
        "- **ROI < 0** → arm lost money while still consuming LLM cost.\n"
        "- The ratio amplifies tiny denominators — read it together with **Sharpe**, **MDD**, and **$/bp**."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4 — Cost per basis point
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Cost per basis point of return ($/bp)")
st.caption("Lower bars = more cost-efficient. A standard financial efficiency unit.")

cpbp_vals = [cpbp_d.get(a, 0) or 0 for a in ARM_ORDER]
fig_cp = go.Figure(go.Bar(
    x=[ARM_SHORT[a] for a in ARM_ORDER], y=cpbp_vals,
    marker_color=[ARM_COLORS[a] for a in ARM_ORDER],
    text=[f"${v:.4f}" for v in cpbp_vals], textposition="outside",
    textfont=dict(color="#cdccca", size=13),
))
fig_cp.update_layout(
    **PLOT_LAYOUT, height=340, margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title="$ per basis point", tickprefix="$",
               title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(**AXIS_STYLE),
    showlegend=False,
)
st.plotly_chart(fig_cp, use_container_width=True)

st.info(
    f"**Insight (§4.4.2)** — **Arm B (Qwen + Council)** is the cost-efficiency winner at "
    f"~${cpbp_d.get('B', 0):.4f}/bp — about "
    f"{(cpbp_d.get('C', 1) / max(cpbp_d.get('B', 1e-9), 1e-9)):.1f}× cheaper "
    f"per basis point than Arm C. Council layered on a cheap open-source model is the dominant "
    f"design from a *cost-per-unit-of-alpha* perspective."
)


# ═══════════════════════════════════════════════════════════════════════════
# 5 — Factorial decomposition of ROI
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Factorial decomposition of ROI")

def _roi(arm: str) -> float:
    return float(roi_d.get(arm) or 0)

model_eff_roi = (_roi("C") + _roi("D")) / 2 - (_roi("A") + _roi("B")) / 2
arch_eff_roi  = (_roi("B") + _roi("D")) / 2 - (_roi("A") + _roi("C")) / 2
inter_roi     = (_roi("A") - _roi("B") - _roi("C") + _roi("D")) / 2

cols = st.columns(3)
for col, label, val, accent, note in [
    (cols[0], "Model effect (GPT − Qwen)",          model_eff_roi, "#fdab43",
     "Negative because Arm A inflates Qwen's avg ROI"),
    (cols[1], "Architecture effect (Council − Mono)", arch_eff_roi, "#a86fdf",
     "Positive ⇒ Council improves $-of-return-per-$-cost"),
    (cols[2], "Interaction (A − B − C + D)/2",       inter_roi,    "#22d3ee",
     "Pairing is roughly additive in ROI space"),
]:
    sgn = "+" if val >= 0 else ""
    pc  = COLOR_POS if val >= 0 else COLOR_NEG
    with col:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;
                    border-top:3px solid {accent};border-radius:10px;
                    padding:1rem 1.2rem;text-align:center">
          <div style="font-size:.7rem;color:#7a7974;margin-bottom:.4rem">{label}</div>
          <div style="font-size:1.65rem;font-weight:700;color:{pc};
                      font-variant-numeric:tabular-nums">{sgn}{val:,.1f}×</div>
          <div style="font-size:.72rem;color:#5a5957;margin-top:.4rem">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 6 — Full comparison table
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Complete arm comparison")
rows = []
for arm in ARM_ORDER:
    arm_r = returns_df[returns_df["arm_id"] == arm]
    arm_d = decisions_df[decisions_df["arm_id"] == arm]
    arm_t = trades_df[trades_df["arm_id"] == arm]
    cr  = arm_r["cum_return_pct"].iloc[-1] if not arm_r.empty else 0.0
    rows.append({
        "Arm":           ARM_LABELS[arm],
        "Return (%)":    f"{cr:+.3f}",
        "Sharpe":        f"{sharpe_d.get(arm):.3f}" if sharpe_d.get(arm) is not None else "—",
        "LLM cost ($)":  f"${arm_d['llm_cost_usd'].sum():.4f}",
        "Cost / trade":  (f"${arm_d['llm_cost_usd'].sum() / max(len(arm_t), 1):.5f}"
                         if len(arm_t) > 0 else "—"),
        "$/bp":          f"${cpbp_d.get(arm):.4f}" if cpbp_d.get(arm) else "—",
        "ROI (×)":       f"{roi_d.get(arm):,.1f}" if roi_d.get(arm) is not None else "—",
        "α (%)":         f"{alpha_d.get(arm, 0):.1f}",
        "Trades":        len(arm_t),
        "Cycles":        len(arm_d),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

render_takeaway(
    "<b>The headline business finding</b>: Council > Monolithic on every cost-efficiency metric. "
    "Pair Council with a cheap model (<b>Arm B</b>) for the best <b>$/bp</b>, or with a strong model "
    "(<b>Arm D</b>) for the best raw ROI multiplier. Arm A's eye-popping ROI is a denominator artefact, "
    "not a recommendation."
)
