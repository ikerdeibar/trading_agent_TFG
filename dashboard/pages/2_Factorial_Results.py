"""Factorial Results — the centerpiece. Decompose return into model & architecture effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db import (
    load_portfolio_snapshots, load_agent_decisions,
    compute_daily_equity, compute_returns,
    ARM_LABELS, ARM_SHORT, ARM_COLORS, ARM_ORDER,
    PLOT_LAYOUT, AXIS_STYLE, LEGEND_STYLE,
    COLOR_POS, COLOR_NEG,
    render_sidebar_about, render_takeaway,
)

st.set_page_config(page_title="Factorial Results", page_icon="🧪", layout="wide")
render_sidebar_about()

st.markdown("## 2×2 Factorial Decomposition")
st.caption(
    "The thesis isolates two factors — **model class** (Qwen 235B vs GPT-4.1) and "
    "**architecture** (Monolithic vs Council) — by varying one at a time while holding "
    "everything else identical (same data, same Judge layer, same 10-asset universe)."
)

portfolio_df = load_portfolio_snapshots()
if portfolio_df.empty:
    st.warning("No portfolio data found.")
    st.stop()

decisions_df = load_agent_decisions()
returns_df   = compute_returns(compute_daily_equity(portfolio_df))


def _last_ret(arm: str) -> float:
    g = returns_df[returns_df["arm_id"] == arm]
    return float(g["cum_return_pct"].iloc[-1]) if not g.empty else 0.0


grid = np.array([
    [_last_ret("A"), _last_ret("B")],   # row 0 — Qwen
    [_last_ret("C"), _last_ret("D")],   # row 1 — GPT-4.1
])
mono_mean    = (grid[0, 0] + grid[1, 0]) / 2          # A + C
council_mean = (grid[0, 1] + grid[1, 1]) / 2          # B + D
qwen_mean    = (grid[0, 0] + grid[0, 1]) / 2          # A + B
gpt_mean     = (grid[1, 0] + grid[1, 1]) / 2          # C + D
arch_eff     = council_mean - mono_mean
model_eff    = gpt_mean - qwen_mean
interaction  = (grid[0, 0] - grid[0, 1] - grid[1, 0] + grid[1, 1]) / 2


# ═══════════════════════════════════════════════════════════════════════════
# 1 — Heatmap (rows = architecture, cols = model — thesis convention)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### The 2×2 grid · cumulative return per cell")
st.caption("Rows = architecture. Columns = model class. Cell label = arm + final return.")

st.info(
    "**Grid legend:** **rows** = agent architecture (Monolithic → Council); "
    "**columns** = LLM model class (Qwen 235B → GPT-4.1)."
)

display_grid = np.array([
    [_last_ret("A"), _last_ret("C")],
    [_last_ret("B"), _last_ret("D")],
])
row_labels = ["Monolithic", "Council"]
col_labels = [
    "Qwen 235B<br><span style='font-size:11px;color:#7a7974'>OpenRouter</span>",
    "GPT-4.1<br><span style='font-size:11px;color:#7a7974'>OpenAI</span>",
]
arm_grid = [["A", "C"], ["B", "D"]]

text_cells = []
for i in range(2):
    row_text = []
    for j in range(2):
        v = display_grid[i, j]
        sign = "+" if v >= 0 else ""
        row_text.append(
            f"<b>Arm {arm_grid[i][j]}</b><br>"
            f"<span style='font-size:24px'>{sign}{v:.3f}%</span>"
        )
    text_cells.append(row_text)

fig_hm = go.Figure(go.Heatmap(
    z=display_grid, x=col_labels, y=row_labels,
    text=text_cells, texttemplate="%{text}",
    textfont=dict(size=15, color="white"),
    colorscale=[[0.0, "#922b21"], [0.35, "#2a2826"], [1.0, "#1e8449"]],
    zmin=-1.5, zmax=2.0, showscale=False,
    hovertemplate="Architecture: %{y}<br>Model: %{x}<br>Return: %{z:+.3f}%<extra></extra>",
    xgap=4, ygap=4,
))

# Marginal mean annotations — bottom = column means (model effect); right = row means (arch effect)
for j, (lab, val) in enumerate([(col_labels[0], qwen_mean), (col_labels[1], gpt_mean)]):
    sign = "+" if val >= 0 else ""
    fig_hm.add_annotation(
        x=lab, y=-0.32, yref="paper",
        text=f"<b>μ = {sign}{val:.3f}%</b>",
        showarrow=False, font=dict(size=12, color="#fdab43"),
    )
for i, (lab, val) in enumerate([(row_labels[0], mono_mean), (row_labels[1], council_mean)]):
    sign = "+" if val >= 0 else ""
    fig_hm.add_annotation(
        x=1.16, xref="paper", y=lab,
        text=f"<b>μ = {sign}{val:.3f}%</b>",
        showarrow=False, font=dict(size=12, color="#a86fdf"),
    )

fig_hm.update_layout(
    **PLOT_LAYOUT, height=460, margin=dict(l=10, r=110, t=20, b=70),
    xaxis=dict(side="top", tickfont=dict(size=14, color="#cdccca"),
               gridcolor="rgba(0,0,0,0)"),
    yaxis=dict(autorange="reversed", tickfont=dict(size=13, color="#cdccca"),
               gridcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_hm, use_container_width=True)

with st.expander("How to read this heatmap"):
    st.markdown(
        "- **Rows** = agent architecture (Monolithic on top, Council below).\n"
        "- **Columns** = LLM model class (Qwen 235B on the left, GPT-4.1 on the right).\n"
        "- The **cell value** is each arm's final cumulative return after 16 sessions.\n"
        "- The **μ annotations** isolate one factor at a time:\n"
        "  - Compare *bottom column μ's* → **model effect** (averaged over both architectures).\n"
        "  - Compare *right-hand row μ's* → **architecture effect** (averaged over both models).\n"
        "- Colour scale: red = loss, green = gain (anchored at zero)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2 — Side-by-side effect bars
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Marginal-mean comparisons")

col_l, col_r = st.columns(2)

# ── Model effect ──
with col_l:
    fig_m = go.Figure(go.Bar(
        x=["Qwen 235B<br>(A + B avg)", "GPT-4.1<br>(C + D avg)"],
        y=[qwen_mean, gpt_mean],
        marker_color=["#7a7974", "#fdab43"],
        text=[f"{qwen_mean:+.3f}%", f"{gpt_mean:+.3f}%"], textposition="outside",
        textfont=dict(color="#cdccca", size=14),
    ))
    fig_m.add_hline(y=0, line_dash="dot", line_color="#393836")
    fig_m.update_layout(
        **PLOT_LAYOUT, height=340, margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text=f"Model effect = <b style='color:#fdab43'>{model_eff:+.3f}%</b>",
                   font=dict(color="#cdccca", size=15)),
        yaxis=dict(title="Mean cumulative return (%)", ticksuffix="%",
                   title_font=dict(color="#7a7974"), **AXIS_STYLE),
        xaxis=dict(**AXIS_STYLE),
        showlegend=False,
    )
    st.plotly_chart(fig_m, use_container_width=True)

# ── Architecture effect ──
with col_r:
    fig_a = go.Figure(go.Bar(
        x=["Monolithic<br>(A + C avg)", "Council<br>(B + D avg)"],
        y=[mono_mean, council_mean],
        marker_color=["#7a7974", "#a86fdf"],
        text=[f"{mono_mean:+.3f}%", f"{council_mean:+.3f}%"], textposition="outside",
        textfont=dict(color="#cdccca", size=14),
    ))
    fig_a.add_hline(y=0, line_dash="dot", line_color="#393836")
    fig_a.update_layout(
        **PLOT_LAYOUT, height=340, margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text=f"Architecture effect = <b style='color:#a86fdf'>{arch_eff:+.3f}%</b>",
                   font=dict(color="#cdccca", size=15)),
        yaxis=dict(title="Mean cumulative return (%)", ticksuffix="%",
                   title_font=dict(color="#7a7974"), **AXIS_STYLE),
        xaxis=dict(**AXIS_STYLE),
        showlegend=False,
    )
    st.plotly_chart(fig_a, use_container_width=True)

st.info(
    f"**Insight (§4.3.1)** — Both factors move in the same direction and roughly the same size: "
    f"upgrading the model is worth +{model_eff:.3f}% on average, switching to the Council "
    f"architecture is worth +{arch_eff:.3f}%. **Architecture is therefore a *first-order* "
    f"performance lever, not a deployment afterthought.**"
)


# ═══════════════════════════════════════════════════════════════════════════
# 3 — Effect-size cards
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Effect-size summary")
c1, c2, c3 = st.columns(3)
for col, label, val, accent, note in [
    (c1, "Model effect (GPT − Qwen)",            model_eff, "#fdab43",
     "Holds architecture constant, averages over Mono & Council"),
    (c2, "Architecture effect (Council − Mono)", arch_eff,  "#a86fdf",
     "Holds model constant, averages over Qwen & GPT-4.1"),
    (c3, "Interaction (A − B − C + D) / 2",      interaction, "#22d3ee",
     "Positive ⇒ the Council premium is bigger for GPT than for Qwen"),
]:
    sgn = "+" if val >= 0 else ""
    pc  = COLOR_POS if val >= 0 else COLOR_NEG
    with col:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;
                    border-top:3px solid {accent};border-radius:10px;
                    padding:1rem 1.2rem;text-align:center">
          <div style="font-size:.7rem;color:#7a7974;margin-bottom:.4rem">{label}</div>
          <div style="font-size:1.95rem;font-weight:700;color:{pc};
                      font-variant-numeric:tabular-nums">{sgn}{val:.3f}%</div>
          <div style="font-size:.72rem;color:#5a5957;margin-top:.4rem">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4 — Win-rate / payoff table
# ═══════════════════════════════════════════════════════════════════════════
import json
st.markdown("### Win rate & implied payoff")
st.caption("Per-arm, computed from the LLM-written `post_mortem` field of each cycle.")

payoff = []
from db import START_EQUITY
for arm in ARM_ORDER:
    arm_ret = returns_df[returns_df["arm_id"] == arm]
    net = arm_ret["equity"].iloc[-1] - START_EQUITY if not arm_ret.empty else 0.0

    wins = losses = 0
    for _, row in decisions_df[decisions_df["arm_id"] == arm].iterrows():
        try:
            pm = (row["post_mortem"] if isinstance(row["post_mortem"], dict)
                  else json.loads(row["post_mortem"] or "{}"))
            for _sym, val in (pm or {}).items():
                pnl = val.get("pnl_pct", 0) if isinstance(val, dict) else 0
                if pnl > 0: wins += 1
                else:       losses += 1
        except Exception:
            pass

    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0.0
    if net > 0 and wins > 0:
        avg_win = f"+${net / wins:,.2f}"
    else:
        avg_win = "n/a"
    payoff.append({
        "Arm":             ARM_LABELS[arm],
        "Total trades":    total,
        "Wins":            wins,
        "Losses":          losses,
        "Win rate (%)":    f"{win_rate:.1f}",
        "Net return ($)":  f"{net:+,.2f}",
        "Implied avg win": avg_win,
    })
st.dataframe(pd.DataFrame(payoff), use_container_width=True, hide_index=True)

render_takeaway(
    f"The interaction term is small (<b>{interaction:+.3f}%</b>), so the two factors are "
    "roughly **additive**: each one independently adds about +1 percentage point of return. "
    "The cell that combines both upgrades — <b>Arm D · GPT-4.1 · Council</b> — is the only "
    "one that meaningfully beats SPY over the validation window."
)
