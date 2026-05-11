"""Trades & Risk — execution, judge-layer blocks, ticker concentration, full log."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db import (
    load_trades, load_agent_decisions, compute_block_rate, compute_action_rate,
    ARM_LABELS, ARM_SHORT, ARM_COLORS, ARM_ORDER,
    PLOT_LAYOUT, AXIS_STYLE, LEGEND_STYLE,
    COLOR_POS, COLOR_NEG,
    render_sidebar_about, render_takeaway,
)

st.set_page_config(page_title="Trades & Risk", page_icon="🔄", layout="wide")
render_sidebar_about()

st.markdown("## Trades & Risk")
st.caption(
    "Every executed and blocked order across all four arms over the validation window. "
    "Trades only fire when the LLM's proposal passes all five Judge-layer gates."
)

trades_df    = load_trades()
decisions_df = load_agent_decisions()
block_df     = compute_block_rate(decisions_df).set_index("arm_id")
alpha_d      = compute_action_rate(decisions_df)


# ═══════════════════════════════════════════════════════════════════════════
# 1 — Block rate vs Action rate per arm
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Judge-layer block rate vs action rate")
st.caption(
    "**Block rate** = % of proposed orders the Judge layer rejected. "
    "**Action rate (α)** = % of OODA cycles that produced ≥1 executed order. "
    "High block rate ≠ high action rate — Council arms propose more *and* get more through."
)

arms = ARM_ORDER
block_rates = []
for arm in arms:
    if arm in block_df.index:
        p = int(block_df.loc[arm, "proposed"])
        b = int(block_df.loc[arm, "blocked"])
        block_rates.append(b / p * 100 if p > 0 else 0)
    else:
        block_rates.append(0)
action_rates = [alpha_d.get(a, 0) for a in arms]

fig_ba = go.Figure()
fig_ba.add_trace(go.Bar(
    x=[ARM_SHORT[a] for a in arms], y=block_rates,
    name="Judge block rate (%)",
    marker_color=COLOR_NEG,
    text=[f"{v:.1f}%" for v in block_rates], textposition="outside",
    textfont=dict(color="#cdccca"),
))
fig_ba.add_trace(go.Bar(
    x=[ARM_SHORT[a] for a in arms], y=action_rates,
    name="Action rate α (%)",
    marker_color=COLOR_POS,
    text=[f"{v:.1f}%" for v in action_rates], textposition="outside",
    textfont=dict(color="#cdccca"),
))
fig_ba.update_layout(
    **PLOT_LAYOUT, barmode="group",
    height=380, margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(title="Rate (%)", range=[0, 110], ticksuffix="%",
               title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(**AXIS_STYLE),
    legend=LEGEND_STYLE,
)
st.plotly_chart(fig_ba, use_container_width=True)

with st.expander("How to read this"):
    st.markdown(
        "- **Pink bar** — % of proposals stopped by the Judge. Anything from 50% to 95% is normal: "
        "the Judge rejects schema errors, stale tickers, position-cap breaches, off-hours requests, "
        "and over-cycled days.\n"
        "- **Green bar** — % of cycles that *still* shipped at least one trade. The two metrics diverge "
        "because a single cycle can survive blocks.\n"
        "- **Healthy** = high α with low blocks (D = α 92%, blocks 47%). **Stuck** = high blocks with "
        "low α (A = α 19%, blocks 95%)."
    )

st.info(
    "**Insight (§4.5)** — The Council arms keep the Judge busy *and* still trade frequently. "
    "Arm A spends the run mostly being blocked and produces few trades; Arm D both proposes more "
    "and survives more of those proposals — a clean signature of multi-agent self-correction."
)


# ═══════════════════════════════════════════════════════════════════════════
# 2 — Block-rate detail table
# ═══════════════════════════════════════════════════════════════════════════
with st.expander("Per-arm proposed / executed / blocked counts", expanded=False):
    rows = []
    for arm in arms:
        if arm in block_df.index:
            p = int(block_df.loc[arm, "proposed"])
            e = int(block_df.loc[arm, "executed"])
            b = int(block_df.loc[arm, "blocked"])
        else:
            p = e = b = 0
        rows.append({
            "Arm":              ARM_LABELS[arm],
            "Proposed":         p,
            "Executed":         e,
            "Blocked":          b,
            "Block rate (%)":   f"{(b/p*100 if p else 0):.1f}",
            "Action rate α":    f"{alpha_d.get(arm, 0):.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3 — Trades by symbol — grouped bar (matches presentation/figures)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Ticker concentration — trades per symbol per arm")

if trades_df.empty:
    st.info("No trades found.")
else:
    arms_sel = st.multiselect("Filter arms", ARM_ORDER, default=ARM_ORDER,
                              format_func=lambda a: ARM_SHORT[a])
    sym_counts = (trades_df[trades_df["arm_id"].isin(arms_sel)]
                  .groupby(["arm_id", "symbol"]).size()
                  .reset_index(name="count"))
    if sym_counts.empty:
        st.info("No trades for the selected arms.")
    else:
        # Sort tickers by total volume across selected arms, descending
        ticker_totals = (sym_counts.groupby("symbol")["count"].sum()
                         .sort_values(ascending=False))
        tickers = ticker_totals.index.tolist()

        n_arms = len(arms_sel)
        group_w = 0.78
        bar_w   = group_w / max(n_arms, 1)
        offsets = (np.linspace(-(group_w - bar_w) / 2,
                                (group_w - bar_w) / 2,
                                n_arms) if n_arms > 0 else [0])
        x = np.arange(len(tickers))

        fig_t = go.Figure()
        for i, arm in enumerate(arms_sel):
            arm_counts = sym_counts[sym_counts["arm_id"] == arm].set_index("symbol")["count"]
            vals = [int(arm_counts.get(t, 0)) for t in tickers]
            fig_t.add_trace(go.Bar(
                x=[float(xi + offsets[i]) for xi in x],
                y=vals,
                width=bar_w * 0.94,
                name=ARM_SHORT[arm],
                marker_color=ARM_COLORS[arm],
                text=[str(v) if v >= 4 else "" for v in vals],
                textposition="outside", textfont=dict(color=ARM_COLORS[arm], size=10),
                hovertemplate=f"<b>{ARM_LABELS[arm]}</b><br>%{{customdata}}<br>%{{y}} trades<extra></extra>",
                customdata=tickers,
            ))

        totals_str = "  ·  ".join(
            f"{ARM_SHORT[a]}: {int(sym_counts[sym_counts['arm_id']==a]['count'].sum())}"
            for a in arms_sel
        )

        fig_t.update_layout(
            **PLOT_LAYOUT,
            height=420, margin=dict(l=10, r=10, t=20, b=20),
            barmode="overlay",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(tickers))),
                ticktext=tickers,
                title=None, **AXIS_STYLE,
            ),
            yaxis=dict(title="Trade count",
                       title_font=dict(color="#7a7974"), **AXIS_STYLE),
            legend=LEGEND_STYLE,
        )
        st.plotly_chart(fig_t, use_container_width=True)
        st.caption(f"Totals: {totals_str}")

        with st.expander("How to read this"):
            st.markdown(
                "- Tickers are sorted left-to-right by total trade volume across the selected arms.\n"
                "- Each ticker has up to four bars, one per arm — same colour palette as everywhere else.\n"
                "- Numeric labels appear over bars with ≥ 4 trades to keep the chart legible."
            )

        st.info(
            "**Insight (§4.5.2)** — Arm D distributes its 357 trades across the 10-asset basket "
            "(NVDA, AAPL, JPM, AMZN, GOOGL all > 40 trades) — a sign of healthy diversification. "
            "Arms with few trades (A: 31, C: 35) concentrate sharply on XOM, suggesting tunnel vision."
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4 — Full trade log
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Full trade log")
if trades_df.empty:
    st.info("No trades to show.")
else:
    f_cols = st.columns(3)
    arm_f  = f_cols[0].multiselect("Arm",    ARM_ORDER, default=ARM_ORDER,
                                    format_func=lambda a: ARM_SHORT[a])
    side_f = f_cols[1].multiselect("Side",   ["buy", "sell"], default=["buy", "sell"])
    sym_all= sorted(trades_df["symbol"].unique())
    sym_f  = f_cols[2].multiselect("Symbol", sym_all, default=sym_all)

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
        use_container_width=True, hide_index=True,
    )

render_takeaway(
    "<b>The Judge layer is doing its job</b>: every arm sits between 47% and 97% block rate — "
    "no proposal escapes the deterministic gates. The Council architecture turns the LLM's noise "
    "into decisions that survive judging more often, ending in 4–10× more executed trades than the "
    "Monolithic counterpart at the same model class."
)
