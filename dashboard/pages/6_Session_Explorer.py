"""Session Explorer — drill into any single trading day across all four arms."""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db import (
    load_portfolio_snapshots, load_agent_decisions, load_trades,
    load_market_snapshots, load_sentiment_snapshots,
    ARM_LABELS, ARM_SHORT, ARM_COLORS, ARM_ORDER,
    PLOT_LAYOUT, AXIS_STYLE, LEGEND_STYLE,
    COLOR_POS, COLOR_NEG,
    render_sidebar_about, render_takeaway,
    apply_datetime_x_range,
)

st.set_page_config(page_title="Session Explorer", page_icon="📅", layout="wide")
render_sidebar_about()

st.markdown("## 📅 Daily Session Explorer")
st.caption(
    "Drill into any single trading day. Useful for explaining what an arm "
    "*actually did* during a specific market move."
)

portfolio_df = load_portfolio_snapshots()
decisions_df = load_agent_decisions()
trades_df    = load_trades()
market_df    = load_market_snapshots()
sent_df      = load_sentiment_snapshots()

available_dates = (sorted(decisions_df["date"].unique(), reverse=True)
                   if not decisions_df.empty else [])
if not available_dates:
    st.warning("No session data found.")
    st.stop()

cd, ca = st.columns([2, 3])
with cd:
    selected_date = st.selectbox("Trading date", available_dates, format_func=lambda d: str(d))
with ca:
    sel_arms = st.multiselect("Arms to display", ARM_ORDER, default=ARM_ORDER,
                              format_func=lambda a: ARM_SHORT[a])

st.markdown("---")

day_dec    = decisions_df[decisions_df["date"] == selected_date]
day_trades = trades_df[trades_df["date"] == selected_date]
day_port   = portfolio_df[portfolio_df["date"] == selected_date]
day_mkt    = market_df[market_df["date"] == selected_date]
day_sent   = sent_df[sent_df["date"] == selected_date]


# ═══════════════════════════════════════════════════════════════════════════
# KPI cards
# ═══════════════════════════════════════════════════════════════════════════
kpis = st.columns(4)
for i, arm in enumerate(ARM_ORDER):
    arm_port = day_port[day_port["arm_id"] == arm].sort_values("captured_at")
    arm_trd  = day_trades[day_trades["arm_id"] == arm]
    arm_dec  = day_dec[day_dec["arm_id"] == arm]
    eq_start = arm_port["equity"].iloc[0]  if not arm_port.empty else 100_000.0
    eq_end   = arm_port["equity"].iloc[-1] if not arm_port.empty else 100_000.0
    pnl      = eq_end - eq_start
    cost     = arm_dec["llm_cost_usd"].sum()
    sign     = "+" if pnl >= 0 else ""
    pc       = COLOR_POS if pnl >= 0 else COLOR_NEG
    c        = ARM_COLORS[arm]
    with kpis[i]:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;border-radius:10px;
                    padding:.95rem 1.1rem">
          <div style="font-size:.66rem;color:{c};text-transform:uppercase;
                      letter-spacing:.1em;font-weight:600;margin-bottom:.3rem">{ARM_LABELS[arm]}</div>
          <div style="font-size:1.35rem;font-weight:700;color:#cdccca;
                      font-variant-numeric:tabular-nums">${eq_end:,.0f}</div>
          <div style="font-size:.85rem;font-weight:600;color:{pc}">
            {sign}${pnl:,.0f} today
          </div>
          <div style="font-size:.7rem;color:#5a5957;margin-top:.3rem">
            {len(arm_trd)} trades · ${cost:.4f} LLM cost
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Intraday equity + trade markers
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Intraday equity with executed-trade markers")
fig_eq = go.Figure()
for arm in sel_arms:
    arm_port = day_port[day_port["arm_id"] == arm].sort_values("captured_at")
    if arm_port.empty:
        continue
    fig_eq.add_trace(go.Scatter(
        x=arm_port["captured_at"], y=arm_port["equity"],
        mode="lines+markers", name=ARM_SHORT[arm],
        line=dict(color=ARM_COLORS[arm], width=2.4),
        marker=dict(size=4),
        hovertemplate=f"<b>{ARM_LABELS[arm]}</b><br>%{{x|%H:%M ET}}<br>$%{{y:,.0f}}<extra></extra>",
    ))
    arm_trd = day_trades[day_trades["arm_id"] == arm]
    n_trades = len(arm_trd)
    # Scale marker size down on dense days so triangles don't overlap
    marker_size = max(7, min(11, int(14 - n_trades * 0.05)))
    for side, sym, color in [("buy", "triangle-up", COLOR_POS),
                              ("sell", "triangle-down", COLOR_NEG)]:
        side_trd = arm_trd[arm_trd["side"] == side]
        if side_trd.empty:
            continue
        # Snap each trade time to the nearest portfolio snapshot's equity
        eq_lookup = arm_port.set_index("captured_at")["equity"].sort_index()
        ys = []
        for ts in side_trd["created_at"]:
            idx = eq_lookup.index.get_indexer([ts], method="nearest")[0]
            ys.append(float(eq_lookup.iloc[idx]))
        fig_eq.add_trace(go.Scatter(
            x=side_trd["created_at"], y=ys,
            mode="markers",
            marker=dict(symbol=sym, size=marker_size, color=color,
                        line=dict(color="#171614", width=0.8)),
            showlegend=False,
            hovertemplate=("<b>" + side.upper() + " %{customdata[0]}</b><br>"
                           "qty %{customdata[1]} @ $%{customdata[2]:.2f}<extra></extra>"),
            customdata=list(zip(side_trd["symbol"], side_trd["qty"], side_trd["price"])),
        ))

_ts_pts = []
for arm in sel_arms:
    ap = day_port[day_port["arm_id"] == arm]
    if not ap.empty:
        _ts_pts.extend([ap["captured_at"].min(), ap["captured_at"].max()])
    atr = day_trades[day_trades["arm_id"] == arm]
    if not atr.empty:
        _ts_pts.extend([atr["created_at"].min(), atr["created_at"].max()])

fig_eq.update_layout(
    **PLOT_LAYOUT, height=380, margin=dict(l=10, r=10, t=20, b=0),
    yaxis=dict(tickprefix="$", tickformat=",",
               title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(**AXIS_STYLE),
    legend=LEGEND_STYLE,
)
if _ts_pts:
    apply_datetime_x_range(fig_eq, min(_ts_pts), max(_ts_pts))
st.plotly_chart(fig_eq, use_container_width=True)
st.caption("Triangle markers show executed orders: ▲ buys (green), ▼ sells (pink).")


# ═══════════════════════════════════════════════════════════════════════════
# Ticker prices
# ═══════════════════════════════════════════════════════════════════════════
if not day_mkt.empty:
    st.markdown("### Ticker mid-prices today")
    tickers     = sorted(day_mkt["symbol"].unique())
    sel_tickers = st.multiselect("Pick tickers", tickers, default=tickers[:5])
    fig_mk = go.Figure()
    palette = ["#22d3ee", "#818cf8", "#fdab43", "#a86fdf", "#34d399",
               "#f472b6", "#facc15", "#38bdf8", "#c084fc", "#86efac"]
    for i, sym in enumerate(sel_tickers):
        s = day_mkt[day_mkt["symbol"] == sym].sort_values("captured_at")
        fig_mk.add_trace(go.Scatter(
            x=s["captured_at"], y=s["mid"],
            mode="lines+markers", name=sym,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=4),
        ))
    fig_mk.update_layout(
        **PLOT_LAYOUT, height=320, margin=dict(l=10, r=10, t=20, b=0),
        yaxis=dict(tickprefix="$", title_font=dict(color="#7a7974"), **AXIS_STYLE),
        xaxis=dict(**AXIS_STYLE),
        legend=LEGEND_STYLE,
    )
    if sel_tickers:
        mk = day_mkt[day_mkt["symbol"].isin(sel_tickers)]
        if not mk.empty:
            apply_datetime_x_range(fig_mk, mk["captured_at"].min(), mk["captured_at"].max())
    st.plotly_chart(fig_mk, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sentiment of the day
# ═══════════════════════════════════════════════════════════════════════════
if not day_sent.empty:
    st.markdown("### Today's sentiment per ticker")
    day_avg = (day_sent.groupby("symbol")["score"].mean()
               .reset_index().sort_values("score", ascending=False))
    fig_s = go.Figure(go.Bar(
        x=day_avg["symbol"], y=day_avg["score"],
        marker_color=day_avg["score"].apply(
            lambda s: COLOR_POS if s > 0.1 else (COLOR_NEG if s < -0.1 else "#7a7974")),
        text=[f"{v:+.2f}" for v in day_avg["score"]], textposition="outside",
        textfont=dict(color="#cdccca"),
        hovertemplate="%{x}<br>%{y:+.3f}<extra></extra>",
    ))
    fig_s.add_hline(y=0, line_dash="dot", line_color="#393836")
    fig_s.update_layout(
        **PLOT_LAYOUT, height=300, margin=dict(l=10, r=10, t=20, b=0),
        yaxis=dict(range=[-1, 1], title="Avg FinBERT score",
                   title_font=dict(color="#7a7974"), **AXIS_STYLE),
        xaxis=dict(**AXIS_STYLE),
        showlegend=False,
    )
    st.plotly_chart(fig_s, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Cycle-by-cycle decision log
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Cycle-by-cycle decision log")
for arm in sel_arms:
    arm_cycles = day_dec[day_dec["arm_id"] == arm].sort_values("cycle_ts")
    if arm_cycles.empty:
        continue
    color = ARM_COLORS[arm]
    st.markdown(
        f"<div style='display:inline-block;background:{color}33;color:{color};"
        f"padding:.2rem .8rem;border-radius:999px;font-size:.75rem;font-weight:600;"
        f"margin:.6rem 0 .4rem 0'>{ARM_LABELS[arm]}</div>",
        unsafe_allow_html=True,
    )
    for _, cyc in arm_cycles.iterrows():
        ts = cyc["cycle_ts"].tz_convert("America/New_York").strftime("%H:%M ET")
        def _count(col):
            from db import _coerce_list
            return len(_coerce_list(cyc.get(col)))
        n_prop  = _count("orders_proposed")
        n_exec  = _count("orders_executed")
        n_block = _count("orders_blocked")
        cost    = cyc["llm_cost_usd"] or 0
        label   = (f"🕐 {ts}  ·  📋 {n_prop} proposed  ·  "
                   f"✅ {n_exec} executed  ·  🚫 {n_block} blocked  ·  ${cost:.4f}")
        with st.expander(label):
            from db import _coerce_list
            executed = _coerce_list(cyc.get("orders_executed"))
            if executed:
                st.markdown("**Orders executed:**")
                for o in executed:
                    if isinstance(o, dict):
                        st.markdown(
                            f"- `{o.get('side','?').upper()}` "
                            f"**{o.get('symbol','?')}** × {o.get('qty','?')} "
                            f"(confidence: {o.get('confidence','?')})"
                        )
            blocked = _coerce_list(cyc.get("orders_blocked"))
            if blocked:
                st.markdown("**Risk-Guard blocks:**")
                for b in blocked:
                    s = b.get("symbol", "?")  if isinstance(b, dict) else "?"
                    r = b.get("reason", "")   if isinstance(b, dict) else str(b)
                    st.markdown(f"- 🚫 `{s}` — {r}")
            reasoning = cyc.get("reasoning") or ""
            if reasoning:
                st.markdown("**LLM reasoning:**")
                st.markdown(
                    f"<div style='background:#22211f;border-radius:8px;padding:.85rem;"
                    f"font-size:.85rem;line-height:1.55;white-space:pre-wrap;"
                    f"max-height:300px;overflow-y:auto;color:#cdccca'>"
                    f"{str(reasoning)[:3000]}</div>",
                    unsafe_allow_html=True,
                )
            mem = cyc.get("memory_packet_in")
            if mem:
                st.markdown("**Memory packet (≤200 tokens):**")
                st.caption(str(mem))


# ═══════════════════════════════════════════════════════════════════════════
# Risk-Guard summary table
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Risk-Guard summary (today)")
all_blocks = []
from db import _coerce_list
for _, row in day_dec[day_dec["arm_id"].isin(sel_arms)].iterrows():
    for b in _coerce_list(row.get("orders_blocked")):
        if isinstance(b, dict):
            all_blocks.append({
                "Arm":    ARM_LABELS.get(row["arm_id"], row["arm_id"]),
                "Time":   row["cycle_ts"].tz_convert("America/New_York").strftime("%H:%M ET"),
                "Symbol": b.get("symbol", "?"),
                "Side":   b.get("side", "?"),
                "Qty":    b.get("qty", "?"),
                "Reason": b.get("reason", "?"),
            })
if all_blocks:
    st.dataframe(pd.DataFrame(all_blocks), use_container_width=True, hide_index=True)
else:
    st.success("No risk-guard blocks on this date.")

render_takeaway(
    "Use this page during the defense to <b>show the jury exactly what an arm did on a given day</b>: "
    "every cycle's reasoning, every executed and blocked order, and the intraday market context. "
    "It is the audit trail that backs every aggregate metric on the other pages."
)
