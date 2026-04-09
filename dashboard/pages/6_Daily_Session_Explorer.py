"""Daily Session Explorer — drill into any trading day across all arms."""
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from db import (
    load_portfolio_snapshots, load_agent_decisions, load_trades,
    load_market_snapshots, load_sentiment_snapshots,
    ARM_LABELS, ARM_COLORS, ARM_ORDER,
)

st.set_page_config(page_title="Daily Session Explorer", page_icon="📅", layout="wide")
st.markdown("## 📅 Daily Session Explorer")
st.caption("Select a trading day to inspect equity, trades, LLM reasoning, sentiment, and risk blocks.")

portfolio_df = load_portfolio_snapshots()
decisions_df = load_agent_decisions()
trades_df    = load_trades()
market_df    = load_market_snapshots()
sent_df      = load_sentiment_snapshots()

available_dates = sorted(decisions_df["date"].unique(), reverse=True) if not decisions_df.empty else []
if not available_dates:
    st.warning("No session data found.")
    st.stop()

col_date, col_arms = st.columns([2, 3])
with col_date:
    selected_date = st.selectbox("Trading Date", available_dates,
                                  format_func=lambda d: str(d))
with col_arms:
    sel_arms = st.multiselect("Arms to display", ARM_ORDER, default=ARM_ORDER)

st.markdown("---")

day_dec   = decisions_df[decisions_df["date"] == selected_date]
day_trades= trades_df[trades_df["date"] == selected_date]
day_port  = portfolio_df[portfolio_df["date"] == selected_date]
day_mkt   = market_df[market_df["date"] == selected_date]
day_sent  = sent_df[sent_df["date"] == selected_date]

# ── KPI cards ─────────────────────────────────────────────────────────────
kpi_cols = st.columns(4)
for i, arm in enumerate(ARM_ORDER):
    arm_port = day_port[day_port["arm_id"] == arm].sort_values("captured_at")
    arm_trd  = day_trades[day_trades["arm_id"] == arm]
    arm_dec  = day_dec[day_dec["arm_id"] == arm]

    eq_start = arm_port["equity"].iloc[0]  if not arm_port.empty else 100_000.0
    eq_end   = arm_port["equity"].iloc[-1] if not arm_port.empty else 100_000.0
    day_pnl  = eq_end - eq_start
    cost     = arm_dec["llm_cost_usd"].sum()
    sign     = "+" if day_pnl >= 0 else ""
    pnl_col  = "#437a22" if day_pnl >= 0 else "#a12c7b"

    with kpi_cols[i]:
        st.markdown(f"""
        <div style="background:#1c1b19;border:1px solid #262523;border-radius:10px;
                    padding:1rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,.3)">
          <div style="font-size:.7rem;color:#7a7974;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:.3rem">{ARM_LABELS[arm]}</div>
          <div style="font-size:1.4rem;font-weight:700;font-variant-numeric:tabular-nums;
                      color:#cdccca">${eq_end:,.0f}</div>
          <div style="font-size:.8rem;font-weight:500;color:{pnl_col}">
            {sign}${day_pnl:,.0f} today
          </div>
          <div style="font-size:.7rem;color:#5a5957;margin-top:.2rem">
            {len(arm_trd)} trades · ${cost:.4f} LLM cost
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Intraday equity ────────────────────────────────────────────────────────
st.markdown("### Intraday Equity + Trade Markers")
fig_eq = go.Figure()
for arm in sel_arms:
    arm_port = day_port[day_port["arm_id"] == arm].sort_values("captured_at")
    if arm_port.empty:
        continue
    fig_eq.add_trace(go.Scatter(
        x=arm_port["captured_at"], y=arm_port["equity"],
        mode="lines+markers", name=ARM_LABELS[arm],
        line=dict(color=ARM_COLORS[arm], width=2.5),
        marker=dict(size=5),
        hovertemplate="%{x|%H:%M ET}<br>$%{y:,.0f}<extra></extra>",
    ))
    arm_trd = day_trades[day_trades["arm_id"] == arm]
    for _, t in arm_trd[arm_trd["side"] == "buy"].iterrows():
        closest = arm_port.iloc[
            (arm_port["captured_at"] - t["created_at"]).abs().argsort().iloc[0]]
        fig_eq.add_trace(go.Scatter(
            x=[t["created_at"]], y=[closest["equity"]],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#437a22"),
            showlegend=False,
            hovertemplate=f"BUY {t['symbol']} ×{t['qty']} @ ${t['price']:.2f}<extra></extra>",
        ))
    for _, t in arm_trd[arm_trd["side"] == "sell"].iterrows():
        closest = arm_port.iloc[
            (arm_port["captured_at"] - t["created_at"]).abs().argsort().iloc[0]]
        fig_eq.add_trace(go.Scatter(
            x=[t["created_at"]], y=[closest["equity"]],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#a12c7b"),
            showlegend=False,
            hovertemplate=f"SELL {t['symbol']} ×{t['qty']} @ ${t['price']:.2f}<extra></extra>",
        ))

fig_eq.update_layout(
    height=340, margin=dict(l=0,r=0,t=10,b=0),
    plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
    font=dict(family="sans-serif", color="#cdccca", size=12),
    yaxis=dict(tickprefix="$", tickformat=",", gridcolor="#262523"),
    xaxis=dict(gridcolor="#262523"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_eq, use_container_width=True)

# ── Ticker prices ──────────────────────────────────────────────────────────
st.markdown("### Ticker Prices")
if not day_mkt.empty:
    tickers     = sorted(day_mkt["symbol"].unique())
    sel_tickers = st.multiselect("Select tickers", tickers, default=tickers[:5])
    TICK_COLORS = ["#01696f","#006494","#da7101","#7a39bb","#437a22",
                   "#a12c7b","#d19900","#a13544","#bb653b","#22d3ee"]
    fig_mkt = go.Figure()
    for idx, sym in enumerate(sel_tickers):
        sym_df = day_mkt[day_mkt["symbol"] == sym].sort_values("captured_at")
        fig_mkt.add_trace(go.Scatter(
            x=sym_df["captured_at"], y=sym_df["mid"],
            mode="lines+markers", name=sym,
            line=dict(color=TICK_COLORS[idx % len(TICK_COLORS)], width=2),
            marker=dict(size=4),
        ))
    fig_mkt.update_layout(
        height=300, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
        font=dict(family="sans-serif", color="#cdccca", size=12),
        yaxis=dict(tickprefix="$", gridcolor="#262523"),
        xaxis=dict(gridcolor="#262523"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_mkt, use_container_width=True)

# ── Sentiment ──────────────────────────────────────────────────────────────
st.markdown("### Sentiment Signal")
if not day_sent.empty:
    day_avg = day_sent.groupby("symbol")["score"].mean().reset_index().sort_values("symbol")
    fig_s = go.Figure(go.Bar(
        x=day_avg["symbol"], y=day_avg["score"],
        marker_color=day_avg["score"].apply(
            lambda s: "#437a22" if s > 0.1 else ("#a12c7b" if s < -0.1 else "#7a7974")),
        hovertemplate="%{x}<br>Avg Score: %{y:.3f}<extra></extra>",
    ))
    fig_s.add_hline(y=0, line_dash="dot", line_color="#393836")
    fig_s.update_layout(
        height=260, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
        font=dict(family="sans-serif", color="#cdccca", size=12),
        yaxis=dict(range=[-1,1], title="FinBERT Score", gridcolor="#262523"),
        showlegend=False,
    )
    st.plotly_chart(fig_s, use_container_width=True)

# ── Cycle-by-cycle decision log ────────────────────────────────────────────
st.markdown("### Cycle-by-Cycle Decision Log")
for arm in sel_arms:
    arm_cycles = day_dec[day_dec["arm_id"] == arm].sort_values("cycle_ts")
    if arm_cycles.empty:
        continue
    color = ARM_COLORS[arm]
    st.markdown(
        f"<div style='display:inline-block;background:{color}33;color:{color};"
        f"padding:.2rem .8rem;border-radius:999px;font-size:.75rem;font-weight:600;"
        f"margin-bottom:.5rem'>{ARM_LABELS[arm]}</div>",
        unsafe_allow_html=True
    )
    for _, cyc in arm_cycles.iterrows():
        ts = cyc["cycle_ts"].tz_convert("America/New_York").strftime("%H:%M ET")
        def _count(col):
            try:
                v = cyc.get(col)
                d = json.loads(v) if isinstance(v, str) else v
                return len(d) if isinstance(d, list) else 0
            except Exception:
                return 0
        n_prop  = _count("orders_proposed")
        n_exec  = _count("orders_executed")
        n_block = _count("orders_blocked")
        cost    = cyc["llm_cost_usd"] or 0
        label   = (f"🕐 {ts}  ·  📋 {n_prop} proposed  ·  "
                   f"✅ {n_exec} executed  ·  🚫 {n_block} blocked  ·  ${cost:.4f}")
        with st.expander(label):
            executed = []
            try:
                v = cyc.get("orders_executed")
                executed = (json.loads(v) if isinstance(v, str) else v) or []
            except Exception:
                pass
            if executed:
                st.markdown("**Orders Executed:**")
                for o in executed:
                    if isinstance(o, dict):
                        st.markdown(
                            f"- `{o.get('side','?').upper()}` "
                            f"**{o.get('symbol','?')}** × {o.get('qty','?')} "
                            f"(confidence: {o.get('confidence','?')})"
                        )
            blocked = []
            try:
                v = cyc.get("orders_blocked")
                blocked = (json.loads(v) if isinstance(v, str) else v) or []
            except Exception:
                pass
            if blocked:
                st.markdown("**Risk Guard Blocks:**")
                for b in blocked:
                    sym    = b.get("symbol","?") if isinstance(b, dict) else "?"
                    reason = b.get("reason","")  if isinstance(b, dict) else str(b)
                    st.markdown(f"- 🚫 `{sym}` — {reason}")
            reasoning = cyc.get("reasoning") or ""
            if reasoning:
                st.markdown("**LLM Reasoning:**")
                st.markdown(
                    f"<div style='background:#22211f;border-radius:8px;padding:1rem;"
                    f"font-size:.85rem;line-height:1.6;white-space:pre-wrap;"
                    f"max-height:300px;overflow-y:auto;color:#cdccca'>"
                    f"{str(reasoning)[:3000]}</div>",
                    unsafe_allow_html=True
                )
            mem = cyc.get("memory_packet_in")
            if mem:
                st.markdown("**Memory Packet:**")
                st.caption(str(mem))

# ── Risk guard summary ─────────────────────────────────────────────────────
st.markdown("### 🚫 Risk Guard Summary")
all_blocks = []
for _, row in day_dec[day_dec["arm_id"].isin(sel_arms)].iterrows():
    try:
        v      = row.get("orders_blocked")
        blocks = (json.loads(v) if isinstance(v, str) else v) or []
        for b in blocks:
            if isinstance(b, dict):
                all_blocks.append({
                    "Arm":    ARM_LABELS.get(row["arm_id"], row["arm_id"]),
                    "Time":   row["cycle_ts"].tz_convert("America/New_York").strftime("%H:%M ET"),
                    "Symbol": b.get("symbol","?"),
                    "Side":   b.get("side","?"),
                    "Qty":    b.get("qty","?"),
                    "Reason": b.get("reason","?"),
                })
    except Exception:
        pass
if all_blocks:
    st.dataframe(pd.DataFrame(all_blocks), use_container_width=True, hide_index=True)
else:
    st.success("No risk guard blocks on this date.")