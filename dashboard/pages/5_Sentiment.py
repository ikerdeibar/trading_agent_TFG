"""Sentiment — FinBERT signal per ticker and its predictive value."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db import (
    load_sentiment_snapshots, load_market_snapshots,
    PLOT_LAYOUT, AXIS_STYLE, LEGEND_STYLE,
    COLOR_POS, COLOR_NEG,
    render_sidebar_about, render_takeaway,
)

st.set_page_config(page_title="Sentiment", page_icon="🧠", layout="wide")
render_sidebar_about()

st.markdown("## FinBERT Sentiment")
st.caption(
    "All four arms share the same on-device FinBERT signal — one reading per ticker per cycle. "
    "Score is a signed [-1, 1] number; labels are BULLISH / NEUTRAL / BEARISH."
)

sent_df   = load_sentiment_snapshots()
market_df = load_market_snapshots()
if sent_df.empty:
    st.warning("No sentiment data found.")
    st.stop()

TICKERS      = sorted(sent_df["symbol"].unique())
LABEL_COLORS = {"BULLISH": COLOR_POS, "NEUTRAL": "#7a7974", "BEARISH": COLOR_NEG}


# ═══════════════════════════════════════════════════════════════════════════
# 1 — Daily heatmap (date × ticker)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Daily average sentiment score · date × ticker")

daily_avg = (
    sent_df.groupby(["date", "symbol"])["score"].mean()
    .reset_index().pivot(index="date", columns="symbol", values="score")
)
n_per_cell = (
    sent_df.groupby(["date", "symbol"])["score"].size()
    .reset_index(name="n").pivot(index="date", columns="symbol", values="n")
    .reindex_like(daily_avg).fillna(0).astype(int)
)

fig_hm = go.Figure(go.Heatmap(
    z=daily_avg.values,
    x=daily_avg.columns.tolist(),
    y=[str(d) for d in daily_avg.index],
    colorscale=[[0, COLOR_NEG], [0.5, "#1c1b19"], [1, COLOR_POS]],
    zmid=0, zmin=-1, zmax=1,
    customdata=n_per_cell.values,
    colorbar=dict(title="Score", tickfont=dict(color="#cdccca")),
    hovertemplate=("Date: %{y}<br>Ticker: %{x}<br>"
                   "Avg score: %{z:+.3f}<br>News reads aggregated (n): %{customdata}<extra></extra>"),
    xgap=2, ygap=2,
))
fig_hm.update_layout(
    **PLOT_LAYOUT,
    height=max(320, len(daily_avg) * 26 + 100),
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(side="top", **AXIS_STYLE),
    yaxis=dict(**AXIS_STYLE),
)
st.plotly_chart(fig_hm, use_container_width=True)

with st.expander("How to read this"):
    st.markdown(
        "- **Green = bullish**, **pink = bearish**, **dark = neutral**.\n"
        "- Hover any cell for **ticker × date × average score** and **n** (number of FinBERT readings rolled into that cell).\n"
        "- Vertical green/pink streaks = a ticker had a clear directional read for several days in a row."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2 — Label distribution per ticker
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Label distribution per ticker")
label_counts = sent_df.groupby(["symbol", "label"]).size().reset_index(name="count")
totals = label_counts.groupby("symbol")["count"].sum().sort_values(ascending=False)
sym_order = totals.index.tolist()

fig_ld = go.Figure()
for label, color in LABEL_COLORS.items():
    lc = label_counts[label_counts["label"] == label].set_index("symbol").reindex(sym_order)["count"].fillna(0)
    fig_ld.add_trace(go.Bar(name=label, x=sym_order, y=lc.values,
                            marker_color=color))
fig_ld.update_layout(
    **PLOT_LAYOUT, barmode="stack",
    height=320, margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(title="Cycle count", title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(**AXIS_STYLE),
    legend=LEGEND_STYLE,
)
st.plotly_chart(fig_ld, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3 — Ticker drill-down
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Ticker drill-down")
selected = st.selectbox("Pick a ticker", TICKERS, index=TICKERS.index("XOM") if "XOM" in TICKERS else 0)

tk_daily = (
    sent_df[sent_df["symbol"] == selected]
    .groupby("date")["score"].mean().reset_index()
    .sort_values("date")
)
fig_tk = go.Figure()
fig_tk.add_trace(go.Scatter(
    x=tk_daily["date"].astype(str), y=tk_daily["score"],
    mode="lines+markers",
    line=dict(color="#22d3ee", width=2.6),
    marker=dict(
        size=9,
        color=tk_daily["score"].apply(
            lambda s: COLOR_POS if s > 0.1 else (COLOR_NEG if s < -0.1 else "#7a7974")
        ),
        line=dict(color="#171614", width=1),
    ),
    hovertemplate="%{x}<br>Avg score: %{y:+.3f}<extra></extra>",
))
fig_tk.add_hline(y=0, line_dash="dot", line_color="#393836")
fig_tk.update_layout(
    **PLOT_LAYOUT, height=380, margin=dict(l=10, r=10, t=20, b=80),
    yaxis=dict(range=[-1, 1], title=f"Avg sentiment score · {selected}",
               title_font=dict(color="#7a7974"), **AXIS_STYLE),
    xaxis=dict(tickangle=-35, automargin=True, **AXIS_STYLE),
)
if not tk_daily.empty:
    fig_tk.update_xaxes(
        range=[str(tk_daily["date"].min()), str(tk_daily["date"].max())],
        autorange=False,
    )
st.plotly_chart(fig_tk, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4 — Sentiment → next-day return correlation
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Sentiment → next-day return correlation")
st.caption(
    "For each ticker/day, compare today's mean sentiment score to the ticker's *next* trading "
    "day's mid-price return. Positive r = bullish sentiment tends to precede positive returns."
)

if market_df.empty:
    st.info("No market data found — cannot compute correlations.")
else:
    px_daily = (
        market_df.sort_values("captured_at")
        .groupby(["date", "symbol"])["mid"].last()
        .reset_index().sort_values(["symbol", "date"])
    )
    px_daily["next_mid"] = px_daily.groupby("symbol")["mid"].shift(-1)
    px_daily["next_ret"] = (px_daily["next_mid"] / px_daily["mid"] - 1) * 100
    sent_daily = (
        sent_df.groupby(["date", "symbol"])["score"]
        .mean().reset_index(name="sent_score")
    )
    merged = (sent_daily.merge(px_daily[["date", "symbol", "next_ret"]],
                               on=["date", "symbol"], how="inner")
              .dropna(subset=["sent_score", "next_ret"]))

    if merged.empty:
        st.info("No overlap between sentiment and next-day returns.")
    else:
        corr_data = []
        for sym in sorted(merged["symbol"].unique()):
            g = merged[merged["symbol"] == sym]
            n = len(g)
            r = g["sent_score"].corr(g["next_ret"]) if n >= 3 else None
            corr_data.append({"symbol": sym, "n": n, "r": r})
        cdf = pd.DataFrame(corr_data).sort_values("r", na_position="last")
        pooled = merged["sent_score"].corr(merged["next_ret"])

        c1, c2 = st.columns([2, 1])
        with c1:
            colors = [COLOR_POS if (v is not None and v >= 0) else COLOR_NEG for v in cdf["r"]]
            fig_corr = go.Figure(go.Bar(
                x=cdf["symbol"], y=cdf["r"].fillna(0),
                marker_color=colors,
                text=[("—" if pd.isna(v) else f"{v:+.2f}") for v in cdf["r"]],
                textposition="outside", textfont=dict(color="#cdccca"),
            ))
            fig_corr.add_hline(y=0, line_dash="dot", line_color="#393836")
            fig_corr.update_layout(
                **PLOT_LAYOUT, height=360, margin=dict(l=10, r=10, t=20, b=70),
                yaxis=dict(title="Pearson r", range=[-1, 1],
                           title_font=dict(color="#7a7974"), **AXIS_STYLE),
                xaxis=dict(tickangle=-25, automargin=True, **AXIS_STYLE),
                showlegend=False,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with c2:
            st.metric("Pooled correlation",
                      f"{pooled:+.3f}" if pd.notna(pooled) else "—")
            st.caption("Pooled across all tickers and days.")
            st.dataframe(
                cdf.rename(columns={"symbol": "Ticker", "n": "n", "r": "r"})
                   .assign(r=lambda d: d["r"].map(lambda v: "—" if pd.isna(v) else f"{v:+.3f}")),
                use_container_width=True, hide_index=True,
            )

        st.markdown(f"#### {selected} — sentiment vs next-day return scatter")
        dsel = merged[merged["symbol"] == selected].copy()
        if dsel.empty:
            st.info(f"No paired data for {selected}.")
        else:
            fig_sc = go.Figure(go.Scatter(
                x=dsel["sent_score"], y=dsel["next_ret"],
                mode="markers",
                marker=dict(size=10, color="#22d3ee",
                            line=dict(color="#171614", width=1)),
                text=dsel["date"].astype(str),
                hovertemplate="%{text}<br>Sent: %{x:+.3f}<br>Next-day: %{y:+.2f}%<extra></extra>",
            ))
            fig_sc.add_hline(y=0, line_dash="dot", line_color="#393836")
            fig_sc.add_vline(x=0, line_dash="dot", line_color="#393836")
            fig_sc.update_layout(
                **PLOT_LAYOUT, height=380, margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(title="Sentiment score (t)", range=[-1, 1],
                           title_font=dict(color="#7a7974"), **AXIS_STYLE),
                yaxis=dict(title="Next-day return % (t→t+1)", ticksuffix="%",
                           title_font=dict(color="#7a7974"), **AXIS_STYLE),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

with st.expander("Raw sentiment data (all readings)"):
    st.dataframe(sent_df.sort_values(["date", "symbol"]),
                 use_container_width=True, hide_index=True)

render_takeaway(
    "FinBERT supplies a <b>shared, deterministic</b> sentiment signal — its job is to remove a "
    "confounder, not to predict returns. The pooled correlation hovers near zero with high noise: "
    "treat sentiment as a <i>contextual feature</i> the LLM can choose to weigh, not as a strategy on "
    "its own."
)
