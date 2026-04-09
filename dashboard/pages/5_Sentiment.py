"""Sentiment Analysis — FinBERT scores per ticker over the experiment."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import load_sentiment_snapshots, load_market_snapshots

st.set_page_config(page_title="Sentiment", page_icon="🧠", layout="wide")
st.markdown("## FinBERT Sentiment Analysis")
st.caption("Shared sentiment signal across all arms. One reading per ticker per cycle.")

sent_df = load_sentiment_snapshots()
market_df = load_market_snapshots()

if sent_df.empty:
    st.warning("No sentiment data found.")
    st.stop()

TICKERS      = sorted(sent_df["symbol"].unique())
LABEL_COLORS = {"BULLISH": "#437a22", "NEUTRAL": "#7a7974", "BEARISH": "#a12c7b"}

# ── Daily average score heatmap (deduplicated) ─────────────────────────────
st.markdown("### Daily Average Sentiment Score")
st.caption("Score = average FinBERT signed score per ticker per day (range −1 to +1).")

daily_avg = (
    sent_df.groupby(["date", "symbol"])["score"]
    .mean()
    .reset_index()
    .pivot(index="date", columns="symbol", values="score")
)

fig = go.Figure(go.Heatmap(
    z=daily_avg.values,
    x=daily_avg.columns.tolist(),
    y=[str(d) for d in daily_avg.index],
    colorscale=[[0, "#a12c7b"], [0.5, "#1c1b19"], [1, "#437a22"]],
    zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title="Score"),
    hovertemplate="Date: %{y}<br>Ticker: %{x}<br>Avg Score: %{z:.3f}<extra></extra>",
))
fig.update_layout(
    height=max(300, len(daily_avg) * 28 + 80),
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="#171614",
    font=dict(family="sans-serif", color="#cdccca", size=11),
    xaxis=dict(side="top"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Label distribution ──────────────────────────────────────────────────────
st.markdown("### Sentiment Label Distribution (all cycles)")
label_counts = sent_df.groupby(["symbol","label"]).size().reset_index(name="count")
fig2 = go.Figure()
for label, color in LABEL_COLORS.items():
    lc = label_counts[label_counts["label"] == label]
    fig2.add_trace(go.Bar(name=label, x=lc["symbol"], y=lc["count"],
                          marker_color=color))
fig2.update_layout(
    barmode="stack", height=300, margin=dict(l=0,r=0,t=10,b=0),
    plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
    font=dict(family="sans-serif", color="#cdccca", size=12),
    yaxis=dict(title="Cycle Count", gridcolor="#262523"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Daily average score over time per ticker ────────────────────────────────
st.markdown("### Daily Average Score — Ticker Drill-Down")
st.caption("Each point is the daily average across all cycles for that ticker.")
selected = st.selectbox("Select ticker", TICKERS)

tk_daily = (
    sent_df[sent_df["symbol"] == selected]
    .groupby("date")["score"]
    .mean()
    .reset_index()
    .sort_values("date")
)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=tk_daily["date"].astype(str), y=tk_daily["score"],
    mode="lines+markers",
    line=dict(color="#01696f", width=2.5),
    marker=dict(
        size=8,
        color=tk_daily["score"].apply(
            lambda s: "#437a22" if s > 0.1 else ("#a12c7b" if s < -0.1 else "#7a7974")
        ),
        line=dict(color="#cdccca", width=1),
    ),
    hovertemplate="%{x}<br>Avg Score: %{y:.3f}<extra></extra>",
))
fig3.add_hline(y=0, line_dash="dot", line_color="#393836")
st.markdown(f"#### {selected} — Daily Avg Sentiment Score")
fig3.update_layout(
    height=420, margin=dict(l=0, r=0, t=20, b=80),
    plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
    font=dict(family="sans-serif", color="#cdccca", size=12),
    yaxis=dict(range=[-1,1], gridcolor="#262523", title="Score"),
    xaxis=dict(
        gridcolor="#262523",
        tickangle=-35,
        nticks=12,
        automargin=True,
        tickfont=dict(size=10),
    ),
)
st.plotly_chart(fig3, use_container_width=True)

# ── Sentiment → Next-Day Return Correlation ────────────────────────────────
st.markdown("### Sentiment vs. Next-Day Return")
st.caption(
    "For each ticker/day, we compare today's average sentiment score to the ticker's "
    "next trading day's return. Positive correlation means more bullish sentiment tends "
    "to be followed by positive next-day returns."
)

if market_df.empty:
    st.info("No market snapshots found, so sentiment-return correlation cannot be computed.")
else:
    px_daily = (
        market_df.sort_values("captured_at")
        .groupby(["date", "symbol"])["mid"]
        .last()
        .reset_index()
        .sort_values(["symbol", "date"])
    )
    px_daily["next_mid"] = px_daily.groupby("symbol")["mid"].shift(-1)
    px_daily["next_day_return_pct"] = (px_daily["next_mid"] / px_daily["mid"] - 1) * 100

    sent_daily = (
        sent_df.groupby(["date", "symbol"])["score"]
        .mean()
        .reset_index(name="sent_score")
    )

    merged = sent_daily.merge(
        px_daily[["date", "symbol", "next_day_return_pct"]],
        on=["date", "symbol"],
        how="inner",
    ).dropna(subset=["sent_score", "next_day_return_pct"])

    if merged.empty:
        st.info("No overlapping sentiment and next-day return observations were found.")
    else:
        corr_rows = []
        for sym, grp in merged.groupby("symbol"):
            n = len(grp)
            corr = grp["sent_score"].corr(grp["next_day_return_pct"]) if n >= 3 else None
            corr_rows.append({"symbol": sym, "n_obs": n, "corr": corr})
        corr_df = pd.DataFrame(corr_rows).sort_values("symbol")
        pooled_corr = merged["sent_score"].corr(merged["next_day_return_pct"])

        c1, c2 = st.columns([2, 1])
        with c1:
            fig4 = go.Figure(go.Bar(
                x=corr_df["symbol"],
                y=corr_df["corr"].fillna(0),
                marker_color=[
                    "#437a22" if (v is not None and v >= 0) else "#a12c7b"
                    for v in corr_df["corr"].tolist()
                ],
                text=[("—" if pd.isna(v) else f"{v:.2f}") for v in corr_df["corr"]],
                textposition="outside",
                hovertemplate="Ticker: %{x}<br>Correlation: %{y:.3f}<extra></extra>",
            ))
            fig4.add_hline(y=0, line_dash="dot", line_color="#393836")
            fig4.update_layout(
                height=340, margin=dict(l=0, r=0, t=20, b=70),
                plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
                font=dict(family="sans-serif", color="#cdccca", size=12),
                yaxis=dict(title="Pearson r", range=[-1, 1], gridcolor="#262523"),
                xaxis=dict(gridcolor="#262523", tickangle=-25, automargin=True),
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True)

        with c2:
            st.metric("Pooled Correlation (all tickers)", f"{pooled_corr:.3f}" if pd.notna(pooled_corr) else "—")
            st.caption(
                "Use this as directional evidence only. Small samples and short windows "
                "can make daily correlation unstable."
            )
            st.dataframe(
                corr_df.rename(columns={"symbol": "Ticker", "n_obs": "Observations", "corr": "Correlation"})
                .assign(Correlation=lambda d: d["Correlation"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Ticker Drill-Down: Sentiment vs Next-Day Return")
        dsel = merged[merged["symbol"] == selected].copy()
        if dsel.empty:
            st.info(f"No overlap data available for {selected}.")
        else:
            fig5 = go.Figure(go.Scatter(
                x=dsel["sent_score"],
                y=dsel["next_day_return_pct"],
                mode="markers",
                marker=dict(size=8, color="#5a8ecf", line=dict(color="#cdccca", width=0.8)),
                text=dsel["date"].astype(str),
                hovertemplate="Date: %{text}<br>Sentiment: %{x:.3f}<br>Next-day return: %{y:.3f}%<extra></extra>",
            ))
            fig5.add_hline(y=0, line_dash="dot", line_color="#393836")
            fig5.add_vline(x=0, line_dash="dot", line_color="#393836")
            st.markdown(f"#### {selected}: sentiment signal vs. next-day return")
            fig5.update_layout(
                height=430, margin=dict(l=0, r=0, t=20, b=50),
                plot_bgcolor="#1c1b19", paper_bgcolor="#171614",
                font=dict(family="sans-serif", color="#cdccca", size=12),
                xaxis=dict(title="Sentiment score (t)", range=[-1, 1], gridcolor="#262523"),
                yaxis=dict(
                    title="Next-day return % (t→t+1)",
                    gridcolor="#262523",
                    tickformat=".2f",
                    automargin=True,
                    tickfont=dict(size=10),
                ),
            )
            st.plotly_chart(fig5, use_container_width=True)

with st.expander("Raw sentiment data (all readings)"):
    st.dataframe(sent_df.sort_values(["date","symbol"]),
                 use_container_width=True, hide_index=True)