"""Methodology — design, pipeline, judge layer, reproducibility checklist."""
from __future__ import annotations

import pathlib
import streamlit as st

from db import render_sidebar_about, render_takeaway

st.set_page_config(page_title="Methodology", page_icon="📐", layout="wide")
render_sidebar_about()

st.markdown("## Methodology")
st.caption(
    "How the experiment was designed and run, end-to-end. The same eight-layer pipeline "
    "and the same Judge layer drive every arm — only the agent layer changes between cells."
)

FIG = pathlib.Path(__file__).resolve().parents[2] / "presentation" / "figures"


def _img(name: str, *, width: int | None = None, caption: str | None = None) -> None:
    p = FIG / name
    if p.exists():
        if width is None:
            st.image(str(p), caption=caption, use_container_width=True)
        else:
            st.image(str(p), width=width, caption=caption)
    else:
        st.info(f"Diagram `{name}` not found at `{p}`. Run `python presentation/generate_figures.py`.")


# ═══════════════════════════════════════════════════════════════════════════
# 1 — 2×2 design
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### The 2×2 factorial design")
st.markdown(
    "Two binary factors are crossed: the LLM (Qwen 235B vs GPT-4.1) and the agent topology "
    "(Monolithic, single LLM call vs Council, five sequential agents). Every other moving part "
    "— prompts schema, news source, sentiment model, technical features, Judge layer, broker, "
    "10-asset universe, $100k starting equity — is held constant."
)
_img("diag_factorial_cells.png", caption="The four arms of the experiment.")


# ═══════════════════════════════════════════════════════════════════════════
# 2 — 8-layer pipeline
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### The 8-layer neuro-symbolic pipeline")
st.markdown(
    "All four arms walk this same pipeline every OODA cycle. The only swap point is layer 5 "
    "(the agent layer). Layers 1–4 build a shared context block; layer 6 is the OODA loop; "
    "layer 7 is the deterministic Judge; layer 8 is the broker + audit log."
)
_img("diag_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3 — Mono vs Council
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Monolithic vs Council — the only variation")
_img("diag_mono_vs_council.png")
st.markdown(
    "- **Monolithic** — one LLM call sees the whole context and emits one JSON output. "
    "Cheap, fast, prone to single-point reasoning errors.\n"
    "- **Council** — five sequential agents (News Analyst → Technical Analyst → "
    "Trader [contrarian] → Risk Manager → Executor) share a chain of thought "
    "and emit one JSON output. The contrarian Trader is *deliberately* designed to "
    "challenge the prior reasoning to suppress sycophancy cascades."
)


# ═══════════════════════════════════════════════════════════════════════════
# 4 — Judge layer
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### The Judge layer — five deterministic gates")
_img("diag_judge_gates.png")
st.markdown(
    "Pure-Python rule engine, identical across all four arms. Every LLM proposal must clear "
    "all five gates before reaching the broker:\n\n"
    "1. **Schema** — BUY/SELL with size_pct > 0 and non-empty reasoning.\n"
    "2. **Whitelist** — ticker in the 10-asset basket.\n"
    "3. **Position cap** — resulting weight ≤ 5% of total equity.\n"
    "4. **Market hours** — 09:45 – 15:45 ET.\n"
    "5. **Cycle cap** — at most 7 OODA cycles per session.\n\n"
    "**LLM proposes, Python disposes.**"
)


# ═══════════════════════════════════════════════════════════════════════════
# 5 — Evaluation framework
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Four-dimensional evaluation framework")
_img("diag_eval_framework.png")
st.markdown(
    "- **Risk-adjusted** — Sharpe, MDD, cumulative return, win rate.\n"
    "- **Business viability** — Return on Intelligence, $/bp, total cost.\n"
    "- **Reliability** — Judge block rate, action rate α, schema violations.\n"
    "- **Behavioural** — trade concentration, sentiment alignment, agent disagreement."
)


# ═══════════════════════════════════════════════════════════════════════════
# 6 — Reproducibility checklist
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Reproducibility checklist")
items = [
    ("Code repository",         "All source under `src/`, dashboard under `dashboard/`."),
    ("Configuration",           "`configs/base.yaml` + `configs/experiments.yaml` define every arm."),
    ("Per-arm execution",       "`scripts/run_arm.py` + `ARM_ID=A|B|C|D` env var."),
    ("Frozen results",          "Parquet files in `dashboard/data/` with a `manifest.json` timestamp."),
    ("Audit log",               "Every cycle's prompt, response, decisions and blocks live in `agent_decision_logs`."),
    ("Live broker",             "Alpaca paper-trading API · 16 NYSE sessions · $100k start."),
    ("Sentiment",               "FinBERT (ProsusAI/finbert) on-device · zero network cost."),
    ("Universe",                "10-ticker liquid basket: NVDA, AAPL, MSFT, GOOGL, AMZN, JPM, XOM, LLY, CAT, NEE."),
    ("Benchmark",               "SPY buy-and-hold over the same window."),
]
for k, v in items:
    st.markdown(
        f"<div style='display:flex;gap:.8rem;padding:.45rem 0;border-bottom:1px solid #262523'>"
        f"<div style='min-width:170px;color:#22d3ee;font-weight:600;font-size:.85rem'>{k}</div>"
        f"<div style='color:#cdccca;font-size:.85rem'>{v}</div></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7 — Literature gap
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Where this thesis sits in the literature")
_img("diag_literature_gap.png")

render_takeaway(
    "The experiment is designed as a <b>controlled isolation rig</b>: hold every other moving part "
    "constant and vary just one knob at a time. That's what lets the dashboard report a model effect "
    "and an architecture effect as separate, attributable numbers — instead of the usual "
    "confounded comparison."
)
