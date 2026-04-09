from __future__ import annotations
from src.agents.base import SYSTEM_PROMPT_BASE, OUTPUT_SCHEMA


def monolithic_prompt(context_str: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BASE + f"\nOutput schema:\n{OUTPUT_SCHEMA}"},
        {"role": "user",   "content": (
            "Analyse the following market context and reason sequentially through:\n"
            "1. News signals per ticker\n"
            "2. Technical indicator signals (RSI, MACD, MA crossovers)\n"
            "3. Portfolio risk and position sizing\n"
            "4. Final order proposal\n\n"
            f"Context:\n{context_str}"
        )},
    ]


def news_analyst_prompt(context_str: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You are a financial news analyst. Summarise the sentiment and key signals from the news for each ticker. "
            "Be concise and factual. Respond in JSON format."
        )},
        {"role": "user",   "content": f"News context:\n{context_str}"},
    ]


def technical_analyst_prompt(context_str: str, news_output: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You are a technical analyst. Interpret RSI, MACD, and moving average signals for each ticker. "
            "Respond in JSON format."
        )},
        {"role": "user",   "content": f"Market context:\n{context_str}\n\nNews analyst output:\n{news_output}"},
    ]


def trader_prompt(context_str: str, prior_outputs: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You are a contrarian trader/synthesiser. Your job is to CHALLENGE the prior analysis, not endorse it. "
            "Identify weaknesses, alternative interpretations, and propose a preliminary order with rationale. "
            "Be sceptical of consensus. Respond in JSON format."
        )},
        {"role": "user", "content": f"Context:\n{context_str}\n\nPrior analysis:\n{prior_outputs}"},
    ]


def risk_manager_prompt(context_str: str, prior_outputs: str, portfolio: dict) -> list[dict]:
    equity = portfolio.get("equity", 100000)
    return [
        {"role": "system", "content": (
            f"You are a risk manager. Total portfolio equity: ${equity:,.2f}. "
            f"Hard rule: no single position may exceed 5% of equity (${equity * 0.05:,.2f}). "
            "Review the proposed orders and flag or adjust any that violate risk limits. "
            "Output a risk-adjusted proposal. Respond in JSON format."
        )},
        {"role": "user", "content": f"Context:\n{context_str}\n\nPrior outputs:\n{prior_outputs}"},
    ]


def executor_prompt(context_str: str, prior_outputs: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You are the executor. Synthesise all prior agent outputs into a final trading decision. "
            f"Output ONLY valid JSON.\nSchema:\n{OUTPUT_SCHEMA}"
        )},
        {"role": "user", "content": f"Context:\n{context_str}\n\nAll prior agent outputs:\n{prior_outputs}"},
    ]
