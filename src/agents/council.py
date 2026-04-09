from __future__ import annotations
import logging
from src.agents.base import BaseAgent, parse_llm_output
from src.agents.prompts import (
    news_analyst_prompt, technical_analyst_prompt,
    trader_prompt, risk_manager_prompt, executor_prompt,
)
from src.agents.llm import call_llm
from src.data.context import context_to_prompt

logger = logging.getLogger(__name__)


class CouncilAgent(BaseAgent):
    """Arms B and D — 5-agent sequential council."""

    def decide(self) -> dict:
        ctx     = self.get_context()
        ctx_str = context_to_prompt(ctx)
        chain   = []   # accumulates all agent outputs

        # 1. News Analyst
        out1, u1 = call_llm(news_analyst_prompt(ctx_str), "NewsAnalyst")
        chain.append(f"[NewsAnalyst]\n{out1}")
        logger.info("NewsAnalyst done (%d chars)", len(out1))

        # 2. Technical Analyst
        out2, u2 = call_llm(technical_analyst_prompt(ctx_str, out1), "TechnicalAnalyst")
        chain.append(f"[TechnicalAnalyst]\n{out2}")
        logger.info("TechnicalAnalyst done (%d chars)", len(out2))

        # 3. Trader/Synthesiser (contrarian — challenges prior analysis)
        prior = "\n\n".join(chain)
        out3, u3 = call_llm(trader_prompt(ctx_str, prior), "TraderSynthesiser")
        chain.append(f"[TraderSynthesiser]\n{out3}")
        logger.info("TraderSynthesiser done (%d chars)", len(out3))

        # 4. Risk Manager
        out4, u4 = call_llm(risk_manager_prompt(ctx_str, "\n\n".join(chain), ctx["portfolio"]), "RiskManager")
        chain.append(f"[RiskManager]\n{out4}")
        logger.info("RiskManager done (%d chars)", len(out4))

        # 5. Executor — produces final JSON
        out5, u5 = call_llm(executor_prompt(ctx_str, "\n\n".join(chain)), "Executor")
        result = parse_llm_output(out5)
        logger.info("Executor orders: %s", result.get("orders", []))

        # Attach full chain to reasoning for audit trail
        result["council_chain"] = chain
        total_tokens = sum(u["prompt_tokens"] + u["completion_tokens"] for u in [u1,u2,u3,u4,u5])
        total_cost   = sum(u["estimated_cost_usd"] for u in [u1,u2,u3,u4,u5])
        result["tokens_used"] = total_tokens
        result["cost_usd"]    = total_cost
        self.last_context = ctx
        return result
