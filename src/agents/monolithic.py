from __future__ import annotations
import logging
from src.agents.base import BaseAgent, parse_llm_output
from src.agents.prompts import monolithic_prompt
from src.agents.llm import call_llm
from src.data.context import context_to_prompt

logger = logging.getLogger(__name__)


class MonolithicAgent(BaseAgent):
    """Arms A and C — single LLM call, sequential reasoning in one pass."""

    def decide(self) -> dict:
        ctx        = self.get_context()
        ctx_str    = context_to_prompt(ctx)
        messages   = monolithic_prompt(ctx_str)
        raw, usage = call_llm(messages, agent_name="MonolithicAgent")
        result     = parse_llm_output(raw)
        result["tokens_used"] = usage["prompt_tokens"] + usage["completion_tokens"]
        result["cost_usd"]    = usage["estimated_cost_usd"]
        logger.info("MonolithicAgent orders: %s", result.get("orders", []))
        self.last_context = ctx
        return result
