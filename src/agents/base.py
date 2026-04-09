from __future__ import annotations
import json
import logging
from abc import ABC, abstractmethod
from typing import Any
from src.data.context import build_context, context_to_prompt
from src.core.config import get_config

logger = logging.getLogger(__name__)

# ── Output schema (identical for all arms) ────────────────────────────────────
OUTPUT_SCHEMA = """
{
  "reasoning": "<brief explanation of decision>",
  "orders": [
    {
      "symbol": "<TICKER>",
      "side": "buy" | "sell" | "hold",
      "qty": <integer shares>,
      "confidence": <0.0-1.0>
    }
  ]
}
Respond ONLY with valid JSON matching this schema. No markdown, no commentary.
"""

SYSTEM_PROMPT_BASE = """You are an autonomous equity trading agent operating on NYSE paper trading.
You will receive a context block containing portfolio state, technical indicators, recent news, and memory from prior sessions.
Your task is to analyse the data and output trading decisions in structured JSON.
Rules:
- Never allocate more than 5% of total equity to a single position
- Only trade tickers present in the context block
- If uncertain, output side=hold with qty=0
- Output must be valid JSON matching the schema exactly
"""


def parse_llm_output(raw: str) -> dict:
    """Extract and validate JSON from LLM response."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw)
        assert "orders" in parsed
        return parsed
    except Exception as e:
        logger.warning("Failed to parse LLM output: %s\nRaw: %s", e, raw[:200])
        return {"reasoning": "parse_error", "orders": []}


class BaseAgent(ABC):
    def __init__(self, session, memory_packet: str = ""):
        self.session       = session
        self.memory_packet = memory_packet
        self.cfg           = get_config()
        self.last_context   = {}

    def get_context(self) -> dict:
        return build_context(self.session, self.memory_packet)

    @abstractmethod
    def decide(self) -> dict:
        """Run full decision cycle. Returns parsed order dict."""
        ...
