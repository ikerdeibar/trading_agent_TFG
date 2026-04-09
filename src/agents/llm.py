from __future__ import annotations
import logging
from src.core.config import get_config

logger = logging.getLogger(__name__)

# Cost per 1M tokens (input, output)
COST_RATES = {
    "gpt-4.1":                        {"input": 2.00,  "output": 8.00},
    "qwen/qwen3-235b-a22b-2507":      {"input": 0.071, "output": 0.10},
    "qwen/qwen3-next-80b-a3b-instruct:free": {"input": 0.0, "output": 0.0},
}

def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_RATES.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


def call_llm(messages: list[dict], agent_name: str = "agent", response_format: dict | None = {"type": "json_object"}) -> tuple[str, dict]:
    """
    Call the LLM for the current arm.
    Returns (content_string, usage_dict) where usage_dict has:
      prompt_tokens, completion_tokens, estimated_cost_usd, latency_ms
    """
    import time
    from openai import OpenAI

    arm = get_config().arm
    provider = arm.llm.provider
    model = arm.llm.model

    if provider == "openai":
        import os
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        import os
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    logger.info("[%s] Calling %s via %s", agent_name, model, provider)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=arm.llm.temperature,
        max_tokens=arm.llm.max_tokens,
        **({"response_format": response_format} if response_format is not None else {}),
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    content = response.choices[0].message.content
    usage = response.usage
    prompt_tokens     = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)

    logger.info(
        "[%s] Response: %d chars | tokens: %d in / %d out | cost: $%.5f | latency: %dms",
        agent_name, len(content), prompt_tokens, completion_tokens, cost, latency_ms
    )

    return content, {
        "prompt_tokens":      prompt_tokens,
        "completion_tokens":  completion_tokens,
        "estimated_cost_usd": cost,
        "latency_ms":         latency_ms,
    }
