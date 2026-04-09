# src/llm/client.py
from __future__ import annotations
import logging
import time
from typing import Optional
import httpx
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from src.core.config import get_config

logger = logging.getLogger(__name__)

# ── Cost table ($/1M tokens) ──────────────────────────────────────────────────
COST_RATES: dict[str, dict[str, float]] = {
    "gpt-4.1":                               {"input": 2.00,  "output": 8.00},
    "qwen/qwen3-235b-a22b-2507":             {"input": 0.071, "output": 0.10},
    "qwen/qwen3-embedding-8b":               {"input": 0.01,  "output": 0.00},
    "qwen/qwen3-next-80b-a3b-instruct:free": {"input": 0.00,  "output": 0.00},
}

# ── Client factory ────────────────────────────────────────────────────────────
_clients: dict[str, OpenAI] = {}

def _get_client(provider: str) -> OpenAI:
    """Lazy-init clients so env vars are available before first call."""
    global _clients
    if provider not in _clients:
        cfg = get_config()
        if provider == "openai":
            _clients["openai"] = OpenAI(
                api_key=cfg.openai_api_key,
                timeout=60.0,
            )
        elif provider == "openrouter":
            _clients["openrouter"] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=cfg.openrouter_api_key,
                timeout=60.0,
                default_headers={
                    "HTTP-Referer": "https://github.com/trading-agent-thesis",
                    "X-Title": "TradingAgent-Thesis",
                },
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _clients[provider]


# ── Cost helper ───────────────────────────────────────────────────────────────
def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_RATES.get(model, {"input": 0.0, "output": 0.0})
    return (
        prompt_tokens     * rates["input"] +
        completion_tokens * rates["output"]
    ) / 1_000_000


# ── 1. Chat / completion ──────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((httpx.TimeoutException, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def generate(
    system_prompt: str,
    user_prompt:   str,
    provider:      str,
    model:         str,
    temperature:   float = 0.1,
    max_tokens:    int   = 1024,
    arm_id:        Optional[str] = None,
    agent_role:    Optional[str] = None,
    cycle_id:      Optional[str] = None,
) -> dict:
    """
    Standard chat completion call used by all 4 arms.
    Always requests JSON output.
    Returns: {content: str, usage: {prompt_tokens, completion_tokens,
                                     estimated_cost_usd, latency_ms}}
    """
    t0     = time.perf_counter()
    client = _get_client(provider)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    latency_ms = round((time.perf_counter() - t0) * 1000)
    usage      = response.usage
    cost       = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)

    logger.info("llm_generate", extra={
        "arm_id":             arm_id,
        "agent_role":         agent_role,
        "cycle_id":           cycle_id,
        "provider":           provider,
        "model":              model,
        "prompt_tokens":      usage.prompt_tokens,
        "completion_tokens":  usage.completion_tokens,
        "estimated_cost_usd": cost,
        "latency_ms":         latency_ms,
    })

    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens":      usage.prompt_tokens,
            "completion_tokens":  usage.completion_tokens,
            "estimated_cost_usd": cost,
            "latency_ms":         latency_ms,
        },
    }


# ── 2. Embedding call ─────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((httpx.TimeoutException, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def embed(
    texts:    list[str],
    model:    str      = "qwen/qwen3-embedding-8b",
    cycle_id: Optional[str] = None,
) -> dict:
    """
    Embedding call for the RAG pipeline.
    Returns: {embeddings: list[list[float]], usage: {...}}
    Always routed through OpenRouter.
    """
    t0     = time.perf_counter()
    client = _get_client("openrouter")

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    latency_ms    = round((time.perf_counter() - t0) * 1000)
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
    cost          = estimate_cost(model, prompt_tokens, 0)

    logger.info("llm_embed", extra={
        "cycle_id":           cycle_id,
        "model":              model,
        "num_texts":          len(texts),
        "prompt_tokens":      prompt_tokens,
        "estimated_cost_usd": cost,
        "latency_ms":         latency_ms,
    })

    return {
        "embeddings": [item.embedding for item in response.data],
        "usage": {
            "prompt_tokens":      prompt_tokens,
            "estimated_cost_usd": cost,
            "latency_ms":         latency_ms,
        },
    }


# ── 3. Dev/test convenience wrapper ──────────────────────────────────────────
def generate_dev(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int = 512,
) -> dict:
    """
    Zero-cost wrapper for development and integration testing.
    Uses the free Qwen3-Next-80B model — identical interface to generate().
    Never use this for live paper-trading runs.
    """
    import time
    time.sleep(8)
    return generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="openrouter",
        model="qwen/qwen3-next-80b-a3b-instruct:free",
        temperature=0.1,
        max_tokens=max_tokens,
        arm_id="DEV",
        agent_role="dev_test",
    )
