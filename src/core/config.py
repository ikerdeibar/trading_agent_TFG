# src/core/config.py
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env on import
load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]   # project root


# ── Pydantic models for config sections ───────────────────────────────────────

class LLMConfig(BaseModel):
    provider:    str
    model:       str
    temperature: float = 0.1
    max_tokens:  int   = 1024

class ArmConfig(BaseModel):
    name:         str
    architecture: str           # "monolithic" | "multi_agent"
    llm:          LLMConfig

class EmbeddingConfig(BaseModel):
    provider: str
    model:    str

class RiskConfig(BaseModel):
    max_position_pct:       float = 5.0
    market_open_et:         str   = "09:45"
    market_close_et:        str   = "15:45"

class RAGConfig(BaseModel):
    chunk_size_tokens:    int = 300
    chunk_overlap_tokens: int = 50
    top_k_chunks:         int = 5

class ScheduleConfig(BaseModel):
    cycle_interval_minutes: int = 65
    session_start_et:       str = "09:15"
    session_end_et:         str = "15:45"

class UniverseConfig(BaseModel):
    tickers:   list[str]
    benchmark: str = "SPY"

class MemoryConfig(BaseModel):
    distilled_packet_max_tokens: int = 200


class AppConfig(BaseModel):
    """Single config object passed through the entire application."""
    arm_id:     str
    arm:        ArmConfig
    embeddings: EmbeddingConfig
    dev_llm:    LLMConfig
    universe:   UniverseConfig
    risk:       RiskConfig
    rag:        RAGConfig
    schedule:   ScheduleConfig
    memory:     MemoryConfig

    # Resolved API keys (from env — never from yaml)
    openai_api_key:      str = ""
    openrouter_api_key:  str = ""

    # Arm-specific Alpaca keys
    alpaca_key_a:     str = ""
    alpaca_secret_a:  str = ""
    alpaca_key_b:     str = ""
    alpaca_secret_b:  str = ""
    alpaca_key_c:     str = ""
    alpaca_secret_c:  str = ""
    alpaca_key_d:     str = ""
    alpaca_secret_d:  str = ""
    alpaca_paper:     bool = True
    alpaca_base_url:  str = "https://paper-api.alpaca.markets/v2"

    # Convenience — resolved at load time from ARM_ID
    alpaca_api_key:      str = ""
    alpaca_secret_key:   str = ""
    
    database_url:        str = ""


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Load and return the application config.
    Cached after first call — call reload_config() in tests to reset.
    """
    base  = _load_yaml(_ROOT / "configs" / "base.yaml")
    exps  = _load_yaml(_ROOT / "configs" / "experiments.yaml")

    arm_id = os.environ.get("ARM_ID", "C").upper()
    if arm_id not in exps["arms"]:
        raise ValueError(
            f"ARM_ID='{arm_id}' not found in experiments.yaml. "
            f"Valid values: {list(exps['arms'].keys())}"
        )

    return AppConfig(
        arm_id     = arm_id,
        arm        = ArmConfig(**exps["arms"][arm_id]),
        embeddings = EmbeddingConfig(**exps["embeddings"]),
        dev_llm    = LLMConfig(**exps["dev"]),
        universe   = UniverseConfig(**base["universe"]),
        risk       = RiskConfig(**base["risk"]),
        rag        = RAGConfig(**base["rag"]),
        schedule   = ScheduleConfig(**base["schedule"]),
        memory     = MemoryConfig(**base["memory"]),
        # Keys from environment only
        openai_api_key     = os.getenv("OPENAI_API_KEY", ""),
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", ""),
        
        alpaca_key_a      = os.getenv("ALPACA_KEY_A", ""),
        alpaca_secret_a   = os.getenv("ALPACA_SECRET_A", ""),
        alpaca_key_b      = os.getenv("ALPACA_KEY_B", ""),
        alpaca_secret_b   = os.getenv("ALPACA_SECRET_B", ""),
        alpaca_key_c      = os.getenv("ALPACA_KEY_C", ""),
        alpaca_secret_c   = os.getenv("ALPACA_SECRET_C", ""),
        alpaca_key_d      = os.getenv("ALPACA_KEY_D", ""),
        alpaca_secret_d   = os.getenv("ALPACA_SECRET_D", ""),
        alpaca_paper      = os.getenv("ALPACA_PAPER", "true").lower() == "true",
        alpaca_base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"),
        # Active arm keys resolved from ARM_ID
        alpaca_api_key    = os.getenv(f"ALPACA_KEY_{os.getenv('ARM_ID', 'A').upper()}", ""),
        alpaca_secret_key = os.getenv(f"ALPACA_SECRET_{os.getenv('ARM_ID', 'A').upper()}", ""),
    )


def reload_config() -> AppConfig:
    """Clear cache and reload — use in tests when ARM_ID changes."""
    get_config.cache_clear()
    return get_config()
