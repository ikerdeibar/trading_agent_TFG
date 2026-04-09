from src.agents.monolithic import MonolithicAgent
from src.agents.council import CouncilAgent
from src.core.config import get_config


def get_agent(session, memory_packet: str = ""):
    """Return the correct agent class for the active ARM_ID."""
    cfg  = get_config()
    arch = cfg.arm.architecture
    if arch == "monolithic":
        return MonolithicAgent(session, memory_packet)
    elif arch == "multi_agent":
        return CouncilAgent(session, memory_packet)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

__all__ = ["MonolithicAgent", "CouncilAgent", "get_agent"]
