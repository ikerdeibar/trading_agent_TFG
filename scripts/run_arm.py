#!/usr/bin/env python3
"""
Entrypoint for one full OODA cycle.
Called by Cloud Scheduler once per cycle during market hours.
Usage: ARM_ID=A python scripts/run_arm.py
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.storage.db import get_session_factory, create_tables
from src.loop.ooda import run_cycle
from src.memory.distiller import distil_session
from src.storage.repository import save_memory, load_memory
from src.execution.broker import get_portfolio
from src.core.config import get_config


def main():
    create_tables()
    cfg = get_config()
    SessionFactory = get_session_factory()
    session = SessionFactory()

    try:
        # Load memory from previous session
        memory_packet = load_memory(session, cfg.arm_id)
        logging.info("Memory loaded: %d chars", len(memory_packet))

        # Run OODA cycle
        summary = run_cycle(session, memory_packet)
        print(
            f"Cycle complete: ARM={summary['arm']} | "
            f"Executed={len(summary['executed'])} | "
            f"Equity=${summary['equity']:,.2f}"
        )

        # Distil and save memory for next session (runs every cycle,
        # overwrites today's entry — last cycle of the day wins)
        portfolio = get_portfolio()
        new_packet = distil_session(session, portfolio)
        save_memory(session, cfg.arm_id, new_packet)
        logging.info("Memory saved: %d chars", len(new_packet))

    except Exception as e:
        logging.error("Cycle failed: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
