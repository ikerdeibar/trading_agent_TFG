from __future__ import annotations
import logging
from sqlalchemy.orm import Session
from sqlalchemy import select
from src.storage.models import Trade, AgentLog
from src.agents.llm import call_llm
from src.core.config import get_config
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _get_todays_trades(session: Session) -> list[Trade]:
    today = datetime.now(tz=timezone.utc).date()
    return list(session.execute(
        select(Trade)
        .where(Trade.created_at >= datetime.combine(today, datetime.min.time()))
        .order_by(Trade.created_at)
    ).scalars().all())


def distil_session(session: Session, portfolio: dict) -> str:
    trades = _get_todays_trades(session)

    if not trades:
        return "No trades executed in the previous session."

    trade_summary = "\n".join([
        f"- {t.side} {t.symbol} x{t.qty} @ ${t.price:.2f} ({t.status})"
        for t in trades
    ])

    equity = portfolio.get("equity", 0)
    cash   = portfolio.get("cash", 0)

    prompt = [
        {"role": "system", "content": (
            "You are a memory distiller for a trading agent. "
            "Summarise the session in plain text, strictly under 200 tokens. "
            "Include: trades made, key reasoning, portfolio outcome. "
            "Be concise — this will be injected into the next session's context. "
            "Return a plain text summary (no JSON required)."
        )},
        {"role": "user", "content": (
            f"Session trades:\n{trade_summary}\n\n"
            f"End-of-session portfolio: equity=${equity:,.2f}, cash=${cash:,.2f}\n\n"
            "Distil into a plain text memory packet under 200 tokens. No json formatting needed, just concise prose."
        )},
    ]

    try:
        packet, _usage = call_llm(prompt, agent_name="Distiller", response_format=None)
        logger.info("Memory packet distilled (%d chars)", len(packet))
        return packet.strip()
    except Exception as e:
        logger.warning("Distillation failed: %s", e)
        return f"Previous session trades: {trade_summary}"
