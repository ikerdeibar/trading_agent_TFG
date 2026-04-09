from __future__ import annotations
from typing import Any
from datetime import datetime, timezone
from sqlalchemy import select, desc as sa_desc
from sqlalchemy.orm import Session
from src.storage.models import (
    Trade, MarketSnapshot, AgentLog,
    PortfolioSnapshot, AgentDecisionLog, SessionMemory
)


def log_trade(session: Session, **kwargs) -> Trade:
    obj = Trade(**kwargs)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return obj


def log_market(session: Session, **kwargs) -> MarketSnapshot:
    obj = MarketSnapshot(**kwargs)
    session.add(obj)
    session.commit()
    return obj


def log_agent(session: Session, agent: str, message: str,
              level: str = "INFO", payload: Any = None) -> AgentLog:
    obj = AgentLog(agent=agent, level=level, message=message, payload=payload)
    session.add(obj)
    session.commit()
    return obj


def log_portfolio(session: Session, arm_id: str = None, **kwargs) -> PortfolioSnapshot:
    obj = PortfolioSnapshot(arm_id=arm_id, **kwargs)
    session.add(obj)
    session.commit()
    return obj


def log_decision(
    session,
    *,
    arm_id: str,
    cycle_ts,
    agent: str,
    reasoning: str,
    orders_proposed: list,
    orders_executed: list,
    orders_blocked: list,
    news_context: dict,
    ta_context: dict,
    portfolio_snapshot: dict,
    memory_packet_in: str = "",
    llm_tokens_used: int = 0,
    llm_cost_usd: float = 0.0,
    sentiment_per_symbol: dict = None,
) -> AgentDecisionLog:
    record = AgentDecisionLog(
        arm_id=arm_id,
        cycle_ts=cycle_ts,
        agent=agent,
        reasoning=reasoning,
        orders_proposed=orders_proposed,
        orders_executed=orders_executed,
        orders_blocked=orders_blocked,
        news_context=news_context,
        ta_context=ta_context,
        portfolio_snapshot=portfolio_snapshot,
        memory_packet_in=memory_packet_in,
        confidence_per_symbol={
            o["symbol"]: o.get("confidence")
            for o in orders_proposed if o.get("symbol")
        },
        sentiment_per_symbol=sentiment_per_symbol or {},
        llm_tokens_used=llm_tokens_used,
        llm_cost_usd=llm_cost_usd,
    )
    session.add(record)
    session.commit()
    return record


def write_post_mortem(session, arm_id: str, current_prices: dict):
    """Look up the most recent un-assessed decision log and fill post_mortem."""
    prev = session.execute(
        select(AgentDecisionLog)
        .where(AgentDecisionLog.arm_id == arm_id)
        .where(AgentDecisionLog.post_mortem == None)
        .order_by(sa_desc(AgentDecisionLog.cycle_ts))
    ).scalars().first()

    if not prev or not prev.orders_executed:
        return

    pm = {}
    for order in prev.orders_executed:
        sym   = order.get("symbol")
        entry = order.get("price", 0)
        exit_ = current_prices.get(sym, 0)
        if entry and exit_:
            pm[sym] = {
                "entry_price": entry,
                "exit_price":  exit_,
                "pnl_pct":     round((exit_ - entry) / entry * 100, 4),
            }

    prev.post_mortem    = pm
    prev.post_mortem_ts = datetime.now(tz=timezone.utc)
    session.commit()


def save_memory(session, arm_id: str, packet: str):
    """Persist end-of-session memory packet for an arm."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    session.query(SessionMemory).filter_by(arm_id=arm_id, date=today).delete()
    session.add(SessionMemory(arm_id=arm_id, date=today, packet=packet))
    session.commit()


def load_memory(session, arm_id: str) -> str:
    """Load the most recent memory packet for an arm."""
    row = (
        session.query(SessionMemory)
        .filter_by(arm_id=arm_id)
        .order_by(sa_desc(SessionMemory.created_at))
        .first()
    )
    return row.packet if row else "No prior session memory available."
