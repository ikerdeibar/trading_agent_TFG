from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float,
    Integer, String, Text, JSON
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(10), nullable=False)
    side        = Column(String(4), nullable=False)   # BUY / SELL
    qty         = Column(Float, nullable=False)
    price       = Column(Float, nullable=False)
    order_id    = Column(String(64), unique=True)
    status      = Column(String(64), default="filled")
    agent       = Column(String(64))                  # which agent placed it
    arm_id      = Column(String(8),   nullable=True, index=True)

    reasoning   = Column(Text)                        # LLM rationale
    created_at  = Column(DateTime, default=datetime.utcnow)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(10), nullable=False)
    price       = Column(Float)
    volume      = Column(BigInteger)
    raw         = Column(JSON)                        # full API payload
    captured_at = Column(DateTime, default=datetime.utcnow)


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    agent       = Column(String(32), nullable=False)
    level       = Column(String(8), default="INFO")   # INFO / WARN / ERROR
    message     = Column(Text)
    payload     = Column(JSON)
    created_at  = Column(DateTime, default=datetime.utcnow)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    arm_id     = Column(String(8),  nullable=True, index=True)

    id              = Column(Integer, primary_key=True, autoincrement=True)
    cash            = Column(Float)
    equity          = Column(Float)
    total_value     = Column(Float)
    positions       = Column(JSON)                    # {symbol: qty}
    captured_at     = Column(DateTime, default=datetime.utcnow)


class AgentDecisionLog(Base):
    __tablename__ = "agent_decision_logs"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    arm_id               = Column(String(8),  nullable=False, index=True)
    cycle_ts             = Column(DateTime(timezone=True), nullable=False, index=True)
    agent                = Column(String(64),  nullable=False)

    # LLM output
    reasoning            = Column(Text,       nullable=True)
    orders_proposed      = Column(JSON,       nullable=True)  # all, incl. holds
    orders_executed      = Column(JSON,       nullable=True)
    orders_blocked       = Column(JSON,       nullable=True)  # {symbol, reason}

    # Context injected into the LLM
    news_context         = Column(JSON,       nullable=True)  # {sym: [{headline,url,ts}]}
    ta_context           = Column(JSON,       nullable=True)  # {sym: {rsi,sma,close}}
    portfolio_snapshot   = Column(JSON,       nullable=True)  # {equity,cash,positions}
    memory_packet_in     = Column(Text,       nullable=True)

    # Per-symbol scores (from LLM output)
    confidence_per_symbol = Column(JSON,      nullable=True)  # {sym: float}
    sentiment_per_symbol  = Column(JSON,      nullable=True)  # {sym: float} — future FinBERT

    # Cost tracking
    llm_tokens_used      = Column(Integer,    nullable=True)
    llm_cost_usd         = Column(Float,      nullable=True)

    # Post-mortem (written next cycle)
    post_mortem          = Column(JSON,       nullable=True)  # {sym: {entry_price, exit_price, pnl_pct}}
    post_mortem_ts       = Column(DateTime(timezone=True), nullable=True)

    created_at           = Column(DateTime(timezone=True),
                                  default=lambda: datetime.now(tz=timezone.utc))

class SessionMemory(Base):
    """Stores end-of-session distilled memory packet per arm."""
    __tablename__ = "session_memory"
    id       = Column(Integer, primary_key=True, autoincrement=True)
    arm_id   = Column(String(8), nullable=False, index=True)
    date     = Column(String(10), nullable=False)   # YYYY-MM-DD
    packet   = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc))

class SentimentSnapshot(Base):
    """One row per symbol per cycle — shared across all arms."""
    __tablename__ = "sentiment_snapshots"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    cycle_ts   = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol     = Column(String(8),  nullable=False, index=True)
    label      = Column(String(16), nullable=False)   # BULLISH / NEUTRAL / BEARISH
    score      = Column(Float,      nullable=False)   # [-1, 1]
    confidence = Column(Float,      nullable=False)   # [0, 1]
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc))

