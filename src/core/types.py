# src/core/types.py
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class Action(str, Enum):
    BUY     = "BUY"
    SELL    = "SELL"
    HOLD    = "HOLD"
    MONITOR = "MONITOR"

class SentimentLabel(str, Enum):
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"

class JudgeOutcome(str, Enum):
    APPROVED = "APPROVED"
    VETOED   = "VETOED"

class VetoReason(str, Enum):
    SCHEMA_INVALID   = "SCHEMA_INVALID"
    TICKER_NOT_IN_WHITELIST = "TICKER_NOT_IN_WHITELIST"
    POSITION_CAP_EXCEEDED   = "POSITION_CAP_EXCEEDED"
    MARKET_CLOSED    = "MARKET_CLOSED"
    CYCLE_CAP_EXCEEDED = "CYCLE_CAP_EXCEEDED"

class Architecture(str, Enum):
    MONOLITHIC   = "monolithic"
    MULTI_AGENT  = "multi_agent"

class ArmID(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    DEV = "DEV"


# ── Market data ───────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    ticker:    str
    timestamp: datetime
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float

class PositionSnapshot(BaseModel):
    ticker:          str
    qty:             float
    market_value:    float
    unrealized_pnl:  float
    weight_pct:      float     # current % of total portfolio

class PortfolioState(BaseModel):
    timestamp:       datetime
    cash:            float
    total_equity:    float
    positions:       list[PositionSnapshot] = Field(default_factory=list)


# ── News & RAG ────────────────────────────────────────────────────────────────

class NewsItem(BaseModel):
    doc_id:       str           # unique hash of headline + source + timestamp
    ticker:       str
    headline:     str
    summary:      Optional[str] = None
    source:       str
    published_at: datetime
    url:          Optional[str] = None

class RAGChunk(BaseModel):
    chunk_id:     str           # doc_id + chunk index
    doc_id:       str           # back-reference to NewsItem
    ticker:       str
    text:         str
    chunk_index:  int
    embedding:    Optional[list[float]] = None   # populated after embed()


# ── Sentiment ─────────────────────────────────────────────────────────────────

class SentimentPacket(BaseModel):
    ticker:     str
    label:      SentimentLabel
    score:      float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0,  le=1.0)
    reasoning:  str
    doc_ids:    list[str] = Field(default_factory=list)   # traceability


# ── Agent output ──────────────────────────────────────────────────────────────

class TradeProposal(BaseModel):
    """
    Structured output every agent must produce per ticker per cycle.
    HOLD and MONITOR are logged directly.
    BUY and SELL are forwarded to the Judge Layer.
    """
    ticker:     str
    action:     Action
    size_pct:   float = Field(
        ge=0.0, le=5.0,
        description="Target portfolio allocation % (0–5). 0 for HOLD/MONITOR."
    )
    reasoning:  str   = Field(min_length=10)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("size_pct")
    @classmethod
    def size_pct_zero_for_hold(cls, v: float, info) -> float:
        # Soft guard: HOLD/MONITOR should not allocate
        action = info.data.get("action")
        if action in (Action.HOLD, Action.MONITOR) and v > 0:
            return 0.0
        return v


# ── Judge layer ───────────────────────────────────────────────────────────────

class JudgeVerdict(BaseModel):
    proposal:    TradeProposal
    outcome:     JudgeOutcome
    veto_reason: Optional[VetoReason] = None
    veto_detail: Optional[str]        = None
    timestamp:   datetime             = Field(default_factory=datetime.utcnow)


# ── Broker / execution ────────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    ticker:    str
    action:    Action           # BUY or SELL only (judge already filtered)
    qty:       float            # calculated by position_sizer
    arm_id:    ArmID
    cycle_id:  str
    run_id:    str

class OrderResult(BaseModel):
    order_id:   str
    ticker:     str
    action:     Action
    qty:        float
    fill_price: Optional[float] = None
    filled_at:  Optional[datetime] = None
    status:     str             # "filled", "pending", "cancelled", "error"
    arm_id:     ArmID
    cycle_id:   str
    run_id:     str


# ── Memory ────────────────────────────────────────────────────────────────────

class MemoryPacket(BaseModel):
    """
    Tier 2 distilled memory — injected into shared context at session start.
    Max ~200 tokens when serialised.
    """
    session_date:  str                          # YYYY-MM-DD
    key_lesson:    str
    risk_posture:  str                          # e.g. "cautious", "neutral", "aggressive"
    monitor_flags: list[str] = Field(default_factory=list)   # tickers to watch
    pnl_summary:   str


# ── Cycle & session events ────────────────────────────────────────────────────

class CycleEvent(BaseModel):
    """
    Append-only event record written to Postgres events table.
    Every state transition in the system emits one of these.
    """
    event_id:    str
    event_type:  str            # e.g. CYCLE_START, LLM_CALL, JUDGE_VETO,
                                #       ORDER_SUBMITTED, ORDER_FILLED, CYCLE_END,
                                #       SESSION_END, SCHEMA_VIOLATION
    run_id:      str
    cycle_id:    str
    arm_id:      ArmID
    timestamp:   datetime = Field(default_factory=datetime.utcnow)
    payload:     dict     = Field(default_factory=dict)   # event-specific data


# ── Shared context (input to every agent call) ────────────────────────────────

class SharedContext(BaseModel):
    """
    Read-only context block assembled once per cycle and passed
    identically to all agents (monolithic or council).
    """
    run_id:          str
    cycle_id:        str
    arm_id:          ArmID
    cycle_number:    int                    # 1–6
    session_date:    str                    # YYYY-MM-DD
    portfolio:       PortfolioState
    bars:            list[OHLCVBar]
    sentiment:       list[SentimentPacket]
    rag_summaries:   dict[str, str]         # ticker → summarised news
    memory:          Optional[MemoryPacket] = None
    ta_indicators:   dict[str, dict]        # ticker → {rsi, macd, ma20, ma50}
