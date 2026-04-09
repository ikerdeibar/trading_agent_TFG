# src/risk/rules.py
from datetime import datetime, time
from zoneinfo import ZoneInfo
from src.core.types import TradeProposal, Action, VetoReason

# ── Constants ─────────────────────────────────────────────────────────────────

WHITELIST: frozenset[str] = frozenset({
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN",
    "JPM",  "XOM",  "LLY",  "CAT",   "NEE",
})

MAX_POSITION_PCT: float = 5.0
MAX_CYCLES_PER_SESSION: int = 6

# Actionable window: 09:45 – 15:45 ET (conservative buffer on open)
_ET = ZoneInfo("America/New_York")
_WINDOW_OPEN  = time(9, 45)
_WINDOW_CLOSE = time(15, 45)


# ── Individual rule functions ─────────────────────────────────────────────────
# Each returns (passed: bool, reason: VetoReason | None, detail: str | None)

def rule_schema(proposal: TradeProposal) -> tuple[bool, VetoReason | None, str | None]:
    """
    Rule 1: Schema validation.
    Pydantic already enforces types upstream; this catches logical gaps
    such as BUY/SELL with size_pct == 0.
    """
    if proposal.action in (Action.BUY, Action.SELL) and proposal.size_pct <= 0.0:
        return (
            False,
            VetoReason.SCHEMA_INVALID,
            f"BUY/SELL requires size_pct > 0, got {proposal.size_pct}",
        )
    if not proposal.reasoning or len(proposal.reasoning.strip()) < 10:
        return (
            False,
            VetoReason.SCHEMA_INVALID,
            "reasoning field is empty or too short",
        )
    return True, None, None


def rule_whitelist(proposal: TradeProposal) -> tuple[bool, VetoReason | None, str | None]:
    """
    Rule 2: Ticker must be in the 10-asset basket.
    Blocks hallucinated or out-of-universe tickers.
    """
    if proposal.ticker not in WHITELIST:
        return (
            False,
            VetoReason.TICKER_NOT_IN_WHITELIST,
            f"{proposal.ticker} is not in the approved basket",
        )
    return True, None, None


def rule_position_cap(
    proposal: TradeProposal,
    current_weight_pct: float,
) -> tuple[bool, VetoReason | None, str | None]:
    """
    Rule 3: Executing this BUY must not push the position above 5% of portfolio.
    current_weight_pct = existing allocation for this ticker (0–100 scale).
    """
    if proposal.action == Action.BUY:
        projected = current_weight_pct + proposal.size_pct
        if projected > MAX_POSITION_PCT:
            return (
                False,
                VetoReason.POSITION_CAP_EXCEEDED,
                (
                    f"{proposal.ticker}: current {current_weight_pct:.2f}% + "
                    f"proposed {proposal.size_pct:.2f}% = {projected:.2f}% "
                    f"exceeds {MAX_POSITION_PCT}% cap"
                ),
            )
    return True, None, None


def rule_market_hours(
    now: datetime | None = None,
) -> tuple[bool, VetoReason | None, str | None]:
    """
    Rule 4: Order must be submitted within the actionable window (09:45–15:45 ET).
    Accepts an optional `now` for testability; defaults to current UTC time.
    """
    if now is None:
        now = datetime.now(tz=_ET)
    else:
        now = now.astimezone(_ET)

    current_time = now.time()
    if not (_WINDOW_OPEN <= current_time <= _WINDOW_CLOSE):
        return (
            False,
            VetoReason.MARKET_CLOSED,
            f"Current ET time {current_time.strftime('%H:%M')} is outside "
            f"actionable window {_WINDOW_OPEN}–{_WINDOW_CLOSE}",
        )
    return True, None, None


def rule_cycle_cap(
    cycles_completed: int,
) -> tuple[bool, VetoReason | None, str | None]:
    """
    Rule 5: Session must not exceed N=6 OODA cycles.
    """
    if cycles_completed >= MAX_CYCLES_PER_SESSION:
        return (
            False,
            VetoReason.CYCLE_CAP_EXCEEDED,
            f"Session already completed {cycles_completed} cycles "
            f"(max {MAX_CYCLES_PER_SESSION})",
        )
    return True, None, None
