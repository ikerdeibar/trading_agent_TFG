# tests/test_judge_rules.py
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
from src.core.types import Action, TradeProposal, VetoReason
from src.risk.rules import (
    rule_schema,
    rule_whitelist,
    rule_position_cap,
    rule_market_hours,
    rule_cycle_cap,
    MAX_POSITION_PCT,
    MAX_CYCLES_PER_SESSION,
)

_ET = ZoneInfo("America/New_York")

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_proposal(**kwargs) -> TradeProposal:
    defaults = dict(
        ticker="AAPL",
        action=Action.BUY,
        size_pct=2.0,
        reasoning="Strong earnings beat with positive momentum signal.",
        confidence=0.8,
    )
    defaults.update(kwargs)
    return TradeProposal(**defaults)


# ── Rule 1: Schema ────────────────────────────────────────────────────────────

def test_schema_valid_buy():
    passed, reason, _ = rule_schema(make_proposal())
    assert passed is True
    assert reason is None

def test_schema_buy_zero_size_fails():
    passed, reason, _ = rule_schema(make_proposal(action=Action.BUY, size_pct=0.0))
    assert passed is False
    assert reason == VetoReason.SCHEMA_INVALID

def test_schema_sell_zero_size_fails():
    passed, reason, _ = rule_schema(make_proposal(action=Action.SELL, size_pct=0.0))
    assert passed is False
    assert reason == VetoReason.SCHEMA_INVALID

def test_schema_hold_zero_size_passes():
    passed, reason, _ = rule_schema(make_proposal(action=Action.HOLD, size_pct=0.0))
    assert passed is True

def test_schema_short_reasoning_fails():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        make_proposal(reasoning="ok")


# ── Rule 2: Whitelist ─────────────────────────────────────────────────────────

def test_whitelist_valid_ticker():
    for ticker in ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN",
                   "JPM", "XOM", "LLY", "CAT", "NEE"]:
        passed, reason, _ = rule_whitelist(make_proposal(ticker=ticker))
        assert passed is True, f"{ticker} should be whitelisted"

def test_whitelist_invalid_ticker():
    passed, reason, _ = rule_whitelist(make_proposal(ticker="TSLA"))
    assert passed is False
    assert reason == VetoReason.TICKER_NOT_IN_WHITELIST

def test_whitelist_hallucinated_ticker():
    passed, reason, _ = rule_whitelist(make_proposal(ticker="FAKESTOCK"))
    assert passed is False
    assert reason == VetoReason.TICKER_NOT_IN_WHITELIST


# ── Rule 3: Position cap ──────────────────────────────────────────────────────

def test_position_cap_within_limit():
    passed, reason, _ = rule_position_cap(make_proposal(size_pct=2.0), current_weight_pct=2.0)
    assert passed is True   # 2 + 2 = 4, under cap

def test_position_cap_exactly_at_limit():
    passed, reason, _ = rule_position_cap(make_proposal(size_pct=2.5), current_weight_pct=2.5)
    assert passed is True   # 2.5 + 2.5 = 5.0, exactly at cap

def test_position_cap_exceeded():
    passed, reason, _ = rule_position_cap(make_proposal(size_pct=3.0), current_weight_pct=3.0)
    assert passed is False  # 3 + 3 = 6 > 5
    assert reason == VetoReason.POSITION_CAP_EXCEEDED

def test_position_cap_sell_always_passes():
    # SELLs reduce exposure — cap rule does not block them
    passed, reason, _ = rule_position_cap(
        make_proposal(action=Action.SELL, size_pct=5.0),
        current_weight_pct=5.0,
    )
    assert passed is True


# ── Rule 4: Market hours ──────────────────────────────────────────────────────

def _et(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 9, hour, minute, tzinfo=_ET)

def test_market_hours_inside_window():
    passed, reason, _ = rule_market_hours(now=_et(10, 30))
    assert passed is True

def test_market_hours_exactly_at_open():
    passed, reason, _ = rule_market_hours(now=_et(9, 45))
    assert passed is True

def test_market_hours_exactly_at_close():
    passed, reason, _ = rule_market_hours(now=_et(15, 45))
    assert passed is True

def test_market_hours_before_open():
    passed, reason, _ = rule_market_hours(now=_et(9, 0))
    assert passed is False
    assert reason == VetoReason.MARKET_CLOSED

def test_market_hours_after_close():
    passed, reason, _ = rule_market_hours(now=_et(16, 0))
    assert passed is False
    assert reason == VetoReason.MARKET_CLOSED

def test_market_hours_premarket():
    passed, reason, _ = rule_market_hours(now=_et(7, 0))
    assert passed is False
    assert reason == VetoReason.MARKET_CLOSED


# ── Rule 5: Cycle cap ─────────────────────────────────────────────────────────

def test_cycle_cap_within_limit():
    for n in range(MAX_CYCLES_PER_SESSION):
        passed, reason, _ = rule_cycle_cap(cycles_completed=n)
        assert passed is True, f"cycle {n} should be allowed"

def test_cycle_cap_at_limit():
    passed, reason, _ = rule_cycle_cap(cycles_completed=MAX_CYCLES_PER_SESSION)
    assert passed is False
    assert reason == VetoReason.CYCLE_CAP_EXCEEDED

def test_cycle_cap_exceeded():
    passed, reason, _ = rule_cycle_cap(cycles_completed=MAX_CYCLES_PER_SESSION + 5)
    assert passed is False
    assert reason == VetoReason.CYCLE_CAP_EXCEEDED
