# tests/test_schemas.py
import pytest
from src.llm.structured import parse_and_validate, parse_proposals, StructuredOutputError
from src.core.types import TradeProposal, Action, VetoReason


# ── parse_and_validate ────────────────────────────────────────────────────────

def test_valid_proposal_parses():
    raw = '{"ticker":"AAPL","action":"BUY","size_pct":2.0,"reasoning":"Strong momentum signal confirmed.","confidence":0.8}'
    result = parse_and_validate(raw, TradeProposal)
    assert result.ticker == "AAPL"
    assert result.action == Action.BUY
    assert result.size_pct == 2.0

def test_malformed_json_raises():
    with pytest.raises(StructuredOutputError) as exc:
        parse_and_validate("not valid json", TradeProposal)
    assert exc.value.reason == VetoReason.SCHEMA_INVALID

def test_missing_field_raises():
    # Missing 'reasoning'
    raw = '{"ticker":"AAPL","action":"BUY","size_pct":2.0,"confidence":0.8}'
    with pytest.raises(StructuredOutputError) as exc:
        parse_and_validate(raw, TradeProposal)
    assert exc.value.reason == VetoReason.SCHEMA_INVALID

def test_invalid_action_raises():
    raw = '{"ticker":"AAPL","action":"YOLO","size_pct":2.0,"reasoning":"Some reasoning here.","confidence":0.8}'
    with pytest.raises(StructuredOutputError):
        parse_and_validate(raw, TradeProposal)

def test_size_pct_out_of_range_raises():
    raw = '{"ticker":"AAPL","action":"BUY","size_pct":99.0,"reasoning":"Some reasoning here.","confidence":0.8}'
    with pytest.raises(StructuredOutputError):
        parse_and_validate(raw, TradeProposal)


# ── parse_proposals ───────────────────────────────────────────────────────────

def test_valid_proposals_list():
    raw = '''{
        "proposals": [
            {"ticker":"AAPL","action":"BUY","size_pct":2.0,
             "reasoning":"Strong earnings beat confirmed.","confidence":0.8},
            {"ticker":"MSFT","action":"HOLD","size_pct":0.0,
             "reasoning":"Neutral signal, monitoring closely.","confidence":0.5}
        ]
    }'''
    results = parse_proposals(raw)
    assert len(results) == 2
    assert results[0].ticker == "AAPL"
    assert results[1].action == Action.HOLD

def test_missing_proposals_key_raises():
    raw = '{"trades": []}'
    with pytest.raises(StructuredOutputError) as exc:
        parse_proposals(raw)
    assert exc.value.reason == VetoReason.SCHEMA_INVALID

def test_partial_failure_returns_valid_subset():
    raw = '''{
        "proposals": [
            {"ticker":"AAPL","action":"BUY","size_pct":2.0,
             "reasoning":"Valid proposal with good signal.","confidence":0.8},
            {"ticker":"MSFT","action":"INVALID_ACTION","size_pct":1.0,
             "reasoning":"Bad action field here.","confidence":0.5}
        ]
    }'''
    results = parse_proposals(raw)
    assert len(results) == 1
    assert results[0].ticker == "AAPL"

def test_all_invalid_proposals_raises():
    raw = '{"proposals": [{"ticker":"AAPL","action":"INVALID","size_pct":99.0,"reasoning":"x","confidence":5.0}]}'
    with pytest.raises(StructuredOutputError):
        parse_proposals(raw)
