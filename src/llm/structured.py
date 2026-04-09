# src/llm/structured.py
from __future__ import annotations
import json
import logging
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
from src.core.types import VetoReason

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when LLM output cannot be parsed or validated."""
    def __init__(self, reason: VetoReason, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


def parse_and_validate(
    raw_content: str,
    schema:      Type[T],
    agent_role:  str  = "unknown",
    cycle_id:    str  = "unknown",
) -> T:
    """
    Parse raw LLM JSON string and validate against a Pydantic schema.

    Steps:
      1. JSON parse — catches malformed JSON
      2. Pydantic validation — catches schema violations
      3. Returns validated model instance

    Raises StructuredOutputError on any failure.
    Both failure types are logged as SCHEMA_VIOLATION events.
    """
    # Step 1 — JSON parse
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        detail = f"[{agent_role}] JSON parse failed at cycle {cycle_id}: {e}"
        logger.warning("schema_violation", extra={
            "violation_type": "json_parse_error",
            "agent_role":     agent_role,
            "cycle_id":       cycle_id,
            "raw_content":    raw_content[:300],   # truncate for log safety
            "error":          str(e),
        })
        raise StructuredOutputError(VetoReason.SCHEMA_INVALID, detail) from e

    # Step 2 — Pydantic validation
    try:
        return schema(**data)
    except ValidationError as e:
        detail = f"[{agent_role}] Schema validation failed at cycle {cycle_id}: {e}"
        logger.warning("schema_violation", extra={
            "violation_type": "pydantic_validation_error",
            "agent_role":     agent_role,
            "cycle_id":       cycle_id,
            "parsed_data":    data,
            "error":          str(e),
        })
        raise StructuredOutputError(VetoReason.SCHEMA_INVALID, detail) from e


def parse_proposals(
    raw_content: str,
    agent_role:  str = "unknown",
    cycle_id:    str = "unknown",
) -> list:
    """
    Parse a JSON object containing a 'proposals' list of TradeProposals.
    Expected LLM output format:
    {
        "proposals": [
            {"ticker": "AAPL", "action": "BUY", "size_pct": 2.0,
             "reasoning": "...", "confidence": 0.8},
            ...
        ]
    }
    Returns a list of validated TradeProposal objects.
    """
    from src.core.types import TradeProposal

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        detail = f"[{agent_role}] JSON parse failed: {e}"
        logger.warning("schema_violation", extra={
            "violation_type": "json_parse_error",
            "agent_role": agent_role,
            "cycle_id":   cycle_id,
            "error":      str(e),
        })
        raise StructuredOutputError(VetoReason.SCHEMA_INVALID, detail) from e

    if "proposals" not in data:
        detail = f"[{agent_role}] Missing 'proposals' key in LLM output"
        logger.warning("schema_violation", extra={
            "violation_type": "missing_proposals_key",
            "agent_role":     agent_role,
            "cycle_id":       cycle_id,
            "parsed_data":    data,
        })
        raise StructuredOutputError(VetoReason.SCHEMA_INVALID, detail)

    proposals = []
    errors    = []
    for i, item in enumerate(data["proposals"]):
        try:
            proposals.append(TradeProposal(**item))
        except ValidationError as e:
            errors.append(f"proposal[{i}]: {e}")
            logger.warning("schema_violation", extra={
                "violation_type": "proposal_validation_error",
                "agent_role":     agent_role,
                "cycle_id":       cycle_id,
                "proposal_index": i,
                "error":          str(e),
            })

    if not proposals:
        detail = f"[{agent_role}] All {len(data['proposals'])} proposals failed validation: {errors}"
        raise StructuredOutputError(VetoReason.SCHEMA_INVALID, detail)

    if errors:
        logger.warning("partial_proposal_failure", extra={
            "agent_role":       agent_role,
            "cycle_id":         cycle_id,
            "total":            len(data["proposals"]),
            "failed":           len(errors),
            "succeeded":        len(proposals),
        })

    return proposals
