# src/risk/judge.py
from datetime import datetime
from src.core.types import (
    TradeProposal, JudgeVerdict, JudgeOutcome, VetoReason, Action
)
from src.risk.rules import (
    rule_schema,
    rule_whitelist,
    rule_position_cap,
    rule_market_hours,
    rule_cycle_cap,
)


def run_judge(
    proposal: TradeProposal,
    current_weight_pct: float,
    cycles_completed: int,
    now: datetime | None = None,
) -> JudgeVerdict:
    """
    Sequential 5-rule validator. Stops at the first failure.
    Returns JudgeVerdict with APPROVED or VETOED outcome.

    Args:
        proposal:            The TradeProposal from the agent layer.
        current_weight_pct:  Existing portfolio allocation for this ticker (0–5 scale).
        cycles_completed:    Number of OODA cycles already completed this session.
        now:                 Override for market hours check (used in tests).
    """

    rules = [
        lambda: rule_schema(proposal),
        lambda: rule_whitelist(proposal),
        lambda: rule_position_cap(proposal, current_weight_pct),
        lambda: rule_market_hours(now),
        lambda: rule_cycle_cap(cycles_completed),
    ]

    for rule in rules:
        passed, veto_reason, veto_detail = rule()
        if not passed:
            return JudgeVerdict(
                proposal=proposal,
                outcome=JudgeOutcome.VETOED,
                veto_reason=veto_reason,
                veto_detail=veto_detail,
            )

    return JudgeVerdict(
        proposal=proposal,
        outcome=JudgeOutcome.APPROVED,
    )


def is_actionable(proposal: TradeProposal) -> bool:
    """
    Returns True only for BUY or SELL — the only actions forwarded to the judge.
    HOLD and MONITOR are logged directly without going through judge.
    """
    return proposal.action in (Action.BUY, Action.SELL)
