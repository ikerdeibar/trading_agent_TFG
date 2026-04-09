from __future__ import annotations
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from src.core.config import get_config

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class RiskViolation(Exception):
    """Raised when a proposed order violates a risk constraint."""
    pass


def _market_open(cfg) -> bool:
    now_et = datetime.now(tz=ET).strftime("%H:%M")
    return cfg.risk.market_open_et <= now_et <= cfg.risk.market_close_et


def _position_pct(portfolio: dict, symbol: str, proposed_value: float) -> float:
    total = portfolio.get("equity", 0)
    if total <= 0:
        return 100.0
    current_mv = portfolio.get("positions", {}).get(symbol, {}).get("market_value", 0.0)
    return ((current_mv + proposed_value) / total) * 100


def check_order(
    session: Session,
    symbol: str,
    qty: float,
    side: str,
    price: float,
    portfolio: dict,
) -> None:
    """
    Neuro-symbolic risk guard. Raises RiskViolation on any breach.
    Three hard constraints only:
      1. Market hours gate (09:45-15:45 ET)
      2. Max position size per asset (5% of equity, buys only)
      3. Sufficient buying power
    """
    cfg = get_config()

    # 1. Market hours gate
    if not _market_open(cfg):
        raise RiskViolation(
            f"Market closed. Trading only allowed "
            f"{cfg.risk.market_open_et}–{cfg.risk.market_close_et} ET."
        )

    # 2. Max position size — buys only
    if side.lower() == "buy":
        proposed_value = qty * price
        pos_pct = _position_pct(portfolio, symbol, proposed_value)
        if pos_pct > cfg.risk.max_position_pct:
            raise RiskViolation(
                f"Position size breach: {symbol} would reach "
                f"{pos_pct:.1f}% of equity "
                f"(max {cfg.risk.max_position_pct}%)."
            )

    # 3. Sufficient buying power
    if side.lower() == "buy":
        cost = qty * price
        if cost > portfolio.get("buying_power", 0):
            raise RiskViolation(
                f"Insufficient buying power: need ${cost:,.2f}, "
                f"have ${portfolio['buying_power']:,.2f}."
            )

    logger.info("Risk checks passed: %s %s x%s", side.upper(), symbol, qty)
