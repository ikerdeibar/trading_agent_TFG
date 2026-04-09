from __future__ import annotations
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from sqlalchemy.orm import Session
from src.core.config import get_config
from src.storage.repository import log_trade

logger = logging.getLogger(__name__)


def _get_client() -> TradingClient:
    cfg = get_config()
    return TradingClient(
        api_key=cfg.alpaca_api_key,
        secret_key=cfg.alpaca_secret_key,
        paper=cfg.alpaca_paper,
    )


def get_portfolio(session: Session | None = None) -> dict:
    client = _get_client()
    account = client.get_account()
    positions = client.get_all_positions()
    return {
        "cash":         float(account.cash),
        "equity":       float(account.equity),
        "buying_power": float(account.buying_power),
        "positions":    {p.symbol: {"qty": float(p.qty), "market_value": float(p.market_value), "current_price": float(p.current_price) if p.current_price else 0.0} for p in positions},
    }


def place_order(
    session: Session,
    symbol: str,
    qty: float,
    side: str,
    arm_id: str = None,
    agent: str = "unknown",
    reasoning: str = "",
) -> dict:
    if qty <= 0:
        raise ValueError(f"Invalid qty {qty} for {symbol}")

    client = _get_client()
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )

    order = client.submit_order(req)
    logger.info("Order placed: %s %s x%s | id=%s", side.upper(), symbol, qty, order.id)

    log_trade(
        session,
        symbol=symbol,
        side=side.upper(),
        qty=qty,
        price=0.0,
        order_id=str(order.id),
        status=str(order.status),
        agent=agent,
        arm_id=arm_id or get_config().arm_id,
        reasoning=reasoning,
    )

    return {
        "order_id": str(order.id),
        "symbol":   symbol,
        "side":     side.upper(),
        "qty":      qty,
        "status":   str(order.status),
    }


def get_open_orders() -> list[dict]:
    client = _get_client()
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    orders = client.get_orders(filter=req)
    return [{"order_id": str(o.id), "symbol": o.symbol, "side": str(o.side), "qty": float(o.qty)} for o in orders]


def cancel_all_orders() -> int:
    client = _get_client()
    cancelled = client.cancel_orders()
    logger.info("Cancelled %d open orders", len(cancelled))
    return len(cancelled)


def reconcile_fills(session: Session) -> None:
    from src.storage.models import Trade
    from sqlalchemy import select
    pending = session.execute(
        select(Trade).where(Trade.price == 0.0).where(Trade.arm_id == get_config().arm_id)
    ).scalars().all()

    if not pending:
        return

    client = _get_client()
    for trade in pending:
        try:
            order = client.get_order_by_id(trade.order_id)
            if order.filled_avg_price:
                trade.price  = float(order.filled_avg_price)
                trade.status = str(order.status)
        except Exception as e:
            logger.warning("Could not reconcile order %s: %s", trade.order_id, e)
    session.commit()
    logger.info("Reconciled %d trades", len(pending))
