from __future__ import annotations
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.agents import get_agent
from src.execution.broker import get_portfolio, place_order, reconcile_fills
from src.risk.guard import check_order, RiskViolation
from src.storage.repository import log_portfolio, log_decision, write_post_mortem
from src.data.sentiment import extract_sentiment_all
from src.storage.models import SentimentSnapshot
from src.data.market import snapshot_market
from src.core.config import get_config

logger = logging.getLogger(__name__)


def run_cycle(session: Session, memory_packet: str = "") -> dict:
    cfg      = get_config()
    cycle_ts = datetime.now(tz=timezone.utc)

    # ── OBSERVE ──────────────────────────────────────────────────────────────
    logger.info("=== OBSERVE ===")
    snapshot_market(session)
    portfolio = get_portfolio()
    log_portfolio(
        session,
        cash=portfolio["cash"],
        arm_id =cfg.arm_id,
        equity=portfolio["equity"],
        total_value=portfolio["equity"],
        positions=portfolio["positions"],
    )

    # Extract current prices for post-mortem on the PREVIOUS cycle
    current_prices = {
        sym: pos.get("current_price", 0)
        for sym, pos in portfolio.get("positions", {}).items()
    }
    try:
        write_post_mortem(session, cfg.arm_id, current_prices)
    except Exception as e:
        logger.warning("Post-mortem write skipped: %s", e)
        session.rollback()

    # ── ORIENT + DECIDE ──────────────────────────────────────────────────────
    logger.info("=== ORIENT + DECIDE ===")
    agent  = get_agent(session, memory_packet)
    result = agent.decide()

    # Capture context that was assembled for the LLM
    ctx      = getattr(agent, "last_context", {})
    news_ctx = {
        sym: [
            {"headline": a.get("headline", ""), "url": a.get("url", ""), "published_at": a.get("created_at", "")}
            for a in data.get("news", [])
        ]
        for sym, data in ctx.get("tickers", {}).items()
    }
    ta_ctx = {
        sym: {k: v for k, v in data.items() if k != "news"}
        for sym, data in ctx.get("tickers", {}).items()
    }

    # ── SENTIMENT (shared, runs once per cycle) ───────────────────────────────
    sentiment_map = {}
    try:
        sentiment_map = extract_sentiment_all(ctx.get("tickers", {}))
        for sym, s in sentiment_map.items():
            session.add(SentimentSnapshot(
                cycle_ts=cycle_ts, symbol=sym,
                label=s["label"], score=s["score"], confidence=s["confidence"],
            ))
        session.commit()
        logger.info("Sentiment snapshots committed for %d symbols", len(sentiment_map))
    except Exception as e:
        logger.warning("Sentiment extraction/commit failed: %s", e)
        session.rollback()

    # ── ACT ──────────────────────────────────────────────────────────────────
    logger.info("=== ACT ===")
    executed = []
    skipped  = []

    for order in result.get("orders", []):
        symbol = order.get("symbol", "")
        side   = order.get("side", "hold")
        qty    = int(order.get("qty", 0))

        if side == "hold" or qty <= 0:
            skipped.append({"symbol": symbol, "reason": "hold"})
            continue

        price = portfolio.get("positions", {}).get(symbol, {}).get("current_price", 1.0)

        try:
            check_order(session, symbol, qty, side, price, portfolio)
            placed = place_order(
                session, symbol, qty, side,
                arm_id=cfg.arm_id,
                agent=type(agent).__name__,
                reasoning=result.get("reasoning", "")[:200],
            )
            executed.append({"symbol": symbol, "side": side, "qty": qty, "price": price, "order_id": str(placed)})
            logger.info("Executed: %s %s x%s", side.upper(), symbol, qty)
        except RiskViolation as e:
            logger.warning("Risk blocked %s %s x%s: %s", side, symbol, qty, e)
            skipped.append({"symbol": symbol, "reason": f"risk_blocked: {e}"})
            session.rollback()
        except Exception as e:
            logger.error("Order failed %s %s x%s: %s", side, symbol, qty, e)
            skipped.append({"symbol": symbol, "reason": f"error: {e}"})
            session.rollback()

    try:
        reconcile_fills(session)
    except Exception as e:
        logger.warning("reconcile_fills failed: %s", e)
        session.rollback()

    # ── LOG DECISION ─────────────────────────────────────────────────────────
    try:
        log_decision(
            session,
            arm_id=cfg.arm_id,
            cycle_ts=cycle_ts,
            agent=type(agent).__name__,
            reasoning=result.get("reasoning", ""),
            orders_proposed=result.get("orders", []),
            orders_executed=executed,
            orders_blocked=skipped,
            news_context=news_ctx,
            ta_context=ta_ctx,
            portfolio_snapshot=portfolio,
            memory_packet_in=memory_packet,
            llm_tokens_used=result.get("tokens_used", 0),
            llm_cost_usd=result.get("cost_usd", 0.0),
            sentiment_per_symbol=sentiment_map,
        )
    except Exception as e:
        logger.error("log_decision failed: %s", e)
        session.rollback()

    summary = {
        "arm":      cfg.arm_id,
        "executed": executed,
        "skipped":  skipped,
        "equity":   portfolio["equity"],
    }
    logger.info("Cycle complete: %d executed, %d skipped", len(executed), len(skipped))
    return summary
