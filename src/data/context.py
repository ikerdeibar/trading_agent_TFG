from __future__ import annotations
import json
from sqlalchemy.orm import Session
from src.core.config import get_config
from src.data.market import fetch_daily_bars, fetch_latest_quotes
from src.data.indicators import compute_indicators
from src.data.news import fetch_news
from src.execution.broker import get_portfolio


def build_context(session: Session, memory_packet: str = "") -> dict:
    cfg     = get_config()
    symbols = cfg.universe.tickers

    portfolio = get_portfolio()
    quotes    = fetch_latest_quotes(symbols)
    bars      = fetch_daily_bars(symbols, lookback_days=90)

    ta_signals = {}
    for sym in symbols:
        ta_signals[sym] = compute_indicators(bars.get(sym, []))

    news = fetch_news(symbols, lookback_hours=24)

    tickers = {}
    for sym in symbols:
        q = quotes.get(sym)
        tickers[sym] = {
            "price": {
                "bid": float(q.bid_price) if q else None,
                "ask": float(q.ask_price) if q else None,
            },
            "indicators": ta_signals[sym],
            "news": news[sym],
        }

    return {
        "arm_id":    cfg.arm_id,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "portfolio": portfolio,
        "tickers":   tickers,
        "memory":    memory_packet or "No prior session memory available.",
        "risk_limits": {
            "max_position_pct": cfg.risk.max_position_pct,
            "max_position_usd": round(portfolio.get("equity", 0) * cfg.risk.max_position_pct / 100, 2),
            "buying_power_usd": round(portfolio.get("buying_power", 0), 2),
        },
    }


def _rsi_label(rsi):
    if rsi is None:
        return "n/a"
    if rsi >= 70:
        return f"{rsi:.1f} (overbought)"
    if rsi <= 30:
        return f"{rsi:.1f} (oversold)"
    return f"{rsi:.1f} (neutral)"


def _macd_label(macd):
    if macd is None:
        return "n/a"
    return f"{macd:+.2f} ({'bullish' if macd > 0 else 'bearish'})"


def context_to_prompt(context: dict) -> str:
    lines = []

    lines.append(f"ARM: {context['arm_id']}  |  Timestamp: {context['timestamp']}")
    lines.append(
    f"Risk limits: max position per asset "
    f"{context['risk_limits']['max_position_pct']}% of equity "
    f"= ${context['risk_limits']['max_position_usd']:,.0f} USD. "
    f"Buying power available: ${context['risk_limits']['buying_power_usd']:,.0f} USD. "
    f"IMPORTANT: size orders so that qty × price ≤ ${context['risk_limits']['max_position_usd']:,.0f}."
    )

    p = context["portfolio"]
    lines.append("")
    lines.append("=== PORTFOLIO ===")
    lines.append(f"Cash: ${p.get('cash', 0):,.2f}  |  Equity: ${p.get('equity', 0):,.2f}  |  Buying power: ${p.get('buying_power', 0):,.2f}")
    positions = p.get("positions", {})
    if positions:
        lines.append("Open positions:")
        for sym, pos in (positions.items() if isinstance(positions, dict) else [(p.get("symbol","?"), p) for p in positions]):
            lines.append(f"  {sym}: qty={pos.get('qty', 0)}  mkt_value=${pos.get('market_value', 0):,.2f}")
    else:
        lines.append("  No open positions.")

    lines.append("")
    lines.append("=== TECHNICAL INDICATORS (20-day daily bars) ===")
    lines.append(f"{'Ticker':<6} {'Price':>8} {'RSI':>20} {'MACD':>18} {'MA20':>8} {'MA50':>8} {'vs MA20':>9} {'vs MA50':>9}")
    lines.append("-" * 92)
    for sym, data in context["tickers"].items():
        ind   = data.get("indicators") or {}
        pd    = data.get("price", {})
        price = pd.get("ask") or pd.get("bid") or 0.0
        rsi   = ind.get("rsi")
        macd  = ind.get("macd")
        ma20  = ind.get("ma20")
        ma50  = ind.get("ma50")
        a20   = ("above" if price > ma20 else "below") if ma20 else "n/a"
        a50   = ("above" if price > ma50 else "below") if ma50 else "n/a"
        ma20s = f"{ma20:.2f}" if ma20 else "n/a"
        ma50s = f"{ma50:.2f}" if ma50 else "n/a"
        lines.append(
            f"{sym:<6} {price:>8.2f} {_rsi_label(rsi):>20} {_macd_label(macd):>18} "
            f"{ma20s:>8} {ma50s:>8} {a20:>9} {a50:>9}"
        )

    lines.append("")
    lines.append("=== NEWS (last 24h) ===")
    for sym, data in context["tickers"].items():
        articles = data.get("news", [])
        lines.append("")
        if articles:
            lines.append(f"{sym}:")
            for a in articles[:5]:
                headline = a.get("headline") or a.get("title", "")
                source   = a.get("source", "")
                lines.append(f"  • [{source}] {headline}")
        else:
            lines.append(f"{sym}: No news.")

    lines.append("")
    lines.append("=== SESSION MEMORY ===")
    lines.append(str(context.get("memory", "No prior session memory available.")))

    return "\n".join(lines)
