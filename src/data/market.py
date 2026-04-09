from __future__ import annotations
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from sqlalchemy.orm import Session
from src.core.config import get_config
from src.storage.repository import log_market

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def _get_client() -> StockHistoricalDataClient:
    cfg = get_config()
    return StockHistoricalDataClient(
        api_key=cfg.alpaca_api_key,
        secret_key=cfg.alpaca_secret_key,
    )


def fetch_latest_quotes(symbols: list[str]) -> dict:
    client = _get_client()
    req = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
    return dict(client.get_stock_latest_quote(req))


def fetch_daily_bars(symbols: list[str], lookback_days: int = 55) -> dict[str, list]:
    client = _get_client()
    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=lookback_days + 10)  # buffer for weekends/holidays
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bar_set = client.get_stock_bars(req)
    # bar_set is a BarSet — access via .data dict {symbol: [Bar, ...]}
    result = {}
    for sym in symbols:
        try:
            result[sym] = list(bar_set[sym])
        except (KeyError, TypeError):
            result[sym] = []
            logger.warning("No bars returned for %s", sym)
    return result


def snapshot_market(session: Session, symbols: list[str] | None = None) -> list[dict]:
    cfg = get_config()
    symbols = symbols or cfg.universe.tickers

    quotes = fetch_latest_quotes(symbols)
    bars   = fetch_daily_bars(symbols, lookback_days=55)

    snapshots = []
    for symbol in symbols:
        q          = quotes.get(symbol)
        b_list     = bars.get(symbol, [])
        latest_bar = b_list[-1] if b_list else None
        price      = float(q.ask_price) if q else None
        volume     = int(latest_bar.volume) if latest_bar else None

        raw = {
            "quote": {
                "bid": float(q.bid_price) if q else None,
                "ask": float(q.ask_price) if q else None,
            },
            "bar": {
                "open":   float(latest_bar.open)   if latest_bar else None,
                "high":   float(latest_bar.high)   if latest_bar else None,
                "low":    float(latest_bar.low)    if latest_bar else None,
                "close":  float(latest_bar.close)  if latest_bar else None,
                "volume": volume,
                "vwap":   float(latest_bar.vwap)   if latest_bar else None,
            },
            "daily_bars": [
                {"t": b.timestamp.isoformat(), "open": float(b.open),
                 "high": float(b.high), "low": float(b.low),
                 "close": float(b.close), "volume": int(b.volume)}
                for b in b_list
            ],
        }
        log_market(session, symbol=symbol, price=price, volume=volume, raw=raw)
        snapshots.append({"symbol": symbol, "price": price, "volume": volume, **raw})
        logger.info("Snapshot saved: %s @ %.2f", symbol, price or 0)

    return snapshots
