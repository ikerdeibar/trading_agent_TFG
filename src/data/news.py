from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from src.core.config import get_config

logger = logging.getLogger(__name__)


def _get_client() -> NewsClient:
    cfg = get_config()
    return NewsClient(
        api_key=cfg.alpaca_api_key,
        secret_key=cfg.alpaca_secret_key,
    )


def fetch_news(symbols: list[str], lookback_hours: int = 24) -> dict[str, list[dict]]:
    client = _get_client()
    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(hours=lookback_hours)

    req = NewsRequest(
        symbols=",".join(symbols),
        start=start,
        end=end,
        limit=50,
        sort="desc",
    )

    result: dict[str, list[dict]] = {s: [] for s in symbols}

    try:
        resp       = client.get_news(req)
        # SDK returns NewsSet with .data = {'news': [dict, ...]}
        news_items = resp.data.get("news", []) if hasattr(resp, "data") else []

        for article in news_items:
            syms = article.get("symbols", []) if isinstance(article, dict) else (article.symbols or [])
            headline  = article.get("headline", "")  if isinstance(article, dict) else article.headline
            summary   = article.get("summary", "")   if isinstance(article, dict) else (article.summary or "")
            url       = article.get("url", "")        if isinstance(article, dict) else (article.url or "")
            published = str(article.get("created_at", "")) if isinstance(article, dict) else (
                article.created_at.isoformat() if article.created_at else "")

            for sym in syms:
                if sym in result and len(result[sym]) < 5:
                    result[sym].append({
                        "headline":  headline,
                        "summary":   summary,
                        "published": published,
                        "url":       url,
                    })
    except Exception as e:
        logger.warning("News fetch failed: %s", e)

    for sym in symbols:
        logger.info("News: %s — %d articles", sym, len(result[sym]))

    return result
