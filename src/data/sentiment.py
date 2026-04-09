from __future__ import annotations
import logging
from functools import lru_cache
from transformers import pipeline

logger = logging.getLogger(__name__)

LABEL_MAP = {"positive": "BULLISH", "negative": "BEARISH", "neutral": "NEUTRAL"}
SCORE_DIR  = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


@lru_cache(maxsize=1)
def _get_pipeline():
    logger.info("Loading FinBERT model (first call only)...")
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
        device=-1,          # CPU; change to 0 if you have a GPU
    )


def _score_texts(texts: list[str]) -> dict:
    """Run FinBERT on a list of texts, return averaged label/score/confidence."""
    pipe = _get_pipeline()
    totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    for text in texts:
        results = pipe(text[:512])          # FinBERT max 512 tokens
        for r in results[0]:
            label = r["label"].lower()
            if label in totals:
                totals[label] += r["score"]

    n = len(texts)
    avg = {k: v / n for k, v in totals.items()}
    dominant = max(avg, key=avg.get)
    float_score = avg["positive"] - avg["negative"]  # range [-1, 1]

    return {
        "label":      LABEL_MAP[dominant],
        "score":      round(float_score, 4),
        "confidence": round(avg[dominant], 4),
    }


def extract_sentiment(symbol: str, articles: list[dict]) -> dict:
    """Return sentiment dict for a single ticker."""
    if not articles:
        return {"label": "NEUTRAL", "score": 0.0, "confidence": 0.0}

    texts = [
        f"{a.get('headline', '')}. {a.get('summary', '')}".strip()
        for a in articles[:5]
        if a.get("headline")
    ]
    if not texts:
        return {"label": "NEUTRAL", "score": 0.0, "confidence": 0.0}

    try:
        return _score_texts(texts)
    except Exception as e:
        logger.warning("FinBERT failed for %s: %s", symbol, e)
        return {"label": "NEUTRAL", "score": 0.0, "confidence": 0.0}


def extract_sentiment_all(tickers: dict) -> dict:
    """
    tickers = context['tickers'] dict from build_context().
    Returns {symbol: {label, score, confidence}} for all tickers.
    Runs once per cycle — result is shared across all arms.
    """
    results = {}
    for sym, data in tickers.items():
        articles = data.get("news", [])
        results[sym] = extract_sentiment(sym, articles)
        logger.info(
            "Sentiment %s: %s (score=%.3f, conf=%.3f)",
            sym,
            results[sym]["label"],
            results[sym]["score"],
            results[sym]["confidence"],
        )
    return results
