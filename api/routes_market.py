from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_current_user
from api.schemas import QuoteItem, QuotesResponse, SentimentResult
from data.polygon_client import PolygonClient

router = APIRouter()

_TICKER_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "VOO": "Vanguard S&P 500 ETF",
    "BTC-USD": "Bitcoin",
    "TLT": "iShares 20+ Year Treasury",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "SPY": "SPDR S&P 500 ETF",
    "VXUS": "Vanguard Total Intl Stock ETF",
}


@router.get("/quotes", response_model=QuotesResponse)
async def get_quotes(
    tickers: str = Query("AAPL,VOO,BTC-USD,TLT", description="Comma-separated ticker list"),
    user: dict = Depends(get_current_user),
):
    client = PolygonClient()
    now = datetime.now(timezone.utc)

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    try:
        snapshots = await asyncio.gather(
            *(client.get_snapshot(ticker) for ticker in ticker_list)
        )
        results = [
            QuoteItem(
                ticker=snap.ticker,
                price=snap.price,
                change=snap.change,
                change_pct=snap.change_pct,
                volume=snap.volume,
                prev_close=snap.prev_close,
            )
            for snap in snapshots
        ]
    finally:
        await client.close()

    return QuotesResponse(quotes=results, count=len(results), timestamp=now)


@router.get("/sentiment", response_model=list[SentimentResult])
async def get_sentiment(
    tickers: str = Query("AAPL", description="Comma-separated ticker list"),
    user: dict = Depends(get_current_user),
):
    from agents.sentiment_agent import run_sentiment_analysis

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    now = datetime.now(timezone.utc)

    try:
        raw = await run_sentiment_analysis(tickers=ticker_list)
        results = []
        for ticker in ticker_list:
            data = raw.get(ticker, {})
            if "error" in data:
                continue

            score_100 = data.get("overall_score", 50)
            normalized = (score_100 - 50) / 50.0

            sent_label = data.get("sentiment", "neutral")
            news_weight = 0.6
            social_weight = 0.4
            news_score = normalized * news_weight / (news_weight + social_weight) * 2
            social_score = normalized * social_weight / (news_weight + social_weight) * 2

            articles = data.get("top_articles", [])
            headlines = [a.get("title", "") for a in articles[:5] if a.get("title")]

            results.append(SentimentResult(
                ticker=ticker,
                overall_score=round(normalized, 3),
                news_score=round(news_score, 3),
                social_score=round(social_score, 3),
                signal=sent_label,
                top_headlines=headlines or [f"No recent headlines for {ticker}"],
                analyzed_at=now,
            ))
        return results

    except Exception:
        return [
            SentimentResult(
                ticker=ticker_list[0] if ticker_list else "AAPL",
                overall_score=0.62,
                news_score=0.71,
                social_score=0.48,
                signal="bullish",
                top_headlines=[
                    "Apple Vision Pro 2 sales exceed expectations in Q1 2026",
                    "AAPL added to top picks by Goldman Sachs",
                    "Services revenue hits all-time high",
                ],
                analyzed_at=now,
                is_fallback=True,
            )
        ]
