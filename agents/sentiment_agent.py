"""
LangGraph Sentiment Analysis Agent.

Pipeline:
  1. [fetch_data]   — Polygon.io news + market snapshot per ticker
  2. [analyze]      — Claude 4 Opus deep financial sentiment analysis
  3. [aggregate]    — merge per-ticker results into final response

When ANTHROPIC_API_KEY is absent the agent falls back to a
rule-based heuristic so development works without credentials.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from data.polygon_client import PolygonClient, TickerNews

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ── State schema ──────────────────────────────────────────────────────

class SentimentState(TypedDict, total=False):
    tickers: list[str]
    query: str | None
    news: dict[str, list[dict]]
    snapshots: dict[str, dict]
    results: dict[str, dict]
    error: str | None


# ── Node: fetch data from Polygon ────────────────────────────────────

async def fetch_data(state: SentimentState) -> SentimentState:
    """Pull news + snapshot for every requested ticker."""
    client = PolygonClient()
    news: dict[str, list[dict]] = {}
    snapshots: dict[str, dict] = {}

    try:
        for ticker in state["tickers"]:
            articles = await client.get_news(ticker, limit=10)
            news[ticker] = [
                {
                    "title": a.title,
                    "description": a.description,
                    "author": a.author,
                    "published_utc": a.published_utc,
                    "url": a.article_url,
                }
                for a in articles
            ]

            snap = await client.get_snapshot(ticker)
            snapshots[ticker] = {
                "price": snap.price,
                "change": snap.change,
                "change_pct": snap.change_pct,
                "volume": snap.volume,
                "prev_close": snap.prev_close,
            }
    finally:
        await client.close()

    return {**state, "news": news, "snapshots": snapshots}


# ── Node: Claude 4 Opus analysis ─────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior quantitative analyst at a top-tier hedge fund.
Analyse the provided financial news articles and market data for the
given ticker. Return ONLY valid JSON (no markdown fences) with this
exact structure:

{
  "overall_score": <int 0-100, 50=neutral, >50=bullish, <50=bearish>,
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": <int 0-100>,
  "key_factors": ["factor1", "factor2", ...],
  "top_articles": [
    {"title": "...", "url": "...", "sentiment": "positive|negative|neutral"}
  ]
}

Rules:
- Base your assessment on concrete facts in the articles, not speculation.
- Weight recent articles more heavily.
- Consider both company-specific and macro factors.
- If articles are contradictory, reflect that in a lower confidence score.
"""


async def analyze(state: SentimentState) -> SentimentState:
    """Run Claude 4 Opus on the collected news + market data."""
    results: dict[str, dict] = {}

    for ticker in state["tickers"]:
        articles = state.get("news", {}).get(ticker, [])
        snapshot = state.get("snapshots", {}).get(ticker, {})

        user_message = (
            f"Ticker: {ticker}\n\n"
            f"Market data: price=${snapshot.get('price', 'N/A')}, "
            f"change={snapshot.get('change_pct', 'N/A')}%, "
            f"volume={snapshot.get('volume', 'N/A')}\n\n"
            f"Recent articles ({len(articles)}):\n"
        )
        for i, a in enumerate(articles, 1):
            user_message += (
                f"\n--- Article {i} ---\n"
                f"Title: {a['title']}\n"
                f"Source: {a['author']}\n"
                f"Date: {a['published_utc']}\n"
                f"Summary: {a['description']}\n"
            )

        if ANTHROPIC_API_KEY:
            results[ticker] = await _call_claude(user_message)
        else:
            results[ticker] = _heuristic_analysis(ticker, articles, snapshot)

    return {**state, "results": results}


async def _call_claude(user_message: str) -> dict:
    """Send the analysis request to Claude 4 Opus via Anthropic API."""
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}],
            },
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["content"][0]["text"]

        # Claude may wrap JSON in markdown fences — strip them
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("\n", 1)
            text = parts[1] if len(parts) > 1 else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "overall_score": 50,
                "sentiment": "neutral",
                "confidence": 20,
                "key_factors": ["Claude response could not be parsed"],
                "top_articles": [],
            }


def _heuristic_analysis(
    ticker: str,
    articles: list[dict],
    snapshot: dict,
) -> dict:
    """Rule-based fallback when Claude is unavailable."""
    positive_keywords = [
        "beats", "upgrade", "buy", "rally", "growth", "record",
        "buyback", "expands", "strong", "exceeds", "raises",
    ]
    negative_keywords = [
        "downgrade", "sell", "decline", "loss", "concern",
        "investigation", "lawsuit", "regulatory", "warns", "miss",
    ]

    pos_count = 0
    neg_count = 0
    top_articles = []

    for a in articles:
        blob = (a.get("title", "") + " " + a.get("description", "")).lower()
        p = sum(1 for kw in positive_keywords if kw in blob)
        n = sum(1 for kw in negative_keywords if kw in blob)
        pos_count += p
        neg_count += n

        art_sent = "positive" if p > n else "negative" if n > p else "neutral"
        top_articles.append({
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "sentiment": art_sent,
        })

    total = pos_count + neg_count or 1
    raw_score = (pos_count - neg_count) / total
    overall_score = int(50 + raw_score * 40)
    overall_score = max(0, min(100, overall_score))

    change_pct = snapshot.get("change_pct", 0)
    if change_pct > 2:
        overall_score = min(100, overall_score + 8)
    elif change_pct < -2:
        overall_score = max(0, overall_score - 8)

    if overall_score >= 60:
        sentiment = "bullish"
    elif overall_score <= 40:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    confidence = min(95, 40 + len(articles) * 5 + abs(overall_score - 50))

    key_factors = []
    if pos_count > neg_count:
        key_factors.append(f"{pos_count} positive signals vs {neg_count} negative in recent news")
    elif neg_count > pos_count:
        key_factors.append(f"{neg_count} negative signals vs {pos_count} positive in recent news")
    if change_pct > 0:
        key_factors.append(f"Price up {change_pct:.1f}% on the day")
    elif change_pct < 0:
        key_factors.append(f"Price down {abs(change_pct):.1f}% on the day")
    if snapshot.get("volume", 0) > 50_000_000:
        key_factors.append("Above-average trading volume")

    return {
        "overall_score": overall_score,
        "sentiment": sentiment,
        "confidence": confidence,
        "key_factors": key_factors or ["Insufficient data for strong conviction"],
        "top_articles": top_articles[:5],
    }


# ── Node: aggregate ──────────────────────────────────────────────────

async def aggregate(state: SentimentState) -> SentimentState:
    """Final passthrough — results already structured per ticker."""
    return state


# ── Graph builder ─────────────────────────────────────────────────────

def build_sentiment_graph():
    """Compile the LangGraph sentiment pipeline."""
    graph = StateGraph(SentimentState)

    graph.add_node("fetch_data", fetch_data)
    graph.add_node("analyze", analyze)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "analyze")
    graph.add_edge("analyze", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


_compiled_sentiment_graph = None


def _get_sentiment_graph():
    global _compiled_sentiment_graph
    if _compiled_sentiment_graph is None:
        _compiled_sentiment_graph = build_sentiment_graph()
    return _compiled_sentiment_graph


# ── Public interface ──────────────────────────────────────────────────

async def run_sentiment_analysis(
    tickers: list[str] | None = None,
    query: str | None = None,
) -> dict:
    """
    Entry point called by the FastAPI endpoint.

    Accepts either a list of tickers or a free-text query.
    Returns a dict keyed by ticker with sentiment analysis.
    """
    if not tickers and query:
        tickers = [query.upper().replace("$", "").strip()]
    if not tickers:
        return {"error": "Provide at least one ticker or query"}

    tickers = [t.upper().strip() for t in tickers]

    graph = _get_sentiment_graph()
    final_state = await graph.ainvoke({
        "tickers": tickers,
        "query": query,
    })

    enriched: dict[str, dict] = {}
    for ticker, result in final_state.get("results", {}).items():
        enriched[ticker] = {
            **result,
            "ticker": ticker,
            "market_data": final_state.get("snapshots", {}).get(ticker, {}),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    return enriched
