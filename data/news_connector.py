"""
NewsAPI connector for financial news headlines.

Fetches and filters news articles relevant to specific tickers/companies.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from config import get_settings

log = structlog.get_logger()


class NewsConnector:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        settings = get_settings()
        if not settings.news_api_key:
            log.warning("newsapi.no_api_key")
            return None
        try:
            from newsapi import NewsApiClient
            self._client = NewsApiClient(api_key=settings.news_api_key)
            return self._client
        except ImportError:
            log.warning("newsapi.not_installed")
            return None

    def fetch_headlines(self, query: str, max_results: int = 20) -> list[dict]:
        """
        Fetch recent news headlines matching a company/ticker query.

        Returns list of dicts with keys: title, source, url, published_at, description.
        """
        client = self._get_client()

        if client is not None:
            response = client.get_everything(
                q=query,
                language="en",
                sort_by="relevancy",
                page_size=max_results,
            )
            return [
                {
                    "title": a["title"],
                    "source": a["source"]["name"],
                    "url": a["url"],
                    "published_at": a["publishedAt"],
                    "description": a.get("description", ""),
                }
                for a in response.get("articles", [])
            ]

        return [
            {
                "title": f"{query} reports strong quarterly earnings",
                "source": "Financial Times",
                "url": "https://example.com",
                "published_at": datetime.now(timezone.utc).isoformat(),
                "description": "Stub headline for development.",
            },
        ]
