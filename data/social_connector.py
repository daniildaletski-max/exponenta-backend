"""
X (Twitter) API v2 connector for social sentiment signals.

Fetches recent posts mentioning a ticker/cashtag and extracts
engagement metrics for sentiment weighting.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from config import get_settings

log = structlog.get_logger()


class SocialConnector:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        settings = get_settings()
        if not settings.twitter_bearer_token:
            log.warning("twitter.no_bearer_token")
            return None
        try:
            import tweepy
            self._client = tweepy.Client(bearer_token=settings.twitter_bearer_token)
            return self._client
        except ImportError:
            log.warning("tweepy.not_installed")
            return None

    def fetch_posts(self, ticker: str, max_results: int = 100) -> list[dict]:
        """
        Fetch recent X posts mentioning $TICKER cashtag.

        Returns list of dicts with keys: text, author, created_at,
        likes, retweets, replies.
        """
        client = self._get_client()

        if client is not None:
            query = f"${ticker} lang:en -is:retweet"
            tweets = client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=["created_at", "public_metrics", "author_id"],
            )
            results = []
            for tweet in tweets.data or []:
                metrics = tweet.public_metrics or {}
                results.append({
                    "text": tweet.text,
                    "author_id": tweet.author_id,
                    "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                    "likes": metrics.get("like_count", 0),
                    "retweets": metrics.get("retweet_count", 0),
                    "replies": metrics.get("reply_count", 0),
                })
            return results

        return [
            {
                "text": f"${ticker} looking bullish! Strong support at current levels.",
                "author_id": "stub",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "likes": 42,
                "retweets": 12,
                "replies": 5,
            },
        ]
