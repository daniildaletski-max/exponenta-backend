import json
import os
from typing import Any, Dict

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "user_settings.json")

DEFAULT_SETTINGS = {
    "risk_tolerance": "moderate",
    "default_universe": "us_mega_cap",
    "currency": "USD",
    "watchlist": ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN"],
    "notifications": True,
    "theme": "dark",
    "auto_refresh": True,
    "refresh_interval": 60,
}


def load_settings() -> Dict[str, Any]:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return dict(DEFAULT_SETTINGS)
        merged = {**DEFAULT_SETTINGS, **saved}
        return merged
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    current = load_settings()
    current.update(settings)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(current, f, indent=2)
    return current
