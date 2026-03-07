import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .features import fetch_ohlcv

logger = logging.getLogger(__name__)

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Communication Services",
}

RISK_ON_SECTORS = {"XLK", "XLY", "XLC"}
RISK_OFF_SECTORS = {"XLU", "XLP", "XLV"}


class MarketIntelligence:

    def sector_rotation_analysis(self) -> dict:
        try:
            sectors = []
            for etf, name in SECTOR_ETFS.items():
                df = fetch_ohlcv(etf, days=120)
                if df is None or len(df) < 22:
                    continue
                close = df["close"].values
                ret_1m = float((close[-1] / close[-22] - 1) * 100) if len(close) >= 22 else 0.0
                ret_3m = float((close[-1] / close[-66] - 1) * 100) if len(close) >= 66 else 0.0
                sectors.append({
                    "etf": etf,
                    "name": name,
                    "return_1m": round(ret_1m, 2),
                    "return_3m": round(ret_3m, 2),
                })

            if not sectors:
                return {"sectors": [], "rotation_signal": "unknown", "leaders": [], "laggards": []}

            sorted_by_1m = sorted(sectors, key=lambda x: x["return_1m"], reverse=True)
            leaders = sorted_by_1m[:3]
            laggards = sorted_by_1m[-3:]

            leader_etfs = {s["etf"] for s in leaders}
            risk_on_leading = len(leader_etfs & RISK_ON_SECTORS)
            risk_off_leading = len(leader_etfs & RISK_OFF_SECTORS)

            if risk_on_leading >= 2:
                rotation_signal = "risk-on"
            elif risk_off_leading >= 2:
                rotation_signal = "risk-off"
            else:
                rotation_signal = "mixed"

            return {
                "sectors": sectors,
                "rotation_signal": rotation_signal,
                "leaders": [{"etf": s["etf"], "name": s["name"], "return_1m": s["return_1m"]} for s in leaders],
                "laggards": [{"etf": s["etf"], "name": s["name"], "return_1m": s["return_1m"]} for s in laggards],
            }
        except Exception as e:
            logger.error(f"sector_rotation_analysis error: {e}")
            return {"sectors": [], "rotation_signal": "unknown", "leaders": [], "laggards": []}

    def market_breadth(self, tickers: list[str]) -> dict:
        try:
            above_ma = 0
            positive_today = 0
            changes = []
            valid = 0

            for ticker in tickers:
                df = fetch_ohlcv(ticker, days=60)
                if df is None or len(df) < 22:
                    continue
                valid += 1
                close = df["close"].values
                ma20 = float(np.mean(np.array(close[-20:])))
                current = float(close[-1])
                prev = float(close[-2]) if len(close) >= 2 else current

                if current > ma20:
                    above_ma += 1
                daily_change = (current / prev - 1) * 100 if prev > 0 else 0
                if daily_change > 0:
                    positive_today += 1
                changes.append(daily_change)

            if valid == 0:
                return {"breadth_score": 50, "pct_above_ma": 50.0, "pct_positive": 50.0, "avg_change": 0.0, "interpretation": "No data available"}

            pct_above_ma = round(above_ma / valid * 100, 1)
            pct_positive = round(positive_today / valid * 100, 1)
            avg_change = round(float(np.mean(changes)), 2) if changes else 0.0
            breadth_score = round(pct_above_ma * 0.6 + pct_positive * 0.4, 1)
            breadth_score = max(0, min(100, breadth_score))

            if breadth_score >= 70:
                interpretation = "Strong breadth — broad market participation, bullish"
            elif breadth_score >= 50:
                interpretation = "Moderate breadth — average participation"
            elif breadth_score >= 30:
                interpretation = "Weak breadth — narrow market, caution advised"
            else:
                interpretation = "Very weak breadth — most stocks declining, bearish signal"

            return {
                "breadth_score": breadth_score,
                "pct_above_ma": pct_above_ma,
                "pct_positive": pct_positive,
                "avg_change": avg_change,
                "interpretation": interpretation,
            }
        except Exception as e:
            logger.error(f"market_breadth error: {e}")
            return {"breadth_score": 50, "pct_above_ma": 50.0, "pct_positive": 50.0, "avg_change": 0.0, "interpretation": "Error calculating breadth"}

    def volatility_regime(self) -> dict:
        try:
            df = fetch_ohlcv("SPY", days=120)
            if df is None or len(df) < 30:
                return {"regime": "unknown", "volatility_pct": 0.0, "trend": "stable"}

            close = df["close"].values
            returns = np.diff(np.log(close))

            vol_20 = float(np.std(returns[-20:]) * np.sqrt(252) * 100)

            vol_10 = float(np.std(returns[-10:]) * np.sqrt(252) * 100) if len(returns) >= 10 else vol_20
            vol_prev_20 = float(np.std(returns[-40:-20]) * np.sqrt(252) * 100) if len(returns) >= 40 else vol_20

            if vol_20 < 12:
                regime = "low"
            elif vol_20 < 20:
                regime = "normal"
            elif vol_20 < 30:
                regime = "elevated"
            else:
                regime = "extreme"

            if vol_10 > vol_prev_20 * 1.15:
                trend = "increasing"
            elif vol_10 < vol_prev_20 * 0.85:
                trend = "decreasing"
            else:
                trend = "stable"

            return {
                "regime": regime,
                "volatility_pct": round(vol_20, 2),
                "trend": trend,
            }
        except Exception as e:
            logger.error(f"volatility_regime error: {e}")
            return {"regime": "unknown", "volatility_pct": 0.0, "trend": "stable"}

    def correlation_regime(self, tickers: list[str]) -> dict:
        try:
            if len(tickers) < 2:
                return {"avg_correlation": 0.0, "regime": "normal", "risk_flag": False}

            price_data = {}
            for t in tickers[:20]:
                df = fetch_ohlcv(t, days=60)
                if df is not None and len(df) >= 30:
                    price_data[t] = df.set_index("date")["close"].tail(30)

            if len(price_data) < 2:
                return {"avg_correlation": 0.0, "regime": "normal", "risk_flag": False}

            prices_df = pd.DataFrame(price_data).dropna()
            if len(prices_df) < 10:
                return {"avg_correlation": 0.0, "regime": "normal", "risk_flag": False}

            returns_df = prices_df.pct_change().dropna()
            corr = returns_df.corr()
            cols = list(corr.columns)

            pairwise = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairwise.append(float(corr.iloc[i, j]))

            if not pairwise:
                return {"avg_correlation": 0.0, "regime": "normal", "risk_flag": False}

            avg_corr = round(float(np.mean(pairwise)), 3)

            if avg_corr > 0.7:
                regime = "herding"
                risk_flag = True
            elif avg_corr < 0.2:
                regime = "diverging"
                risk_flag = False
            else:
                regime = "normal"
                risk_flag = False

            return {
                "avg_correlation": avg_corr,
                "regime": regime,
                "risk_flag": risk_flag,
            }
        except Exception as e:
            logger.error(f"correlation_regime error: {e}")
            return {"avg_correlation": 0.0, "regime": "normal", "risk_flag": False}

    def generate_intelligence_report(self, tickers: list[str]) -> dict:
        try:
            sector_rotation = self.sector_rotation_analysis()
            breadth = self.market_breadth(tickers)
            volatility = self.volatility_regime()
            correlation = self.correlation_regime(tickers)

            score = 50.0
            if breadth["breadth_score"] >= 60:
                score += 15
            elif breadth["breadth_score"] < 40:
                score -= 15

            rotation = sector_rotation.get("rotation_signal", "mixed")
            if rotation == "risk-on":
                score += 10
            elif rotation == "risk-off":
                score -= 10

            vol_regime = volatility.get("regime", "normal")
            if vol_regime == "low":
                score += 10
            elif vol_regime == "elevated":
                score -= 10
            elif vol_regime == "extreme":
                score -= 20

            if correlation.get("risk_flag", False):
                score -= 10

            if volatility.get("trend") == "increasing":
                score -= 5
            elif volatility.get("trend") == "decreasing":
                score += 5

            score = max(0, min(100, round(score, 1)))

            if score >= 70:
                market_regime = "bullish"
            elif score >= 55:
                market_regime = "neutral"
            elif score >= 40:
                market_regime = "cautious"
            else:
                market_regime = "bearish"

            insights = []

            if rotation == "risk-on":
                leaders = sector_rotation.get("leaders", [])
                leader_names = ", ".join(s["name"] for s in leaders[:2]) if leaders else "growth sectors"
                insights.append(f"Risk-on rotation detected — {leader_names} leading, favoring growth exposure")
            elif rotation == "risk-off":
                insights.append("Defensive rotation underway — utilities and staples outperforming, consider reducing risk")

            bs = breadth["breadth_score"]
            if bs >= 70:
                insights.append(f"Broad market participation ({breadth['pct_above_ma']:.0f}% above 20-day MA) supports uptrend continuation")
            elif bs < 40:
                insights.append(f"Narrow breadth warning — only {breadth['pct_above_ma']:.0f}% of stocks above 20-day MA")

            if vol_regime == "elevated" or vol_regime == "extreme":
                insights.append(f"Volatility is {vol_regime} at {volatility['volatility_pct']:.1f}% annualized — tighten stops and reduce position sizes")
            elif vol_regime == "low":
                insights.append(f"Low volatility ({volatility['volatility_pct']:.1f}%) environment — watch for a potential vol expansion breakout")

            if correlation.get("risk_flag"):
                insights.append(f"High correlation ({correlation['avg_correlation']:.2f}) across holdings — diversification benefits reduced, systemic risk elevated")

            if volatility.get("trend") == "increasing" and vol_regime != "low":
                insights.append("Volatility trend is rising — market uncertainty increasing")

            if not insights:
                insights.append("Market conditions are balanced — no strong directional signals detected")

            insights = insights[:5]

            return {
                "sector_rotation": sector_rotation,
                "breadth": breadth,
                "volatility": volatility,
                "correlation": correlation,
                "overall_market_score": score,
                "market_regime": market_regime,
                "key_insights": insights,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"generate_intelligence_report error: {e}")
            return {
                "sector_rotation": {"sectors": [], "rotation_signal": "unknown", "leaders": [], "laggards": []},
                "breadth": {"breadth_score": 50, "pct_above_ma": 50.0, "pct_positive": 50.0, "avg_change": 0.0, "interpretation": "Error"},
                "volatility": {"regime": "unknown", "volatility_pct": 0.0, "trend": "stable"},
                "correlation": {"avg_correlation": 0.0, "regime": "normal", "risk_flag": False},
                "overall_market_score": 50,
                "market_regime": "neutral",
                "key_insights": ["Unable to generate full intelligence report"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
