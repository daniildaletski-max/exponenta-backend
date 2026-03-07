import logging
import hashlib
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from prediction.cache_manager import SmartCache

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import yfinance as yf
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 3600

GLOBAL_UNIVERSE = {
    "us_mega_cap": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD",
        "ABBV", "MRK", "CRM", "AVGO", "PEP", "KO", "LLY",
    ],
    "us_growth": [
        "AMD", "PLTR", "CRWD", "DDOG", "NET", "PANW",
        "SHOP", "SQ", "COIN", "UBER", "ABNB", "SOFI", "HOOD",
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    ],
    "etf": [
        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI",
    ],
}

CROSS_ASSET_TICKERS = {
    "vix": "^VIX",
    "spy": "SPY",
    "sector_xlk": "XLK",
    "sector_xlf": "XLF",
    "sector_xle": "XLE",
    "sector_xlv": "XLV",
    "sector_xli": "XLI",
    "sector_xlp": "XLP",
    "sector_xly": "XLY",
}

ALL_SYMBOLS = []
for syms in GLOBAL_UNIVERSE.values():
    ALL_SYMBOLS.extend(syms)
ALL_SYMBOLS = list(set(ALL_SYMBOLS))

FORWARD_DAYS = 5
DEFAULT_LOOKBACK = 1260

_cross_asset_cache = SmartCache("cross_asset", max_size=50, default_ttl=1800)
_feature_matrix_cache = SmartCache("feature_matrix", max_size=20, default_ttl=900)


def _cache_path(symbol: str, days: int) -> Path:
    h = hashlib.md5(f"{symbol}_{days}_{datetime.now().strftime('%Y-%m-%d')}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.parquet"


def _fetch_ohlcv_polygon(symbol: str, days: int) -> Optional[pd.DataFrame]:
    try:
        import data.polygon_client as polygon_client
        if not polygon_client.is_polygon_available():
            return None
        df = polygon_client.get_aggregates_df(symbol, days=days)
        return df
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"Polygon OHLCV failed for {symbol}: {e}")
        return None


def _fetch_ohlcv_yfinance(symbol: str, days: int) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df is None or len(df) < 50:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return None

        df = df[required].dropna()
        df = df.sort_values("date").reset_index(drop=True)

        for col in ["open", "high", "low", "close"]:
            df = df[df[col] > 0]
        df["ret"] = df["close"].pct_change()
        df = df[(df["ret"].abs() < 0.5) | (df["ret"].isna())]
        df = df.drop(columns=["ret"])
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.reset_index(drop=True)

        if len(df) < 50:
            return None
        return df
    except Exception as e:
        logger.error(f"yfinance failed for {symbol}: {e}")
        return None


def fetch_ohlcv(symbol: str, days: int = DEFAULT_LOOKBACK) -> Optional[pd.DataFrame]:
    cp = _cache_path(symbol, days)
    if cp.exists():
        age = time.time() - os.path.getmtime(cp)
        if age < CACHE_TTL:
            try:
                return pd.read_parquet(cp)
            except Exception:
                pass

    df = _fetch_ohlcv_polygon(symbol, days)

    if df is None:
        df = _fetch_ohlcv_yfinance(symbol, days)

    if df is None:
        return None

    try:
        df.to_parquet(cp, index=False)
    except Exception:
        pass
    return df


def _fetch_cross_asset_data(days: int = DEFAULT_LOOKBACK) -> dict:
    cache_key = f"cross_asset_{days}"
    cached = _cross_asset_cache.get_adaptive(cache_key)
    if cached is not None:
        return cached

    result = {}

    def _fetch_one(key_ticker):
        key, ticker = key_ticker
        try:
            df = fetch_ohlcv(ticker, days=days)
            if df is not None and len(df) > 50:
                return key, df
        except Exception:
            pass
        return key, None

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = [executor.submit(_fetch_one, (key, ticker)) for key, ticker in CROSS_ASSET_TICKERS.items()]
        for future in as_completed(futures):
            try:
                key, df = future.result()
                if df is not None:
                    result[key] = df
            except Exception:
                pass

    _cross_asset_cache.set(cache_key, result)
    return result


def _detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    body = c - o
    body_abs = body.abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    candle_range = h - l

    avg_body = body_abs.rolling(20).mean()

    df["pattern_doji"] = ((body_abs < candle_range * 0.1) & (candle_range > 0)).astype(int)

    df["pattern_hammer"] = (
        (lower_wick > body_abs * 2) &
        (upper_wick < body_abs * 0.5) &
        (body_abs > 0)
    ).astype(int)

    df["pattern_shooting_star"] = (
        (upper_wick > body_abs * 2) &
        (lower_wick < body_abs * 0.5) &
        (body_abs > 0)
    ).astype(int)

    prev_body = body.shift(1)
    prev_o = o.shift(1)
    prev_c = c.shift(1)

    df["pattern_bullish_engulfing"] = (
        (prev_body < 0) &
        (body > 0) &
        (o <= prev_c) &
        (c >= prev_o) &
        (body_abs > prev_body.abs())
    ).astype(int)

    df["pattern_bearish_engulfing"] = (
        (prev_body > 0) &
        (body < 0) &
        (o >= prev_c) &
        (c <= prev_o) &
        (body_abs > prev_body.abs())
    ).astype(int)

    df["pattern_morning_star"] = 0
    df["pattern_evening_star"] = 0
    if len(df) > 2:
        body_2ago = body.shift(2)
        body_1ago = body.shift(1)
        body_1ago_abs = body_abs.shift(1)
        c_2ago = c.shift(2)

        morning = (
            (body_2ago < 0) &
            (body_1ago_abs < body_abs.shift(2).abs() * 0.3) &
            (body > 0) &
            (c > (o.shift(2) + c_2ago) / 2)
        )
        df["pattern_morning_star"] = morning.astype(int)

        evening = (
            (body_2ago > 0) &
            (body_1ago_abs < body_abs.shift(2).abs() * 0.3) &
            (body < 0) &
            (c < (o.shift(2) + c_2ago) / 2)
        )
        df["pattern_evening_star"] = evening.astype(int)

    df["pattern_dragonfly_doji"] = (
        (body_abs < candle_range * 0.1) &
        (lower_wick > candle_range * 0.6) &
        (upper_wick < candle_range * 0.1) &
        (candle_range > 0)
    ).astype(int)

    df["pattern_gravestone_doji"] = (
        (body_abs < candle_range * 0.1) &
        (upper_wick > candle_range * 0.6) &
        (lower_wick < candle_range * 0.1) &
        (candle_range > 0)
    ).astype(int)

    df["pattern_big_body"] = (body_abs > avg_body * 1.5).astype(int)

    return df


def _compute_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    df["vwap"] = cum_tp_vol / (cum_vol + 1e-10)
    df["price_vs_vwap"] = (close - df["vwap"]) / (df["vwap"] + 1e-10)

    daily_ret = close.pct_change()
    for w in [14, 20]:
        try:
            vol_weighted_gain = ((daily_ret.clip(lower=0)) * volume).rolling(w).sum()
            vol_weighted_loss = ((-daily_ret.clip(upper=0)) * volume).rolling(w).sum()
            df[f"vol_rsi_{w}"] = 100 - (100 / (1 + vol_weighted_gain / (vol_weighted_loss + 1e-10)))
        except Exception:
            pass

    for w in [10, 20, 50]:
        vol_sma = volume.rolling(w).mean()
        price_change = close.pct_change(w)
        df[f"accum_dist_ratio_{w}"] = (volume / (vol_sma + 1e-10)) * price_change.clip(-1, 1)

    df["volume_trend_5"] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
    df["volume_trend_20"] = volume.rolling(20).mean() / (volume.rolling(50).mean() + 1e-10)

    df["volume_price_confirm"] = (
        (close.pct_change() > 0) & (volume > volume.rolling(20).mean())
    ).astype(int) - (
        (close.pct_change() < 0) & (volume > volume.rolling(20).mean())
    ).astype(int)

    return df


def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    daily_ret = close.pct_change()

    for w in [10, 20, 60]:
        rolling_vol = daily_ret.rolling(w).std() * np.sqrt(252)
        df[f"vol_regime_{w}d"] = rolling_vol
        vol_median = rolling_vol.rolling(120).median()
        df[f"vol_regime_z_{w}d"] = (rolling_vol - vol_median) / (rolling_vol.rolling(120).std() + 1e-10)

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    df["trend_alignment"] = (
        (close > sma_20).astype(int) +
        (close > sma_50).astype(int) +
        (close > sma_200).astype(int) +
        (sma_20 > sma_50).astype(int) +
        (sma_50 > sma_200).astype(int)
    )

    df["trend_strength_composite"] = (
        df.get("adx", pd.Series(0, index=df.index)).fillna(0) / 100 * 0.3 +
        df["trend_alignment"] / 5 * 0.3 +
        (1 - daily_ret.rolling(20).std() / (daily_ret.rolling(20).std().rolling(120).quantile(0.95) + 1e-10)).clip(0, 1) * 0.2 +
        (close.pct_change(20).abs().clip(0, 0.2) / 0.2) * 0.2
    )

    df["mean_reversion_z"] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-10)

    for w in [20, 60]:
        rolling_skew = daily_ret.rolling(w).skew()
        rolling_kurt = daily_ret.rolling(w).kurt()
        df[f"return_skew_{w}d"] = rolling_skew
        df[f"return_kurtosis_{w}d"] = rolling_kurt

    highs_20 = close.rolling(20).max()
    df["drawdown_from_high_20"] = (close - highs_20) / (highs_20 + 1e-10)

    highs_60 = close.rolling(60).max()
    df["drawdown_from_high_60"] = (close - highs_60) / (highs_60 + 1e-10)

    return df


def _compute_cross_asset_features(df: pd.DataFrame, cross_data: dict) -> pd.DataFrame:
    if not cross_data:
        return df

    dates = pd.to_datetime(df["date"])
    close = df["close"]
    daily_ret = close.pct_change()

    if "vix" in cross_data:
        vix_df = cross_data["vix"]
        vix_close = vix_df.set_index("date")["close"].reindex(dates.values, method="ffill")
        df["vix_level"] = vix_close.values
        df["vix_sma_20"] = pd.Series(vix_close.values).rolling(20).mean().values
        df["vix_vs_sma"] = (df["vix_level"] - df["vix_sma_20"]) / (df["vix_sma_20"] + 1e-10)
        vix_ret = pd.Series(vix_close.values).pct_change()
        df["vix_corr_20"] = daily_ret.rolling(20).corr(vix_ret)
        df["vix_corr_60"] = daily_ret.rolling(60).corr(vix_ret)

    if "spy" in cross_data:
        spy_df = cross_data["spy"]
        spy_close = spy_df.set_index("date")["close"].reindex(dates.values, method="ffill")
        spy_ret = pd.Series(spy_close.values).pct_change()

        for w in [20, 60]:
            cov = daily_ret.rolling(w).cov(spy_ret)
            var = spy_ret.rolling(w).var()
            df[f"spy_beta_{w}d"] = cov / (var + 1e-10)
            df[f"spy_corr_{w}d"] = daily_ret.rolling(w).corr(spy_ret)

        df["spy_relative_strength_20"] = close.pct_change(20) - pd.Series(spy_close.values).pct_change(20)
        df["spy_relative_strength_60"] = close.pct_change(60) - pd.Series(spy_close.values).pct_change(60)

    for sector_key in ["sector_xlk", "sector_xlf", "sector_xle", "sector_xlv"]:
        if sector_key in cross_data:
            sec_df = cross_data[sector_key]
            sec_close = sec_df.set_index("date")["close"].reindex(dates.values, method="ffill")
            sec_ret = pd.Series(sec_close.values).pct_change()
            short_name = sector_key.replace("sector_", "")
            df[f"{short_name}_corr_20"] = daily_ret.rolling(20).corr(sec_ret)
            df[f"{short_name}_rs_20"] = close.pct_change(20) - pd.Series(sec_close.values).pct_change(20)

    return df


def _compute_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    for w in [5, 10, 21, 63]:
        df[f"weekly_ret_{w}d"] = close.pct_change(w)
        df[f"weekly_vol_{w}d"] = close.pct_change().rolling(w).std()
        df[f"weekly_high_{w}d"] = high.rolling(w).max()
        df[f"weekly_low_{w}d"] = low.rolling(w).min()
        df[f"weekly_range_{w}d"] = (df[f"weekly_high_{w}d"] - df[f"weekly_low_{w}d"]) / (close + 1e-10)
        df[f"price_pos_in_range_{w}d"] = (close - df[f"weekly_low_{w}d"]) / (df[f"weekly_high_{w}d"] - df[f"weekly_low_{w}d"] + 1e-10)

    for w in [5, 21]:
        df[f"avg_volume_{w}d"] = volume.rolling(w).mean()
        df[f"volume_change_{w}d"] = volume / (volume.rolling(w).mean() + 1e-10)

    df["monthly_momentum"] = close.pct_change(21)
    df["quarterly_momentum"] = close.pct_change(63)
    df["semi_annual_momentum"] = close.pct_change(126)

    df["momentum_acceleration"] = close.pct_change(21) - close.pct_change(21).shift(21)

    for w in [5, 21]:
        df[f"rsi_{w}d"] = RSIIndicator(close, window=w).rsi()

    df["intraday_range"] = (high - low) / (close + 1e-10)
    df["intraday_range_sma10"] = df["intraday_range"].rolling(10).mean()
    df["intraday_range_expansion"] = df["intraday_range"] / (df["intraday_range_sma10"] + 1e-10)

    df["annual_momentum"] = close.pct_change(252)
    df["momentum_ratio_short_long"] = close.pct_change(21) / (close.pct_change(63) + 1e-10)

    for w in [10, 30, 60]:
        df[f"close_vs_high_{w}d"] = (close - high.rolling(w).max()) / (close + 1e-10)
        df[f"close_vs_low_{w}d"] = (close - low.rolling(w).min()) / (close + 1e-10)

    df["gap_open"] = (df["open"] - close.shift(1)) / (close.shift(1) + 1e-10)
    df["gap_open_sma5"] = df["gap_open"].rolling(5).mean()

    df["ema_cross_5_20"] = (df.get("ema_5", close.ewm(span=5).mean()) - df.get("ema_20", close.ewm(span=20).mean())) / (close + 1e-10)
    df["ema_cross_20_50"] = (df.get("ema_20", close.ewm(span=20).mean()) - df.get("ema_50", close.ewm(span=50).mean())) / (close + 1e-10)

    df["volume_momentum_5"] = volume.pct_change(5)
    df["volume_momentum_20"] = volume.pct_change(20)

    return df


def _compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    df["bid_ask_proxy"] = (high - low) / (close + 1e-10)
    df["bid_ask_proxy_sma10"] = df["bid_ask_proxy"].rolling(10).mean()
    df["bid_ask_proxy_z"] = (df["bid_ask_proxy"] - df["bid_ask_proxy_sma10"]) / (df["bid_ask_proxy"].rolling(20).std() + 1e-10)

    daily_ret = close.pct_change()
    df["kyle_lambda"] = daily_ret.abs() / (volume + 1e-10) * 1e6
    df["kyle_lambda_sma20"] = df["kyle_lambda"].rolling(20).mean()

    df["amihud_illiq"] = daily_ret.abs() / (volume * close + 1e-10) * 1e9
    df["amihud_illiq_sma20"] = df["amihud_illiq"].rolling(20).mean()

    buy_vol = volume * ((close - low) / (high - low + 1e-10))
    sell_vol = volume * ((high - close) / (high - low + 1e-10))
    df["order_flow_imbalance"] = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-10)
    df["order_flow_imbalance_sma5"] = df["order_flow_imbalance"].rolling(5).mean()
    df["order_flow_imbalance_sma20"] = df["order_flow_imbalance"].rolling(20).mean()

    df["volume_clock"] = volume / (volume.rolling(20).mean() + 1e-10)
    df["volume_clock_std"] = df["volume_clock"].rolling(10).std()

    df["trade_intensity"] = volume / (df["bid_ask_proxy"] + 1e-10)
    df["trade_intensity_z"] = (df["trade_intensity"] - df["trade_intensity"].rolling(20).mean()) / (df["trade_intensity"].rolling(20).std() + 1e-10)

    df["price_impact_ratio"] = daily_ret.abs() / (np.log(volume + 1) + 1e-10)
    df["price_impact_ratio_sma10"] = df["price_impact_ratio"].rolling(10).mean()

    for w in [5, 10, 20]:
        cum_ofi = df["order_flow_imbalance"].rolling(w).sum()
        df[f"cum_ofi_{w}d"] = cum_ofi

    return df


def _compute_sentiment_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    try:
        from prediction.sentiment import _compute_technical_sentiment
        tech_sent = _compute_technical_sentiment(symbol)
        if tech_sent and isinstance(tech_sent, dict):
            score = float(tech_sent.get("score", 50))
            df["sentiment_technical_score"] = score / 100.0

            signals = tech_sent.get("signals", [])
            bullish_count = sum(1 for s in signals if s.get("type") == "bullish")
            bearish_count = sum(1 for s in signals if s.get("type") == "bearish")
            total_signals = bullish_count + bearish_count
            df["sentiment_signal_ratio"] = bullish_count / max(total_signals, 1)
            df["sentiment_signal_count"] = float(total_signals)

            indicators = tech_sent.get("indicators", {})
            for ind_name, ind_val in indicators.items():
                try:
                    df[f"sent_ind_{ind_name}"] = float(ind_val)
                except (ValueError, TypeError):
                    pass
        else:
            df["sentiment_technical_score"] = np.nan
            df["sentiment_signal_ratio"] = np.nan
            df["sentiment_signal_count"] = np.nan
    except Exception as e:
        logger.debug(f"Sentiment features failed for {symbol}: {e}")
        df["sentiment_technical_score"] = np.nan
        df["sentiment_signal_ratio"] = np.nan
        df["sentiment_signal_count"] = np.nan

    return df


def _fetch_fundamental_features(symbol: str) -> Optional[dict]:
    try:
        from prediction.fundamentals import (
            get_key_ratios, get_key_metrics, get_income_statements,
            get_analyst_estimates, get_company_overview
        )
        ratios = get_key_ratios(symbol)
        metrics = get_key_metrics(symbol)
        income = get_income_statements(symbol, "quarter", 8)
        estimates = get_analyst_estimates(symbol, 4)

        result = {}

        def _sf(val, default=np.nan):
            if val is None:
                return default
            try:
                v = float(val)
                return v if np.isfinite(v) else default
            except (ValueError, TypeError):
                return default

        result["fund_pe_ratio"] = _sf(ratios.get("pe_ratio") or metrics.get("pe_ratio"))
        result["fund_pb_ratio"] = _sf(ratios.get("pb_ratio") or metrics.get("pb_ratio"))
        result["fund_ps_ratio"] = _sf(ratios.get("ps_ratio"))
        result["fund_peg_ratio"] = _sf(ratios.get("peg_ratio"))
        result["fund_roe"] = _sf(ratios.get("roe"))
        result["fund_roa"] = _sf(ratios.get("roa"))
        result["fund_debt_to_equity"] = _sf(ratios.get("debt_to_equity"))
        result["fund_current_ratio"] = _sf(ratios.get("current_ratio"))
        result["fund_gross_margin"] = _sf(ratios.get("gross_margin"))
        result["fund_operating_margin"] = _sf(ratios.get("operating_margin"))
        result["fund_net_margin"] = _sf(ratios.get("net_margin"))
        result["fund_ev_to_ebitda"] = _sf(metrics.get("ev_to_ebitda"))
        result["fund_fcf_yield"] = _sf(metrics.get("fcf_yield"))
        result["fund_earnings_yield"] = _sf(metrics.get("earnings_yield"))

        if income and len(income) >= 2:
            rev_curr = _sf(income[0].get("revenue"), 0)
            rev_prev = _sf(income[1].get("revenue"), 0)
            if rev_prev and abs(rev_prev) > 1:
                result["fund_revenue_growth_qoq"] = (rev_curr - rev_prev) / abs(rev_prev) * 100
            else:
                result["fund_revenue_growth_qoq"] = np.nan

            if len(income) >= 5:
                rev_yoy = _sf(income[4].get("revenue"), 0)
                if rev_yoy and abs(rev_yoy) > 1:
                    result["fund_revenue_growth_yoy"] = (rev_curr - rev_yoy) / abs(rev_yoy) * 100
                else:
                    result["fund_revenue_growth_yoy"] = np.nan
            else:
                result["fund_revenue_growth_yoy"] = np.nan

            ni_curr = _sf(income[0].get("net_income"), 0)
            ni_prev = _sf(income[1].get("net_income"), 0)
            if ni_prev and abs(ni_prev) > 1:
                result["fund_earnings_growth_qoq"] = (ni_curr - ni_prev) / abs(ni_prev) * 100
            else:
                result["fund_earnings_growth_qoq"] = np.nan

            eps_curr = _sf(income[0].get("eps_diluted"), 0)
            eps_prev = _sf(income[1].get("eps_diluted"), 0)
            if eps_prev and abs(eps_prev) > 0.01:
                result["fund_eps_growth_qoq"] = (eps_curr - eps_prev) / abs(eps_prev) * 100
            else:
                result["fund_eps_growth_qoq"] = np.nan
        else:
            result["fund_revenue_growth_qoq"] = np.nan
            result["fund_revenue_growth_yoy"] = np.nan
            result["fund_earnings_growth_qoq"] = np.nan
            result["fund_eps_growth_qoq"] = np.nan

        if estimates and income and len(estimates) > 0:
            est = estimates[0]
            matching_inc = next((i for i in income if i.get("date") == est.get("date")), None)
            if matching_inc:
                actual_eps = _sf(matching_inc.get("eps_diluted"))
                est_eps = _sf(est.get("estimated_eps_avg"))
                if actual_eps is not None and est_eps and abs(est_eps) > 0.01:
                    result["fund_earnings_surprise_pct"] = (actual_eps - est_eps) / abs(est_eps) * 100
                else:
                    result["fund_earnings_surprise_pct"] = np.nan

                actual_rev = _sf(matching_inc.get("revenue"))
                est_rev = _sf(est.get("estimated_revenue_avg"))
                if actual_rev is not None and est_rev and abs(est_rev) > 1:
                    result["fund_revenue_surprise_pct"] = (actual_rev - est_rev) / abs(est_rev) * 100
                else:
                    result["fund_revenue_surprise_pct"] = np.nan
            else:
                result["fund_earnings_surprise_pct"] = np.nan
                result["fund_revenue_surprise_pct"] = np.nan

            if len(estimates) >= 2:
                curr_est_eps = _sf(estimates[0].get("estimated_eps_avg"))
                prev_est_eps = _sf(estimates[1].get("estimated_eps_avg"))
                if curr_est_eps is not None and prev_est_eps and abs(prev_est_eps) > 0.01:
                    result["fund_analyst_revision_momentum"] = (curr_est_eps - prev_est_eps) / abs(prev_est_eps) * 100
                else:
                    result["fund_analyst_revision_momentum"] = np.nan
            else:
                result["fund_analyst_revision_momentum"] = np.nan
        else:
            result["fund_earnings_surprise_pct"] = np.nan
            result["fund_revenue_surprise_pct"] = np.nan
            result["fund_analyst_revision_momentum"] = np.nan

        return result
    except Exception as e:
        logger.warning(f"Fundamental features fetch failed for {symbol}: {e}")
        return None


def _fetch_insider_intensity(symbol: str) -> Optional[dict]:
    try:
        from prediction.flow_tracker import get_smart_money_flow
        flow = get_smart_money_flow(symbol)
        result = {}

        insider_score = flow.get("insider_score", {})
        result["insider_buying_score"] = float(insider_score.get("score", 50))

        insider_summary = flow.get("insider_summary", {})
        result["insider_buy_value"] = float(insider_summary.get("total_buy_value", 0))
        result["insider_sell_value"] = float(insider_summary.get("total_sell_value", 0))
        total = result["insider_buy_value"] + result["insider_sell_value"]
        result["insider_buy_ratio"] = result["insider_buy_value"] / max(total, 1.0)

        clusters = flow.get("insider_clusters", [])
        result["insider_cluster_count"] = float(len(clusters) if clusters else 0)

        inst = flow.get("institutional_holders", {})
        result["institutional_ownership_pct"] = float(inst.get("institutional_ownership_pct", 0))
        result["institutional_change_qoq"] = float(inst.get("change_qoq_pct", 0))
        result["institutional_new_positions"] = float(inst.get("new_positions", 0))
        result["institutional_increased"] = float(inst.get("increased_positions", 0))
        result["institutional_decreased"] = float(inst.get("decreased_positions", 0))
        total_pos = result["institutional_increased"] + result["institutional_decreased"]
        result["institutional_net_flow_ratio"] = result["institutional_increased"] / max(total_pos, 1.0)

        return result
    except Exception as e:
        logger.warning(f"Insider intensity fetch failed for {symbol}: {e}")
        return None


def _apply_fundamental_features(df: pd.DataFrame, fund_data: Optional[dict], insider_data: Optional[dict]) -> pd.DataFrame:
    if fund_data:
        for key, val in fund_data.items():
            df[key] = val

    if insider_data:
        for key, val in insider_data.items():
            df[key] = val

    return df


def compute_features(df: pd.DataFrame, cross_data: Optional[dict] = None, fundamental_data: Optional[dict] = None, insider_data: Optional[dict] = None) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    for w in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        df[f"ret_{w}d"] = close.pct_change(w)
        df[f"vol_{w}d"] = close.pct_change().rolling(w).std()

    for w in [20, 50, 200]:
        sma = df.get(f"sma_{w}")
        if sma is not None:
            df[f"price_vs_sma{w}"] = (close - sma) / (sma + 1e-10)

    try:
        macd = MACD(close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
    except Exception:
        pass

    try:
        adx = ADXIndicator(high, low, close)
        df["adx"] = adx.adx()
        df["di_plus"] = adx.adx_pos()
        df["di_minus"] = adx.adx_neg()
    except Exception:
        pass

    try:
        cci = CCIIndicator(high, low, close)
        df["cci"] = cci.cci()
    except Exception:
        pass

    try:
        rsi = RSIIndicator(close)
        df["rsi"] = rsi.rsi()
    except Exception:
        pass

    try:
        stoch = StochasticOscillator(high, low, close)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
    except Exception:
        pass

    try:
        williams = WilliamsRIndicator(high, low, close)
        df["williams_r"] = williams.williams_r()
    except Exception:
        pass

    for p in [5, 10, 20]:
        try:
            roc = ROCIndicator(close, window=p)
            df[f"roc_{p}"] = roc.roc()
        except Exception:
            pass

    try:
        bb = BollingerBands(close)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["sma_20"] + 1e-10)
        df["bb_pctb"] = bb.bollinger_pband()
    except Exception:
        pass

    try:
        atr = AverageTrueRange(high, low, close)
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = df["atr"] / (close + 1e-10)
    except Exception:
        pass

    try:
        obv = OnBalanceVolumeIndicator(close, volume)
        df["obv"] = obv.on_balance_volume()
    except Exception:
        pass

    try:
        ad = AccDistIndexIndicator(high, low, close, volume)
        df["ad_line"] = ad.acc_dist_index()
    except Exception:
        pass

    df["volume_sma_20"] = volume.rolling(20).mean()
    df["relative_volume"] = volume / (df["volume_sma_20"] + 1)
    df["body_size"] = abs(close - df["open"]) / (close + 1e-10)
    df["upper_shadow"] = (high - df[["open", "close"]].max(axis=1)) / (close + 1e-10)
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - low) / (close + 1e-10)
    df["is_bullish"] = (close > df["open"]).astype(int)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month

    df = _detect_candlestick_patterns(df)
    df = _compute_volume_profile(df)
    df = _compute_multi_timeframe_features(df)
    df = _compute_regime_features(df)
    df = _compute_microstructure_features(df)

    if cross_data:
        df = _compute_cross_asset_features(df, cross_data)

    df = _apply_fundamental_features(df, fundamental_data, insider_data)

    df["fwd_ret_1d"] = close.pct_change(1).shift(-1)
    df["fwd_dir_1d"] = (df["fwd_ret_1d"] > 0).astype(int)
    df["fwd_ret_5d"] = close.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    df["fwd_dir_5d"] = (df["fwd_ret_5d"] > 0).astype(int)
    df["fwd_ret_20d"] = close.pct_change(20).shift(-20)
    df["fwd_dir_20d"] = (df["fwd_ret_20d"] > 0).astype(int)

    return df


def build_feature_matrix(symbol: str, days: int = DEFAULT_LOOKBACK) -> Optional[pd.DataFrame]:
    cache_key = f"{symbol}_{days}"
    cached = _feature_matrix_cache.get_adaptive(cache_key)
    if cached is not None:
        return cached

    df = fetch_ohlcv(symbol, days=days)
    if df is None:
        return None
    try:
        cross_data = _fetch_cross_asset_data(days=days)
    except Exception:
        cross_data = {}

    fundamental_data = None
    insider_data = None
    try:
        fundamental_data = _fetch_fundamental_features(symbol)
    except Exception:
        pass
    try:
        insider_data = _fetch_insider_intensity(symbol)
    except Exception:
        pass

    df = compute_features(df, cross_data=cross_data, fundamental_data=fundamental_data, insider_data=insider_data)

    try:
        df = _compute_sentiment_features(df, symbol)
    except Exception:
        pass

    _feature_matrix_cache.set(cache_key, df)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "open", "high", "low", "close", "volume", "dividends", "stock_splits", "capital_gains"}
    exclude.update({
        f"weekly_high_{w}d" for w in [5, 10, 21, 63]
    })
    exclude.update({
        f"weekly_low_{w}d" for w in [5, 10, 21, 63]
    })
    return [c for c in df.columns
            if c not in exclude
            and not c.startswith("fwd_")
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
