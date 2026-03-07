import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    """Supported backtesting strategies.

    String-based so JSON serialisation and backward-compatible string
    comparisons (``strategy == "momentum"``) keep working.
    """
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    MACD = "macd"
    ML_SIGNAL = "ml_signal"


@dataclass
class TransactionCosts:
    """Per-trade cost model.

    Attributes:
        commission_per_share: Fixed commission in USD charged per share
            on both entry and exit.  Default $0.005 (Interactive Brokers-like).
        spread_pct: Half-spread as a fraction of price.  Applied on each
            side of the trade (buy at ask = mid + spread, sell at bid =
            mid - spread).  Default 0.02% (2 bps).
        slippage_pct: Market-impact / slippage as a fraction of price.
            Applied on each side.  Default 0.05% (5 bps).
    """
    commission_per_share: float = 0.005
    spread_pct: float = 0.0002     # 0.02%
    slippage_pct: float = 0.0005   # 0.05%

    # -- convenience helpers ------------------------------------------------

    def effective_buy_price(self, mid_price: float) -> float:
        """Price actually paid when *buying* (worse = higher)."""
        return mid_price * (1.0 + self.spread_pct + self.slippage_pct)

    def effective_sell_price(self, mid_price: float) -> float:
        """Price actually received when *selling* (worse = lower)."""
        return mid_price * (1.0 - self.spread_pct - self.slippage_pct)

    def round_trip_cost(self, mid_price: float, shares: float) -> float:
        """Total dollar cost for a full round-trip trade of *shares* shares."""
        spread_slip = mid_price * (self.spread_pct + self.slippage_pct) * 2.0
        commission = self.commission_per_share * shares * 2.0
        return spread_slip * shares + commission


@dataclass
class TradeRecord:
    """Detailed record for a single completed (or still-open) trade."""
    entry_date: str
    entry_price: float
    entry_price_effective: float  # after costs
    entry_reason: str
    exit_date: str
    exit_price: float
    exit_price_effective: float   # after costs
    exit_reason: str
    shares: float
    pnl_dollar: float             # net of costs
    pnl_pct: float                # net of costs
    gross_pnl_dollar: float       # before costs
    gross_pnl_pct: float          # before costs
    transaction_costs: float      # total dollar costs for this trade
    holding_days: int
    is_open: bool = False


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_history(ticker: str, days: int = 756) -> Optional[pd.DataFrame]:
    from prediction.features import fetch_ohlcv
    df = fetch_ohlcv(ticker, days=days)
    if df is not None and len(df) > 30:
        return df
    return None


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["atr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1).rolling(14).mean()

    return df


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------
# Each strategy function returns a list of *raw* signal dicts with at least:
#   entry_date, entry_price, entry_reason, entry_idx,
#   exit_date, exit_price, exit_reason, holding_days
# Transaction costs are applied *after* by the equity-curve builder.

def _run_momentum_strategy(df: pd.DataFrame, params: dict) -> list:
    rsi_buy = params.get("rsi_buy", 30)
    rsi_sell = params.get("rsi_sell", 70)
    trades: list = []
    position: Optional[dict] = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is None:
            if pd.notna(prev["rsi"]) and prev["rsi"] < rsi_buy and row["rsi"] >= rsi_buy:
                position = {
                    "entry_date": str(row["date"])[:10],
                    "entry_price": float(row["close"]),
                    "entry_reason": f"RSI crossed above {rsi_buy}",
                    "entry_idx": i,
                }
        else:
            if pd.notna(prev["rsi"]) and prev["rsi"] > rsi_sell and row["rsi"] <= rsi_sell:
                pnl_pct = (row["close"] - position["entry_price"]) / position["entry_price"] * 100
                trades.append({
                    **position,
                    "exit_date": str(row["date"])[:10],
                    "exit_price": float(row["close"]),
                    "exit_reason": f"RSI crossed below {rsi_sell}",
                    "pnl_pct": round(float(pnl_pct), 2),
                    "pnl_dollar": round(float(row["close"] - position["entry_price"]), 2),
                    "holding_days": i - position["entry_idx"],
                })
                position = None

    if position is not None:
        last = df.iloc[-1]
        pnl_pct = (last["close"] - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            **position,
            "exit_date": str(last["date"])[:10],
            "exit_price": float(last["close"]),
            "exit_reason": "Still open",
            "pnl_pct": round(float(pnl_pct), 2),
            "pnl_dollar": round(float(last["close"] - position["entry_price"]), 2),
            "holding_days": len(df) - 1 - position["entry_idx"],
        })

    return trades


def _run_mean_reversion_strategy(df: pd.DataFrame, params: dict) -> list:
    bb_period = params.get("bb_period", 20)
    bb_std = params.get("bb_std", 2.0)
    trades: list = []
    position: Optional[dict] = None

    for i in range(1, len(df)):
        row = df.iloc[i]

        if position is None:
            if pd.notna(row["bb_lower"]) and row["close"] < row["bb_lower"]:
                position = {
                    "entry_date": str(row["date"])[:10],
                    "entry_price": float(row["close"]),
                    "entry_reason": f"Price below lower BB ({bb_std}\u03c3)",
                    "entry_idx": i,
                }
        else:
            if pd.notna(row["bb_mid"]) and row["close"] > row["bb_mid"]:
                pnl_pct = (row["close"] - position["entry_price"]) / position["entry_price"] * 100
                trades.append({
                    **position,
                    "exit_date": str(row["date"])[:10],
                    "exit_price": float(row["close"]),
                    "exit_reason": "Price returned to BB midline",
                    "pnl_pct": round(float(pnl_pct), 2),
                    "pnl_dollar": round(float(row["close"] - position["entry_price"]), 2),
                    "holding_days": i - position["entry_idx"],
                })
                position = None

    if position is not None:
        last = df.iloc[-1]
        pnl_pct = (last["close"] - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            **position,
            "exit_date": str(last["date"])[:10],
            "exit_price": float(last["close"]),
            "exit_reason": "Still open",
            "pnl_pct": round(float(pnl_pct), 2),
            "pnl_dollar": round(float(last["close"] - position["entry_price"]), 2),
            "holding_days": len(df) - 1 - position["entry_idx"],
        })

    return trades


def _run_trend_following_strategy(df: pd.DataFrame, params: dict) -> list:
    fast_ma = params.get("fast_ma", 20)
    slow_ma = params.get("slow_ma", 50)
    trades: list = []
    position: Optional[dict] = None

    fast = df["close"].rolling(fast_ma).mean()
    slow = df["close"].rolling(slow_ma).mean()

    for i in range(1, len(df)):
        row = df.iloc[i]

        if position is None:
            if pd.notna(fast.iloc[i]) and pd.notna(slow.iloc[i]) and pd.notna(fast.iloc[i-1]) and pd.notna(slow.iloc[i-1]):
                if fast.iloc[i-1] <= slow.iloc[i-1] and fast.iloc[i] > slow.iloc[i]:
                    position = {
                        "entry_date": str(row["date"])[:10],
                        "entry_price": float(row["close"]),
                        "entry_reason": f"SMA{fast_ma} crossed above SMA{slow_ma}",
                        "entry_idx": i,
                    }
        else:
            if pd.notna(fast.iloc[i]) and pd.notna(slow.iloc[i]) and pd.notna(fast.iloc[i-1]) and pd.notna(slow.iloc[i-1]):
                if fast.iloc[i-1] >= slow.iloc[i-1] and fast.iloc[i] < slow.iloc[i]:
                    pnl_pct = (row["close"] - position["entry_price"]) / position["entry_price"] * 100
                    trades.append({
                        **position,
                        "exit_date": str(row["date"])[:10],
                        "exit_price": float(row["close"]),
                        "exit_reason": f"SMA{fast_ma} crossed below SMA{slow_ma}",
                        "pnl_pct": round(float(pnl_pct), 2),
                        "pnl_dollar": round(float(row["close"] - position["entry_price"]), 2),
                        "holding_days": i - position["entry_idx"],
                    })
                    position = None

    if position is not None:
        last = df.iloc[-1]
        pnl_pct = (last["close"] - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            **position,
            "exit_date": str(last["date"])[:10],
            "exit_price": float(last["close"]),
            "exit_reason": "Still open",
            "pnl_pct": round(float(pnl_pct), 2),
            "pnl_dollar": round(float(last["close"] - position["entry_price"]), 2),
            "holding_days": len(df) - 1 - position["entry_idx"],
        })

    return trades


def _run_macd_strategy(df: pd.DataFrame, params: dict) -> list:
    trades: list = []
    position: Optional[dict] = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is None:
            if pd.notna(row["macd_hist"]) and pd.notna(prev["macd_hist"]):
                if prev["macd_hist"] <= 0 and row["macd_hist"] > 0:
                    position = {
                        "entry_date": str(row["date"])[:10],
                        "entry_price": float(row["close"]),
                        "entry_reason": "MACD histogram turned positive",
                        "entry_idx": i,
                    }
        else:
            if pd.notna(row["macd_hist"]) and pd.notna(prev["macd_hist"]):
                if prev["macd_hist"] >= 0 and row["macd_hist"] < 0:
                    pnl_pct = (row["close"] - position["entry_price"]) / position["entry_price"] * 100
                    trades.append({
                        **position,
                        "exit_date": str(row["date"])[:10],
                        "exit_price": float(row["close"]),
                        "exit_reason": "MACD histogram turned negative",
                        "pnl_pct": round(float(pnl_pct), 2),
                        "pnl_dollar": round(float(row["close"] - position["entry_price"]), 2),
                        "holding_days": i - position["entry_idx"],
                    })
                    position = None

    if position is not None:
        last = df.iloc[-1]
        pnl_pct = (last["close"] - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            **position,
            "exit_date": str(last["date"])[:10],
            "exit_price": float(last["close"]),
            "exit_reason": "Still open",
            "pnl_pct": round(float(pnl_pct), 2),
            "pnl_dollar": round(float(last["close"] - position["entry_price"]), 2),
            "holding_days": len(df) - 1 - position["entry_idx"],
        })

    return trades


def _run_ml_signal_strategy(df: pd.DataFrame, params: dict) -> list:
    """Use the PredictionEngine ensemble scores to generate long signals.

    The strategy recomputes a lightweight feature set per bar and uses
    the engine's ``predict`` method.  Because the engine may require
    network calls (sentiment, etc.) and that would be prohibitively slow
    bar-by-bar over hundreds of days, we fall back to a *simulated*
    ML signal based on a composite of RSI + MACD + Bollinger %B that
    mimics the ensemble output for backtesting purposes.

    Parameters via ``params``:
        buy_threshold  (float): composite score above which we enter
            (default 0.65).
        sell_threshold (float): composite score below which we exit
            (default 0.40).
    """
    buy_threshold = params.get("buy_threshold", 0.65)
    sell_threshold = params.get("sell_threshold", 0.40)

    # Build composite signal [0..1] from available indicators
    scores = np.full(len(df), np.nan)
    for i in range(len(df)):
        row = df.iloc[i]
        parts: list[float] = []

        # RSI component: 0 when oversold (buy), 1 when overbought
        if pd.notna(row.get("rsi")):
            parts.append(np.clip((100.0 - row["rsi"]) / 100.0, 0, 1))

        # MACD histogram component: positive = bullish
        if pd.notna(row.get("macd_hist")) and pd.notna(row.get("atr")) and row["atr"] > 0:
            macd_norm = np.clip(row["macd_hist"] / row["atr"], -1, 1)
            parts.append((macd_norm + 1) / 2.0)

        # Bollinger %B component: low = oversold
        if pd.notna(row.get("bb_lower")) and pd.notna(row.get("bb_upper")):
            bb_range = row["bb_upper"] - row["bb_lower"]
            if bb_range > 0:
                pct_b = (row["close"] - row["bb_lower"]) / bb_range
                parts.append(np.clip(1.0 - pct_b, 0, 1))

        # SMA trend component
        if pd.notna(row.get("sma50")) and row["sma50"] > 0:
            trend = row["close"] / row["sma50"] - 1.0
            parts.append(np.clip((trend + 0.1) / 0.2, 0, 1))

        if parts:
            scores[i] = float(np.mean(parts))

    trades: list = []
    position: Optional[dict] = None

    for i in range(1, len(df)):
        score = scores[i]
        if np.isnan(score):
            continue
        row = df.iloc[i]

        if position is None:
            if score >= buy_threshold:
                position = {
                    "entry_date": str(row["date"])[:10],
                    "entry_price": float(row["close"]),
                    "entry_reason": f"ML composite score {score:.2f} >= {buy_threshold}",
                    "entry_idx": i,
                }
        else:
            if score <= sell_threshold:
                pnl_pct = (row["close"] - position["entry_price"]) / position["entry_price"] * 100
                trades.append({
                    **position,
                    "exit_date": str(row["date"])[:10],
                    "exit_price": float(row["close"]),
                    "exit_reason": f"ML composite score {score:.2f} <= {sell_threshold}",
                    "pnl_pct": round(float(pnl_pct), 2),
                    "pnl_dollar": round(float(row["close"] - position["entry_price"]), 2),
                    "holding_days": i - position["entry_idx"],
                })
                position = None

    if position is not None:
        last = df.iloc[-1]
        pnl_pct = (last["close"] - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            **position,
            "exit_date": str(last["date"])[:10],
            "exit_price": float(last["close"]),
            "exit_reason": "Still open",
            "pnl_pct": round(float(pnl_pct), 2),
            "pnl_dollar": round(float(last["close"] - position["entry_price"]), 2),
            "holding_days": len(df) - 1 - position["entry_idx"],
        })

    return trades


# ---------------------------------------------------------------------------
# Strategy registry  (backward-compatible dict + new enum)
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "momentum": _run_momentum_strategy,
    "mean_reversion": _run_mean_reversion_strategy,
    "trend_following": _run_trend_following_strategy,
    "macd": _run_macd_strategy,
    "ml_signal": _run_ml_signal_strategy,
}

STRATEGY_DEFAULTS: Dict[str, dict] = {
    "momentum": {"rsi_buy": 30, "rsi_sell": 70},
    "mean_reversion": {"bb_period": 20, "bb_std": 2.0},
    "trend_following": {"fast_ma": 20, "slow_ma": 50},
    "macd": {},
    "ml_signal": {"buy_threshold": 0.65, "sell_threshold": 0.40},
}


# ---------------------------------------------------------------------------
# Trade-log builder  (enriches raw signal trades with cost-aware P&L)
# ---------------------------------------------------------------------------

def _build_trade_log(
    raw_trades: list,
    initial_capital: float,
    costs: TransactionCosts,
) -> List[TradeRecord]:
    """Convert raw strategy trades into ``TradeRecord`` objects that
    include effective prices, transaction costs and net P&L."""
    records: List[TradeRecord] = []
    capital = initial_capital

    for t in raw_trades:
        entry_mid = t["entry_price"]
        exit_mid = t["exit_price"]
        is_open = t["exit_reason"] == "Still open"

        entry_eff = costs.effective_buy_price(entry_mid)
        exit_eff = costs.effective_sell_price(exit_mid) if not is_open else exit_mid

        shares = capital / entry_eff

        gross_pnl = (exit_mid - entry_mid) * shares
        gross_pnl_pct = (exit_mid - entry_mid) / entry_mid * 100.0 if entry_mid else 0.0

        if is_open:
            txn_cost = costs.commission_per_share * shares  # only entry side
            txn_cost += entry_mid * (costs.spread_pct + costs.slippage_pct) * shares
        else:
            txn_cost = costs.round_trip_cost(entry_mid, shares)

        net_pnl = (exit_eff - entry_eff) * shares if not is_open else gross_pnl - txn_cost
        net_pnl_pct = net_pnl / (entry_eff * shares) * 100.0 if (entry_eff * shares) else 0.0

        records.append(TradeRecord(
            entry_date=t["entry_date"],
            entry_price=round(entry_mid, 4),
            entry_price_effective=round(entry_eff, 4),
            entry_reason=t["entry_reason"],
            exit_date=t["exit_date"],
            exit_price=round(exit_mid, 4),
            exit_price_effective=round(exit_eff, 4),
            exit_reason=t["exit_reason"],
            shares=round(shares, 4),
            pnl_dollar=round(net_pnl, 2),
            pnl_pct=round(net_pnl_pct, 2),
            gross_pnl_dollar=round(gross_pnl, 2),
            gross_pnl_pct=round(gross_pnl_pct, 2),
            transaction_costs=round(txn_cost, 2),
            holding_days=t["holding_days"],
            is_open=is_open,
        ))

        # Roll capital forward for next trade sizing
        if not is_open:
            capital = capital + net_pnl

    return records


# ---------------------------------------------------------------------------
# Equity curve (cost-aware)
# ---------------------------------------------------------------------------

def _build_equity_curve(
    df: pd.DataFrame,
    trades: list,
    initial_capital: float = 10000,
    costs: Optional[TransactionCosts] = None,
) -> list:
    """Build a daily equity curve accounting for transaction costs.

    When *costs* is ``None``, behaviour is identical to the legacy
    implementation (no friction applied).
    """
    if costs is None:
        costs = TransactionCosts(commission_per_share=0, spread_pct=0, slippage_pct=0)

    equity: list = []
    capital = initial_capital
    position_open = False
    shares = 0.0

    trade_entries: Dict[str, dict] = {}
    trade_exits: Dict[str, dict] = {}
    for t in trades:
        trade_entries[t["entry_date"]] = t
        if t.get("exit_reason") != "Still open":
            trade_exits[t["exit_date"]] = t

    for i in range(len(df)):
        date_str = str(df.iloc[i]["date"])[:10]
        price = float(df.iloc[i]["close"])

        if date_str in trade_entries and not position_open:
            buy_price = costs.effective_buy_price(price)
            shares = capital / buy_price
            capital = 0.0  # fully invested
            # Deduct entry commission
            shares -= (costs.commission_per_share * shares) / buy_price if buy_price > 0 else 0
            position_open = True

        if date_str in trade_exits and position_open:
            sell_price = costs.effective_sell_price(price)
            capital = shares * sell_price
            # Deduct exit commission
            capital -= costs.commission_per_share * shares
            shares = 0.0
            position_open = False

        current_value = (shares * price) if position_open else capital
        equity.append({
            "date": date_str,
            "value": round(float(current_value), 2),
        })

    return equity


def _build_benchmark_equity(df: pd.DataFrame, initial_capital: float = 10000) -> list:
    if len(df) == 0:
        return []
    first_price = float(df.iloc[0]["close"])
    equity: list = []
    for i in range(len(df)):
        price = float(df.iloc[i]["close"])
        value = initial_capital * (price / first_price)
        equity.append({
            "date": str(df.iloc[i]["date"])[:10],
            "value": round(float(value), 2),
        })
    return equity


# ---------------------------------------------------------------------------
# Metrics (risk-adjusted, using trade log when available)
# ---------------------------------------------------------------------------

def _compute_metrics(
    equity_curve: list,
    trades: list,
    benchmark_equity: list,
    trading_days: int = 252,
    trade_log: Optional[List[TradeRecord]] = None,
) -> dict:
    if not equity_curve or len(equity_curve) < 2:
        return {}

    values = np.array([e["value"] for e in equity_curve])
    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]

    total_return = (values[-1] - values[0]) / values[0] * 100
    n_years = len(values) / trading_days
    cagr = ((values[-1] / values[0]) ** (1 / max(n_years, 0.01)) - 1) * 100 if values[0] > 0 else 0

    mean_ret = float(np.mean(returns)) if len(returns) > 0 else 0.0
    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 1.0
    sharpe = (mean_ret / std_ret * np.sqrt(trading_days)) if std_ret > 0 else 0

    neg_returns = returns[returns < 0]
    downside_std = float(np.std(neg_returns, ddof=1)) if len(neg_returns) > 1 else std_ret
    sortino = (mean_ret / downside_std * np.sqrt(trading_days)) if downside_std > 0 else 0

    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) * 100

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Use trade_log (cost-aware) for per-trade stats when available,
    # otherwise fall back to the raw trades list.
    if trade_log:
        winning = [t for t in trade_log if t.pnl_pct > 0]
        losing = [t for t in trade_log if t.pnl_pct <= 0]
        all_count = len(trade_log)
        win_rate = len(winning) / all_count * 100 if all_count else 0
        gross_profit = sum(t.pnl_dollar for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl_dollar for t in losing)) if losing else 0
        avg_win = float(np.mean([t.pnl_pct for t in winning])) if winning else 0
        avg_loss = float(np.mean([t.pnl_pct for t in losing])) if losing else 0
        avg_holding = float(np.mean([t.holding_days for t in trade_log])) if trade_log else 0
        total_costs = sum(t.transaction_costs for t in trade_log)
    else:
        winning = [t for t in trades if t["pnl_pct"] > 0]
        losing = [t for t in trades if t["pnl_pct"] <= 0]
        all_count = len(trades)
        win_rate = len(winning) / all_count * 100 if all_count else 0
        gross_profit = sum(t["pnl_pct"] for t in winning) if winning else 0
        gross_loss = abs(sum(t["pnl_pct"] for t in losing)) if losing else 0
        avg_win = float(np.mean([t["pnl_pct"] for t in winning])) if winning else 0
        avg_loss = float(np.mean([t["pnl_pct"] for t in losing])) if losing else 0
        avg_holding = float(np.mean([t["holding_days"] for t in trades])) if trades else 0
        total_costs = 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    bench_return = 0.0
    if benchmark_equity and len(benchmark_equity) >= 2:
        bv = np.array([e["value"] for e in benchmark_equity])
        bench_return = (bv[-1] - bv[0]) / bv[0] * 100

    drawdown_curve: list = []
    for i, e in enumerate(equity_curve):
        drawdown_curve.append({
            "date": e["date"],
            "drawdown": round(float(drawdowns[i]) * 100, 2),
        })

    monthly_returns: Dict[str, Dict[str, float]] = {}
    for i in range(1, len(equity_curve)):
        date_str = equity_curve[i]["date"]
        month_key = date_str[:7]
        if month_key not in monthly_returns:
            monthly_returns[month_key] = {"start": equity_curve[i - 1]["value"], "end": equity_curve[i]["value"]}
        else:
            monthly_returns[month_key]["end"] = equity_curve[i]["value"]

    monthly_heatmap: list = []
    for month, vals in sorted(monthly_returns.items()):
        ret = (vals["end"] - vals["start"]) / vals["start"] * 100 if vals["start"] > 0 else 0
        parts = month.split("-")
        monthly_heatmap.append({
            "year": int(parts[0]),
            "month": int(parts[1]),
            "return_pct": round(float(ret), 2),
        })

    return {
        "total_return": round(float(total_return), 2),
        "cagr": round(float(cagr), 2),
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
        "max_drawdown": round(float(max_drawdown), 2),
        "calmar": round(float(calmar), 2),
        "win_rate": round(float(win_rate), 1),
        "profit_factor": round(float(min(profit_factor, 999)), 2),
        "total_trades": all_count,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "avg_win_pct": round(float(avg_win), 2),
        "avg_loss_pct": round(float(avg_loss), 2),
        "avg_holding_days": round(float(avg_holding), 1),
        "benchmark_return": round(float(bench_return), 2),
        "alpha": round(float(total_return - bench_return), 2),
        "total_transaction_costs": round(float(total_costs), 2),
        "drawdown_curve": drawdown_curve,
        "monthly_heatmap": monthly_heatmap,
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def run_walk_forward(
    ticker: str,
    strategy: str = "trend_following",
    params: Optional[dict] = None,
    train_days: int = 252,
    test_days: int = 63,
    total_days: int = 756,
    initial_capital: float = 10000,
    costs: Optional[TransactionCosts] = None,
    benchmark: str = "SPY",
) -> dict:
    """Run a walk-forward analysis.

    The data is split into consecutive (train, test) windows.  The
    strategy is run on each test window independently (training window
    is reserved for parameter calibration in future extensions).  The
    test-window results are stitched together to form the out-of-sample
    equity curve.

    Returns a dict with per-fold results, aggregate metrics and the
    combined equity curve.
    """
    if costs is None:
        costs = TransactionCosts()

    strategy_fn = STRATEGY_MAP.get(strategy)
    if not strategy_fn:
        return {"error": f"Unknown strategy: {strategy}. Available: {list(STRATEGY_MAP.keys())}"}

    strategy_params = {**STRATEGY_DEFAULTS.get(strategy, {}), **(params or {})}

    # Fetch enough data for indicators warm-up + all folds
    warmup = 250
    df = _fetch_history(ticker, days=total_days + warmup)
    if df is None:
        return {"error": f"Could not fetch price data for {ticker}"}
    df = _compute_indicators(df)

    # We need at least warmup + train_days + test_days
    if len(df) < warmup + train_days + test_days:
        return {"error": f"Insufficient data for walk-forward ({len(df)} bars available)"}

    # Start from after the warmup period
    start_idx = warmup
    folds: list = []
    combined_equity: list = []
    all_trades: list = []
    capital = initial_capital

    fold_num = 0
    while start_idx + train_days + test_days <= len(df):
        train_start = start_idx
        train_end = start_idx + train_days
        test_start = train_end
        test_end = min(train_end + test_days, len(df))

        test_slice = df.iloc[test_start:test_end].reset_index(drop=True)
        if len(test_slice) < 5:
            break

        fold_trades = strategy_fn(test_slice, strategy_params)
        fold_equity = _build_equity_curve(test_slice, fold_trades, capital, costs)

        # Scale fold equity to start from current capital level
        if fold_equity:
            scale = capital / fold_equity[0]["value"] if fold_equity[0]["value"] > 0 else 1.0
            for pt in fold_equity:
                pt["value"] = round(pt["value"] * scale, 2)
            capital = fold_equity[-1]["value"]

        fold_log = _build_trade_log(fold_trades, capital, costs)

        train_dates = (
            str(df.iloc[train_start]["date"])[:10],
            str(df.iloc[train_end - 1]["date"])[:10],
        )
        test_dates = (
            str(df.iloc[test_start]["date"])[:10],
            str(df.iloc[test_end - 1]["date"])[:10],
        )

        folds.append({
            "fold": fold_num,
            "train_period": {"start": train_dates[0], "end": train_dates[1]},
            "test_period": {"start": test_dates[0], "end": test_dates[1]},
            "trades": len(fold_trades),
            "return_pct": round(
                (fold_equity[-1]["value"] - fold_equity[0]["value"]) / fold_equity[0]["value"] * 100, 2
            ) if fold_equity and fold_equity[0]["value"] > 0 else 0.0,
        })

        # Strip entry_idx before appending
        for t in fold_trades:
            t.pop("entry_idx", None)
        all_trades.extend(fold_trades)
        combined_equity.extend(fold_equity)

        start_idx += test_days  # roll forward
        fold_num += 1

    if not folds:
        return {"error": "No complete walk-forward folds could be constructed"}

    # Benchmark
    bench_df = _fetch_history(benchmark, days=total_days + warmup)
    benchmark_equity: list = []
    if bench_df is not None:
        bench_slice = bench_df.tail(len(combined_equity)).reset_index(drop=True)
        benchmark_equity = _build_benchmark_equity(bench_slice, initial_capital)

    trade_log = _build_trade_log(all_trades, initial_capital, costs)
    metrics = _compute_metrics(combined_equity, all_trades, benchmark_equity, trade_log=trade_log)

    return {
        "ticker": ticker,
        "strategy": strategy,
        "strategy_params": strategy_params,
        "mode": "walk_forward",
        "train_days": train_days,
        "test_days": test_days,
        "total_folds": len(folds),
        "initial_capital": initial_capital,
        "final_value": combined_equity[-1]["value"] if combined_equity else initial_capital,
        "benchmark": benchmark,
        "transaction_costs": asdict(costs) if costs else None,
        "metrics": metrics,
        "folds": folds,
        "equity_curve": combined_equity,
        "benchmark_equity": benchmark_equity,
        "trades": all_trades,
        "trade_log": [asdict(r) for r in trade_log],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main entry point  (backward-compatible)
# ---------------------------------------------------------------------------

def run_backtest(
    ticker: str,
    strategy: str = "trend_following",
    params: Optional[dict] = None,
    period: str = "1y",
    initial_capital: float = 10000,
    benchmark: str = "SPY",
    # --- new optional parameters (old callers never pass these) ---
    costs: Optional[Dict[str, float]] = None,
    walk_forward: bool = False,
    train_days: int = 252,
    test_days: int = 63,
) -> dict:
    """Run a full backtest for *ticker* using the named *strategy*.

    This function preserves full backward compatibility: existing callers
    that pass only (ticker, strategy, params, period, initial_capital,
    benchmark) will get the same output shape as before, plus the new
    ``trade_log`` and ``transaction_costs`` keys.

    New parameters
    --------------
    costs : dict, optional
        Override default transaction-cost assumptions.  Accepted keys:
        ``commission_per_share``, ``spread_pct``, ``slippage_pct``.
        When *None* the :class:`TransactionCosts` defaults are used.
    walk_forward : bool
        When ``True``, run a walk-forward analysis instead of a single
        in-sample backtest.  ``train_days`` and ``test_days`` control
        the fold geometry.
    train_days / test_days : int
        Walk-forward fold sizes (only used when ``walk_forward=True``).
    """
    # Resolve strategy name via enum when possible (handles both
    # plain strings and Strategy enum members).
    strategy_key = strategy.value if isinstance(strategy, Strategy) else strategy

    # Build transaction-cost model
    tc = TransactionCosts(**(costs or {}))

    period_days = {
        "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "3y": 1095,
    }.get(period, 365)

    # ---- walk-forward mode ------------------------------------------------
    if walk_forward:
        return run_walk_forward(
            ticker=ticker,
            strategy=strategy_key,
            params=params,
            train_days=train_days,
            test_days=test_days,
            total_days=period_days,
            initial_capital=initial_capital,
            costs=tc,
            benchmark=benchmark,
        )

    # ---- single-pass back-test (original behaviour, now cost-aware) -------
    strategy_fn = STRATEGY_MAP.get(strategy_key)
    if not strategy_fn:
        return {"error": f"Unknown strategy: {strategy_key}. Available: {list(STRATEGY_MAP.keys())}"}

    strategy_params = {**STRATEGY_DEFAULTS.get(strategy_key, {}), **(params or {})}

    df = _fetch_history(ticker, days=period_days + 250)
    if df is None:
        return {"error": f"Could not fetch price data for {ticker}"}

    df = _compute_indicators(df)
    df = df.tail(period_days).reset_index(drop=True)

    if len(df) < 30:
        return {"error": f"Insufficient data for {ticker} ({len(df)} days)"}

    trades = strategy_fn(df, strategy_params)

    # Build cost-aware equity curve
    equity_curve = _build_equity_curve(df, trades, initial_capital, tc)

    # Benchmark
    bench_df = _fetch_history(benchmark, days=period_days + 250)
    benchmark_equity: list = []
    if bench_df is not None:
        bench_df = bench_df.tail(period_days).reset_index(drop=True)
        benchmark_equity = _build_benchmark_equity(bench_df, initial_capital)

    # Build detailed trade log
    trade_log = _build_trade_log(trades, initial_capital, tc)

    # Metrics now incorporate trade-log costs
    metrics = _compute_metrics(equity_curve, trades, benchmark_equity, trade_log=trade_log)

    # Strip internal index before serialisation (backward compat)
    for t in trades:
        t.pop("entry_idx", None)

    return {
        "ticker": ticker,
        "strategy": strategy_key,
        "strategy_params": strategy_params,
        "period": period,
        "initial_capital": initial_capital,
        "final_value": equity_curve[-1]["value"] if equity_curve else initial_capital,
        "benchmark": benchmark,
        "transaction_costs": asdict(tc),
        "metrics": metrics,
        "equity_curve": equity_curve,
        "benchmark_equity": benchmark_equity,
        "trades": trades,
        "trade_log": [asdict(r) for r in trade_log],
        "data_points": len(df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
