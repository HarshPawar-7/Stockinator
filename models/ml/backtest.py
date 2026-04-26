"""
Backtesting Framework

Tests valuation model accuracy against historical data.
Computes Sharpe ratio, hit rate, and max drawdown for signal-based strategies.

Reference: stock_valuation_ml_reference.md §5.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int

    # Accuracy
    hit_rate: float          # % of correct BUY/SELL signals
    buy_hit_rate: float      # % of BUY signals that went up
    sell_hit_rate: float     # % of SELL signals that went down

    # Returns (annualized)
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float

    # Model accuracy
    avg_mos_accuracy: float  # How close MOS was to actual price movement

    def to_dict(self) -> dict:
        return {
            "total_signals": self.total_signals,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "hold_signals": self.hold_signals,
            "hit_rate": round(self.hit_rate, 4),
            "buy_hit_rate": round(self.buy_hit_rate, 4),
            "sell_hit_rate": round(self.sell_hit_rate, 4),
            "avg_return": round(self.avg_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_mos_accuracy": round(self.avg_mos_accuracy, 4),
        }


def run_backtest(
    valuations: pd.DataFrame,
    future_returns: pd.Series,
    holding_period_months: int = 12,
) -> BacktestResult:
    """
    Backtest valuation signals against actual future returns.

    Args:
        valuations: DataFrame with columns: ticker, date, signal, margin_of_safety, ensemble_value, price
        future_returns: Series of actual returns over holding period (indexed by ticker+date)
        holding_period_months: Period for return measurement

    Returns:
        BacktestResult with performance metrics
    """
    if len(valuations) == 0:
        raise ValueError("No valuation data for backtesting")

    # Merge valuations with future returns
    df = valuations.copy()
    df["actual_return"] = future_returns

    # Drop rows without future returns
    df = df.dropna(subset=["actual_return"])

    if len(df) == 0:
        raise ValueError("No matching future return data")

    # Signal counts
    total = len(df)
    buy_mask = df["signal"].isin(["BUY", "STRONG_BUY"])
    sell_mask = df["signal"].isin(["SELL", "STRONG_SELL"])
    hold_mask = df["signal"] == "HOLD"

    buy_count = buy_mask.sum()
    sell_count = sell_mask.sum()
    hold_count = hold_mask.sum()

    # Hit rates
    buy_hits = (df.loc[buy_mask, "actual_return"] > 0).sum() if buy_count > 0 else 0
    sell_hits = (df.loc[sell_mask, "actual_return"] < 0).sum() if sell_count > 0 else 0

    buy_hit_rate = buy_hits / buy_count if buy_count > 0 else 0
    sell_hit_rate = sell_hits / sell_count if sell_count > 0 else 0

    total_hits = buy_hits + sell_hits
    total_directional = buy_count + sell_count
    hit_rate = total_hits / total_directional if total_directional > 0 else 0

    # Strategy returns (long BUYs, short SELLs, skip HOLDs)
    strategy_returns = []
    if buy_count > 0:
        strategy_returns.extend(df.loc[buy_mask, "actual_return"].tolist())
    if sell_count > 0:
        strategy_returns.extend((-df.loc[sell_mask, "actual_return"]).tolist())

    if strategy_returns:
        returns_arr = np.array(strategy_returns)
        avg_return = float(np.mean(returns_arr))
        std_return = float(np.std(returns_arr))
        # Annualize
        annualization_factor = 12 / holding_period_months
        sharpe = (avg_return * annualization_factor) / (std_return * np.sqrt(annualization_factor)) if std_return > 0 else 0

        # Max drawdown (cumulative)
        cumulative = np.cumsum(returns_arr)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0
    else:
        avg_return = 0.0
        sharpe = 0.0
        max_dd = 0.0

    # MOS accuracy
    mos_accuracy = 0.0
    if "margin_of_safety" in df.columns:
        mos = df["margin_of_safety"].dropna()
        actual = df.loc[mos.index, "actual_return"]
        if len(mos) > 0:
            # How well does MOS predict actual returns?
            mos_accuracy = float(np.corrcoef(mos, actual)[0, 1]) if len(mos) >= 2 else 0

    result = BacktestResult(
        total_signals=total,
        buy_signals=buy_count,
        sell_signals=sell_count,
        hold_signals=hold_count,
        hit_rate=hit_rate,
        buy_hit_rate=buy_hit_rate,
        sell_hit_rate=sell_hit_rate,
        avg_return=avg_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        avg_mos_accuracy=mos_accuracy,
    )

    logger.info(
        "Backtest: %d signals | Hit rate: %.1f%% | Sharpe: %.2f | Max DD: %.2f%%",
        total, hit_rate * 100, sharpe, max_dd * 100,
    )

    return result
