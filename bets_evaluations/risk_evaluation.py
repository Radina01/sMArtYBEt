import numpy as np
import pandas as pd


def bootstrap_roi(bets, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    rois = []
    stakes = bets.get('stake', pd.Series(1.0, index=bets.index)).values
    for _ in range(n_boot):
        idx = rng.randint(0, len(bets), len(bets))
        sample = bets.iloc[idx]
        stakes_s = stakes[idx]
        profit = np.where(sample['correct'].values, stakes_s * (sample['odds'].values - 1), -stakes_s).sum()
        turnover = stakes_s.sum()
        roi = profit / turnover if turnover > 0 else 0.0
        rois.append(roi)
    arr = np.array(rois)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5), arr


def max_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    trough_idx = np.argmax(drawdowns)  # the trough of the maximum drawdown
    peak_idx = np.argmax(equity[:trough_idx + 1])  # the peak preceding the maximum drawdown
    max_dd = drawdowns.max()  # maximum absolute drawdown
    pct = max_dd / running_max[peak_idx] if running_max[peak_idx] > 0 else np.nan  # maximum drawdown as percentage
    return max_dd, peak_idx, trough_idx, pct


def sharpe_ratio(equity, risk_free_rate=0.0, periods_per_year=252):
    returns = np.diff(equity) / equity[:-1]
    if len(returns) < 2:
        return np.nan
    mean_r = np.mean(returns)  # average return per period
    std_r = np.std(returns, ddof=1)  # standard deviation of returns
    if std_r == 0:
        return np.nan
    return (mean_r - risk_free_rate / periods_per_year) / std_r * np.sqrt(periods_per_year)
