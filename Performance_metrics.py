"""
Performance & Risk Metrics Module
----------------------------------
Institutional-style reporting: Sharpe/Sortino/Calmar, drawdown analysis,
historical VaR/CVaR, trade-level statistics, turnover, and beta vs a
benchmark. Kept standalone from PairsTradingBacktester so it can be tested
and reused independently.
"""

import numpy as np
import pandas as pd


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Drawdown at each point in time, relative to the running peak."""
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def max_drawdown(equity_curve: pd.Series) -> dict:
    """
    Worst peak-to-trough decline, plus the dates involved and how long it took
    to recover (None if the curve never recovers within the sample).
    """
    dd = compute_drawdown_series(equity_curve)
    trough_date = dd.idxmin()
    dd_value = dd.min()
    peak_date = equity_curve.loc[:trough_date].idxmax()
    peak_value = equity_curve.loc[peak_date]

    after_trough = equity_curve.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) else None
    duration_days = (recovery_date - peak_date).days if recovery_date is not None else None

    return {
        'max_drawdown': dd_value,
        'peak_date': peak_date,
        'trough_date': trough_date,
        'recovery_date': recovery_date,
        'duration_days': duration_days,
        'drawdown_series': dd,
    }


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = len(equity_curve) / periods_per_year
    if years <= 0:
        return np.nan
    return (1 + total_return) ** (1 / years) - 1


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = daily_returns - rf / periods_per_year
    if excess.empty or excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = daily_returns - rf / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std()
    if excess.empty or len(downside) == 0 or downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def calmar_ratio(ann_return: float, max_dd: float) -> float:
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    return ann_return / abs(max_dd)


def historical_var_cvar(daily_returns: pd.Series, confidence: float = 0.95) -> dict:
    """Historical (non-parametric) VaR and CVaR/Expected Shortfall on daily returns."""
    clean = daily_returns.dropna()
    if clean.empty:
        return {'VaR': np.nan, 'CVaR': np.nan, 'confidence': confidence}
    alpha = 1 - confidence
    var = -np.percentile(clean, alpha * 100)
    tail = clean[clean <= -var]
    cvar = -tail.mean() if len(tail) else var
    return {'VaR': var, 'CVaR': cvar, 'confidence': confidence}


def compute_beta(strat_returns: pd.Series, benchmark_returns: pd.Series):
    """OLS beta of strategy returns against a benchmark (e.g. SPY) — tests the
    'market neutrality' claim rather than assuming it."""
    aligned = pd.concat([strat_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    aligned.columns = ['strat', 'bench']
    bench_var = aligned['bench'].var()
    if bench_var == 0:
        return np.nan
    return aligned['strat'].cov(aligned['bench']) / bench_var


def extract_trades(pos: pd.Series, strat_ret: pd.Series) -> pd.DataFrame:
    """
    Reconstructs discrete trades from the position series: a trade opens when
    pos moves from 0 to nonzero, and closes when it returns to 0. Returns a
    DataFrame with one row per trade (entry/exit dates, direction, compounded
    return, holding period).
    """
    pos = pos.fillna(0)
    trades = []
    in_trade = False
    entry_idx = None
    direction = None

    for i in range(len(pos)):
        current = pos.iloc[i]
        if not in_trade and current != 0:
            in_trade = True
            entry_idx = i
            direction = current
        elif in_trade and current == 0:
            exit_idx = i
            # returns earned strictly while the position was open
            segment = strat_ret.iloc[entry_idx + 1: exit_idx + 1]
            trade_return = (1 + segment).prod() - 1
            trades.append({
                'entry_date': pos.index[entry_idx],
                'exit_date': pos.index[exit_idx],
                'direction': 'long_spread' if direction == 1 else 'short_spread',
                'holding_days': (pos.index[exit_idx] - pos.index[entry_idx]).days,
                'trade_return': trade_return,
            })
            in_trade = False

    return pd.DataFrame(trades, columns=['entry_date', 'exit_date', 'direction', 'holding_days', 'trade_return'])


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            'n_trades': 0, 'win_rate': np.nan, 'profit_factor': np.nan,
            'avg_win': np.nan, 'avg_loss': np.nan, 'avg_holding_days': np.nan,
        }

    wins = trades.loc[trades['trade_return'] > 0, 'trade_return']
    losses = trades.loc[trades['trade_return'] <= 0, 'trade_return']

    loss_sum = losses.sum()
    profit_factor = (wins.sum() / abs(loss_sum)) if loss_sum != 0 else np.inf

    return {
        'n_trades': len(trades),
        'win_rate': len(wins) / len(trades),
        'profit_factor': profit_factor,
        'avg_win': wins.mean() if len(wins) else np.nan,
        'avg_loss': losses.mean() if len(losses) else np.nan,
        'avg_holding_days': trades['holding_days'].mean(),
    }


def turnover(pos: pd.Series) -> float:
    """Average absolute position change per period — a proxy for trading activity/cost drag."""
    return pos.diff().abs().mean()


def compute_full_report(equity_curve: pd.Series, pos: pd.Series, strat_ret: pd.Series,
                         benchmark_returns: pd.Series = None, periods_per_year: int = 252,
                         rf: float = 0.0, var_confidence: float = 0.95) -> dict:
    """Runs every metric above and returns a single structured report dict."""
    daily_ret = equity_curve.pct_change().dropna()

    dd = max_drawdown(equity_curve)
    ann_ret = annualized_return(equity_curve, periods_per_year)
    sharpe = sharpe_ratio(daily_ret, rf, periods_per_year)
    sortino = sortino_ratio(daily_ret, rf, periods_per_year)
    calmar = calmar_ratio(ann_ret, dd['max_drawdown'])
    var_cvar = historical_var_cvar(daily_ret, var_confidence)

    trades = extract_trades(pos, strat_ret)
    tstats = trade_stats(trades)

    beta = compute_beta(daily_ret, benchmark_returns) if benchmark_returns is not None else None

    return {
        'annualized_return': ann_ret,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': dd['max_drawdown'],
        'drawdown_duration_days': dd['duration_days'],
        'var_95': var_cvar['VaR'],
        'cvar_95': var_cvar['CVaR'],
        'beta_vs_benchmark': beta,
        'turnover': turnover(pos),
        'drawdown_series': dd['drawdown_series'],
        'trades': trades,
        **tstats,
    }
