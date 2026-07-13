"""
Statistical Validation Module
------------------------------
Tools to justify — not assume — that a pair is cointegrated and mean-reverting
before trusting the strategy's z-score logic. Used by PairsTradingBacktester
but kept standalone and independently testable.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def run_adf_test(series: pd.Series, name: str = "series", significance: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    Null hypothesis: the series has a unit root (i.e. is NOT stationary).
    A low p-value (< significance) lets us reject the null -> series is stationary.
    """
    series = series.dropna()
    stat, pvalue, used_lag, n_obs, crit_values, _ = adfuller(series, autolag='AIC')
    return {
        'name': name,
        'adf_stat': stat,
        'p_value': pvalue,
        'used_lag': used_lag,
        'n_obs': n_obs,
        'critical_values': crit_values,
        'is_stationary': pvalue < significance,
    }


def run_engle_granger_test(y: pd.Series, x: pd.Series, significance: float = 0.05) -> dict:
    """
    Engle-Granger two-step cointegration test between y and x.
    Null hypothesis: NO cointegration.
    A low p-value (< significance) lets us reject the null -> series are cointegrated.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    stat, pvalue, crit_values = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return {
        'coint_stat': stat,
        'p_value': pvalue,
        'critical_values': crit_values,  # [1%, 5%, 10%]
        'is_cointegrated': pvalue < significance,
    }


def compute_static_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Full-sample OLS hedge ratio, used only for the validation report (not for trading)."""
    aligned = pd.concat([y, x], axis=1).dropna()
    x_const = sm.add_constant(aligned.iloc[:, 1])
    model = sm.OLS(aligned.iloc[:, 0], x_const).fit()
    return model.params.iloc[1]


def compute_half_life(spread: pd.Series) -> float:
    """
    Half-life of mean reversion via AR(1) regression on the spread:
        delta_spread_t = alpha + beta * spread_{t-1} + epsilon_t
        half_life = -ln(2) / beta

    Returns np.inf if beta >= 0 (no mean reversion detected -> half-life undefined).
    """
    spread = spread.dropna()
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lagged, delta = lagged.align(delta, join='inner')

    lagged_const = sm.add_constant(lagged)
    model = sm.OLS(delta, lagged_const).fit()
    beta = model.params.iloc[1]

    if beta >= 0:
        return np.inf
    return -np.log(2) / beta


def rolling_cointegration_pvalue(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """
    Rolling Engle-Granger p-value over time. Reveals periods where the
    cointegration relationship breaks down (p-value rises above 0.05),
    which a single full-sample test would hide.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    aligned.columns = ['y', 'x']
    pvalues = pd.Series(index=aligned.index, dtype=float)

    for i in range(window, len(aligned) + 1):
        chunk = aligned.iloc[i - window:i]
        try:
            _, pvalue, _ = coint(chunk['y'], chunk['x'])
        except Exception:
            pvalue = np.nan
        pvalues.iloc[i - 1] = pvalue

    return pvalues


def validation_report(y: pd.Series, x: pd.Series, window: int, significance: float = 0.05) -> dict:
    """Runs the full statistical validation suite and returns a structured report."""
    hedge_ratio = compute_static_hedge_ratio(y, x)
    spread = y - hedge_ratio * x

    eg = run_engle_granger_test(y, x, significance)
    adf_spread = run_adf_test(spread, name="spread", significance=significance)
    half_life = compute_half_life(spread)

    return {
        'static_hedge_ratio': hedge_ratio,
        'engle_granger': eg,
        'adf_spread': adf_spread,
        'half_life_days': half_life,
    }
