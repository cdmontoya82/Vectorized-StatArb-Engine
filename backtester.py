"""
Vectorized Statistical Arbitrage Engine (Institutional Grade)
-----------------------------------------------------------
This engine performs Pairs Trading backtesting with dynamic hedge ratios,
transaction friction handling, and Monte Carlo robustness validation.
Designed for research-grade financial engineering portfolios.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


class PairsTradingBacktester:

    # window_size is structural (defines rolling-window length) and must stay
    # an integer. It is deliberately excluded from Monte Carlo perturbation.
    _NON_PERTURBABLE_PARAMS = {'window_size'}

    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.params = self.config['trading_params']
        self.market = self.config['market_data']
        self.portfolio = self.config['portfolio']
        self.sim = self.config['simulation_params']
        self.df = pd.DataFrame()

    def fetch_data(self):
        """Downloads adjusted close data using vectorized yfinance calls."""
        assets = [self.portfolio['asset_y'], self.portfolio['asset_x']]
        print(f"📡 Downloading market data for: {assets}...")
        data = yf.download(assets, start=self.market['start_date'], end=self.market['end_date'])['Close']
        self.df = data.dropna()

    def run_backtest(self, custom_params=None, return_series=False):
        """
        Executes the strategy using vectorized pandas operations.

        return_series=False (default): returns the final cumulative return as a
            scalar — used by run_monte_carlo() for the distribution histogram.
        return_series=True: returns a dict with the full z_score and equity
            curve series — used by plot_results() for the dashboard.
        """
        p = custom_params if custom_params else self.params
        window = int(p['window_size'])  # guard against float window sizes from Monte Carlo perturbation

        y = self.df[self.portfolio['asset_y']]
        x = self.df[self.portfolio['asset_x']]

        # Calculate Rolling Hedge Ratio (Rolling OLS)
        hedge_ratio = y.rolling(window).cov(x) / x.rolling(window).var()
        spread = y - (hedge_ratio * x)

        # Calculate Z-Score
        z_score = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

        # --- Strategy Logic (Stateful, vectorized via ffill) ---
        # raw signal: -1 short spread, +1 long spread, 0 flat-on-exit, NaN = "hold previous position"
        # Built as a pd.Series indexed like z_score (NOT a bare numpy array) so that
        # later arithmetic against date-indexed Series aligns correctly instead of
        # silently producing all-NaN results.
        raw = pd.Series(np.nan, index=z_score.index)
        raw[z_score > p['z_entry']] = -1          # enter short spread
        raw[z_score < -p['z_entry']] = 1          # enter long spread
        raw[z_score.abs() < p['z_exit']] = 0      # mean-reversion exit (z_exit is now actually used)

        pos = raw.ffill().fillna(0)               # hold position between entry and exit signals
        pos[z_score.abs() > p['z_stop']] = 0       # hard stop-loss overrides everything

        # Performance Calculation
        ret_y = y.pct_change()
        ret_x = x.pct_change()

        # Daily PnL — pos is a properly date-indexed Series, so this aligns correctly
        strat_ret = pos.shift(1) * (ret_y - (hedge_ratio.shift(1) * ret_x))

        # Subtract Transaction Costs
        transaction_cost = self.market['transaction_cost']
        costs = pos.diff().abs() * transaction_cost

        final_ret = (strat_ret - costs).fillna(0)
        equity_curve = final_ret.cumsum() + 1  # +1 so the curve starts at 1.0 (base capital)

        if return_series:
            return {'z_score': z_score, 'equity_curve': equity_curve}

        return equity_curve.iloc[-1] - 1  # final cumulative return as a scalar

    def run_monte_carlo(self):
        """Validates strategy robustness by perturbing parameters."""
        print(f"🎲 Iniciando Monte Carlo ({self.sim['monte_carlo_runs']} iteraciones)...")
        results = []
        var = self.sim['param_variation']

        for _ in range(self.sim['monte_carlo_runs']):
            # Perturb parameters by +/- var, except window_size (kept fixed/integer)
            rand_params = {
                k: (v if k in self._NON_PERTURBABLE_PARAMS else v * np.random.uniform(1 - var, 1 + var))
                for k, v in self.params.items()
            }
            results.append(self.run_backtest(rand_params))

        plt.figure(figsize=(10, 5))
        plt.hist(results, bins=30, color='purple', alpha=0.7)
        plt.title("Monte Carlo Robustness Validation (Strategy Sensitivity)")
        plt.xlabel("Cumulative Return")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_results(self):
        """Plots the base strategy performance: Z-Score risk map + Equity curve."""
        series = self.run_backtest(return_series=True)
        z_score = series['z_score']
        equity = series['equity_curve']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # --- Upper chart: Statistical Risk Mapping ---
        ax1.plot(z_score.index, z_score, color='steelblue', linewidth=1, label='Z-Score')
        ax1.axhline(self.params['z_entry'], color='red', linestyle='--', linewidth=1, label='Entry (+/-)')
        ax1.axhline(-self.params['z_entry'], color='red', linestyle='--', linewidth=1)
        ax1.axhline(self.params['z_stop'], color='black', linestyle=':', linewidth=1, label='Stop (+/-)')
        ax1.axhline(-self.params['z_stop'], color='black', linestyle=':', linewidth=1)
        ax1.axhspan(-self.params['z_exit'], self.params['z_exit'], color='gray', alpha=0.2, label='Exit Zone')
        ax1.set_title("Statistical Risk Mapping")
        ax1.set_ylabel("Z-Score")
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # --- Lower chart: Equity Curve ---
        ax2.plot(equity.index, equity, color='purple', linewidth=1.5)
        ax2.set_title("Equity Curve")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Equity Growth (1.0 = Base)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('performance_chart.png', dpi=120)
        plt.show()

        final_ret = equity.iloc[-1] - 1
        print(f"✅ Backtest finalizado. Retorno estimado: {final_ret:.2%}")


if __name__ == "__main__":
    bt = PairsTradingBacktester()
    bt.fetch_data()
    bt.plot_results()
    bt.run_monte_carlo()
