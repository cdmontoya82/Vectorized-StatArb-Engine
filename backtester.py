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
import statsmodels.api as sm

class PairsTradingBacktester:
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
        data = yf.download(assets, start=self.market['start_date'], end=self.market['end_date'])['Close']
        self.df = data.dropna()

    def run_backtest(self, custom_params=None):
        """Executes the strategy using vectorized pandas operations."""
        p = custom_params if custom_params else self.params
        y = self.df[self.portfolio['asset_y']]
        x = self.df[self.portfolio['asset_x']]
        
        # Calculate Rolling Hedge Ratio (Rolling OLS)
        hedge_ratio = y.rolling(p['window_size']).cov(x) / x.rolling(p['window_size']).var()
        spread = y - (hedge_ratio * x)
        
        # Calculate Z-Score
        z_score = (spread - spread.rolling(p['window_size']).mean()) / spread.rolling(p['window_size']).std()
        
        # Strategy Logic (Vectorized)
        # 1 = Long Spread, -1 = Short Spread
        pos = np.where(z_score > p['z_entry'], -1, np.where(z_score < -p['z_entry'], 1, 0))
        # Hard Stop-Loss logic
        pos = np.where(np.abs(z_score) > p['z_stop'], 0, pos)
        
        # Performance Calculation
        ret_y = y.pct_change()
        ret_x = x.pct_change()
        strat_ret = pd.Series(pos).shift(1) * (ret_y - (hedge_ratio.shift(1) * ret_x))
        
        # Subtract Transaction Costs
        transaction_cost = self.market['transaction_cost']
        costs = np.abs(pd.Series(pos).diff()) * transaction_cost
        final_ret = strat_ret - costs
        
        return final_ret.cumsum().iloc[-1]

    def run_monte_carlo(self):
        """Validates strategy robustness by perturbing parameters."""
        print(f"🎲 Iniciando Monte Carlo ({self.sim['monte_carlo_runs']} iteraciones)...")
        results = []
        var = self.sim['param_variation']
        
        for _ in range(self.sim['monte_carlo_runs']):
            # Perturb parameters by +/- 10%
            rand_params = {k: v * np.random.uniform(1-var, 1+var) for k, v in self.params.items()}
            results.append(self.run_backtest(rand_params))
        
        plt.figure(figsize=(10, 5))
        plt.hist(results, bins=30, color='purple', alpha=0.7)
        plt.title("Monte Carlo Robustness Validation (Strategy Sensitivity)")
        plt.xlabel("Cumulative Return")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_results(self):
        """Plots the base strategy performance."""
        # Simple backtest plot to visualize equity
        ret = self.run_backtest()
        print(f"✅ Backtest finalizado. Retorno estimado: {ret:.2%}")

if __name__ == "__main__":
    bt = PairsTradingBacktester()
    bt.fetch_data()
    bt.run_monte_carlo()
