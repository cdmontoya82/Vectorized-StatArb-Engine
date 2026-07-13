# ==============================================================================
# STATISTICAL ARBITRAGE PAIRS TRADING ENGINE (INSTITUTIONAL GRADE)
# ==============================================================================
# This algorithm implements a mean-reversion strategy (Pairs Trading)
# with integrated risk management (Hard Stop-Loss) and dynamic Hedge Ratio
# calculation to strictly eliminate Look-Ahead Bias.
# ==============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class PairsTradingBacktester:
    """
    Vectorized Backtesting Engine for Pairs Arbitrage.
    """
    def __init__(self, asset_y, asset_x, start_date, end_date, z_entry=2.0, z_exit=0.5, z_stop=3.5, transaction_cost=0.001):
        self.asset_y = asset_y
        self.asset_x = asset_x
        self.start_date = start_date
        self.end_date = end_date
        
        # Statistical Model Parameters
        self.z_entry = z_entry  # Threshold to open position
        self.z_exit = z_exit    # Threshold to take profit
        self.z_stop = z_stop    # Emergency brake (Hard Stop-Loss)
        self.transaction_cost = transaction_cost # Slippage and Commissions (0.1%)
        
        self.df = pd.DataFrame()

    def fetch_data(self):
        """Downloads market telemetry and aligns time series."""
        print(f"📡 Fetching market telemetry for {self.asset_y} and {self.asset_x}...")
        data = yf.download([self.asset_y, self.asset_x], start=self.start_date, end=self.end_date)
        
        self.df = data['Close'].copy()
        self.df = self.df[[self.asset_y, self.asset_x]].dropna()

    def calculate_spread_and_zscore(self, window=30):
        """
        Calculates dynamic Hedge Ratio and Z-Score (Rolling OLS).
        This ensures the model only uses historical data at any given time step T.
        """
        print("🧠 Calculating Dynamic Hedge Ratio and Z-Score...")
        
        # 1. Dynamic Hedge Ratio (Rolling Beta = Covariance / Variance)
        rolling_cov = self.df[self.asset_y].rolling(window=window).cov(self.df[self.asset_x])
        rolling_var = self.df[self.asset_x].rolling(window=window).var()
        self.df['Hedge_Ratio'] = rolling_cov / rolling_var
        
        # 2. Spread Calculation (Market Inefficiency)
        self.df['Spread'] = self.df[self.asset_y] - (self.df['Hedge_Ratio'] * self.df[self.asset_x])
        
        # 3. Rolling Z-Score
        spread_mean = self.df['Spread'].rolling(window=window).mean()
        spread_std = self.df['Spread'].rolling(window=window).std()
        self.df['Z-Score'] = (self.df['Spread'] - spread_mean) / spread_std
        
        self.df = self.df.dropna()

    def generate_signals(self):
        """Generates trading signals with strict risk management (Stop-Loss)."""
        print("⚡ Evaluating topological and risk rules...")
        self.df['Position'] = 0 

        # Entry Rules
        self.df.loc[self.df['Z-Score'] > self.z_entry, 'Position'] = -1  # Short Spread
        self.df.loc[self.df['Z-Score'] < -self.z_entry, 'Position'] = 1   # Long Spread
        
        # Normal Exit Rules (Take Profit)
        self.df.loc[(self.df['Z-Score'] < self.z_exit) & (self.df['Z-Score'] > -self.z_exit), 'Position'] = 0

        # RISK MANAGEMENT: Hard Stop-Loss (If cointegration breaks violently)
        self.df.loc[self.df['Z-Score'].abs() > self.z_stop, 'Position'] = 0

        # Forward fill the open positions
        self.df['Position'] = self.df['Position'].replace(0, np.nan).ffill().fillna(0)

        # Force exits again after forward-fill to avoid holding in target zones
        self.df.loc[(self.df['Z-Score'] < self.z_exit) & (self.df['Z-Score'] > -self.z_exit), 'Position'] = 0
        self.df.loc[self.df['Z-Score'].abs() > self.z_stop, 'Position'] = 0

    def calculate_returns(self):
        """Simulates portfolio growth applying the dynamic Hedge Ratio and Transaction Costs."""
        print("💰 Calculating PnL (Profit and Loss) with Transaction Friction...")
        self.df['Ret_Y'] = self.df[self.asset_y].pct_change()
        self.df['Ret_X'] = self.df[self.asset_x].pct_change()

        # Shift(1) on position and Hedge Ratio is crucial to avoid Look-Ahead Bias.
        self.df['Strategy_Return'] = self.df['Position'].shift(1) * (
            self.df['Ret_Y'] - (self.df['Hedge_Ratio'].shift(1) * self.df['Ret_X'])
        )
        
        # REAL-WORLD FRICTION: Detect position changes and apply transaction costs
        # The .diff().abs() method isolates exactly when trades are opened or closed
        trade_signals = self.df['Position'].diff().abs()
        self.df['Strategy_Return'] = self.df['Strategy_Return'] - (trade_signals * self.transaction_cost).fillna(0)
        
        self.df['Cumulative_Return'] = (1 + self.df['Strategy_Return'].fillna(0)).cumprod()

    def plot_results(self):
        """Generates the analytical dashboard."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.5, 1]})
        
        # Chart 1: Z-Score and Risk Mapping
        axes[0].plot(self.df.index, self.df['Z-Score'], label='Z-Score', color='#1f77b4', linewidth=1.2)
        axes[0].axhline(self.z_stop, color='darkred', linestyle='-', label=f'Stop-Loss (+{self.z_stop})')
        axes[0].axhline(-self.z_stop, color='darkred', linestyle='-', label=f'Stop-Loss (-{self.z_stop})')
        axes[0].axhline(self.z_entry, color='red', linestyle='--', label=f'Short Entry (+{self.z_entry})')
        axes[0].axhline(-self.z_entry, color='green', linestyle='--', label=f'Long Entry (-{self.z_entry})')
        axes[0].axhline(0, color='black', alpha=0.3)
        axes[0].fill_between(self.df.index, self.z_exit, -self.z_exit, color='gray', alpha=0.15, label='Take Profit Zone')
        axes[0].set_title(f'Statistical Risk Mapping: {self.asset_y} vs {self.asset_x}', fontsize=14)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Chart 2: Equity Curve
        axes[1].plot(self.df.index, self.df['Cumulative_Return'], label='Strategy Equity', color='purple', linewidth=2)
        axes[1].set_title('Equity Curve (Drawdowns mitigated by Stop-Loss)', fontsize=14)
        axes[1].set_ylabel('Compound Growth (1.0 = Base)')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        total_return = (self.df['Cumulative_Return'].iloc[-1] - 1) * 100
        print("\n" + "=" * 50)
        print(f"🚀 NET ALGORITHM RETURN: {total_return:.2f}%")
        print("=" * 50 + "\n")

# ==========================================
# ENTRY POINT (MAIN)
# ==========================================
if __name__ == "__main__":
    # Adjusted Backtest: Integrated Stop-Loss at Z=3.5 to cut losses during systemic crashes.
    # Added 10 Bps (0.1%) of transaction friction to simulate real institutional environments.
    backtest = PairsTradingBacktester(
        asset_y='KO', 
        asset_x='PEP', 
        start_date='2020-01-01', 
        end_date='2024-01-01', 
        z_entry=2.0, 
        z_exit=0.5, 
        z_stop=3.5,  # <--- STRICT RISK MANAGEMENT INJECTED
        transaction_cost=0.001 # <--- REAL WORLD FRICTION (Commissions & Slippage)
    )
    
    backtest.fetch_data()
    backtest.calculate_spread_and_zscore(window=30)
    backtest.generate_signals()
    backtest.calculate_returns()
    backtest.plot_results()
