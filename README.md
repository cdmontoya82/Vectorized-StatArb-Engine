📉 Vectorized Statistical Arbitrage Engine

An institutional-grade, highly optimized backtesting engine designed to execute Pairs Trading strategies based on financial asset cointegration.

This project was developed to model market inefficiencies through mean-reversion, implementing robust risk management safeguards that outperform traditional academic models by surviving real-world market friction and exogenous shocks.

🧠 Quant Architecture & Mathematics

Unlike directional trading approaches, this engine extracts alpha from the relative relationship between two cointegrated assets (e.g., KO vs. PEP), insulating the portfolio from systemic market risk (Market Neutrality).

Institutional Features:

Dynamic Hedge Ratio (Rolling OLS): The algorithm calculates moving variance and covariance to adjust the hedge weight daily. This strictly eliminates Look-Ahead Bias, ensuring the model only makes decisions based on historically available data at time T.

Integrated Hard Stop-Loss: Incorporates a strict risk topology. If the spread suffers an exogenous shock and the Z-Score breaches a critical threshold (e.g., Z > 3.5), the engine assumes a breakdown in cointegration and automatically liquidates the portfolio to prevent severe drawdowns (like the March 2020 crash).

Real-World Friction: Factors in transaction costs (slippage and commissions in basis points) every time a position is opened or closed, simulating actual broker conditions.

Vectorized Engine: Zero for loops. All signal evaluation and return calculations are executed via matrix transformations in Pandas, allowing the backtesting of decades of tick data in milliseconds.

🚀 Quick Start

The system is packaged in a highly modular OOP class. Run the backtester.py file to execute the simulation.

from backtester import PairsTradingBacktester

# Initialize the oracle with strict risk management
engine = PairsTradingBacktester(
    asset_y='KO', 
    asset_x='PEP', 
    start_date='2020-01-01', 
    end_date='2024-01-01',
    z_entry=2.0,  # Enter trade at 2 standard deviations
    z_exit=0.5,   # Take profit near the mean
    z_stop=3.5,   # Cut losses if market regime collapses
    transaction_cost=0.001 # 10 Bps transaction friction
)

engine.fetch_data()
engine.calculate_spread_and_zscore(window=30)
engine.generate_signals()
engine.calculate_returns()
engine.plot_results()


📊 Analytical Output

The engine generates a two-layer dashboard to evaluate strategy performance:

Statistical Risk Mapping: Visualizes the spread along with dynamic Entry bands (Green/Red), Exit bands (Gray), and the Hard Stop-Loss limit (Dark Red).

Equity Curve (PnL): The compound growth of the market-neutral portfolio, demonstrating mitigated drawdowns.

(Note: Add your generated performance chart here by dragging and dropping image_e56e68.png into the GitHub editor!)

🛠️ Tech Stack

Pandas & NumPy: Time-series matrix processing and vectorization.

Statsmodels: Econometric modeling and rolling variance calculation.

YFinance: Market telemetry data ingestion.

Matplotlib: Analytical rendering of financial metrics.

Disclaimer: This repository is for research and financial data engineering modeling purposes only. It does not constitute investment advice.
