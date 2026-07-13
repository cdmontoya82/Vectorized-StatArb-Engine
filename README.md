# 📉 Vectorized Statistical Arbitrage Engine

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)

An institutional-grade, highly optimized backtesting engine designed to execute Pairs Trading strategies based on financial asset cointegration.

## 🧠 Quant Architecture & Mathematics

Unlike directional trading approaches, this engine extracts alpha from the relative relationship between two cointegrated assets (e.g., KO vs. PEP), insulating the portfolio from systemic market risk (**Market Neutrality**).

### Institutional Features:
*   **Dynamic Hedge Ratio (Rolling OLS):** The algorithm calculates moving variance and covariance to adjust the hedge weight daily. This strictly eliminates **Look-Ahead Bias**, ensuring the model only makes decisions based on historically available data at time T.
*   **Integrated Hard Stop-Loss:** Incorporates a strict risk topology. If the spread suffers an exogenous shock and the Z-Score breaches a critical threshold (e.g., Z > 3.5), the engine assumes a breakdown in cointegration and automatically liquidates the portfolio to prevent severe drawdowns.
*   **Real-World Friction:** Factors in transaction costs (slippage and commissions in basis points) every time a position is opened or closed, simulating actual broker conditions.
*   **Vectorized Engine:** Zero `for` loops. All signal evaluation and return calculations are executed via matrix transformations in Pandas, allowing the backtesting of decades of tick data in milliseconds.

## 🚀 Quick Start

The system is packaged in a highly modular OOP class. 

```bash
pip install -r requirements.txt
