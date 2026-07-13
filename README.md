# 📉 Vectorized Statistical Arbitrage Engine

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)

An institutional-grade, highly optimized backtesting engine designed to execute **Pairs Trading** strategies based on financial asset cointegration. This engine is built to identify market inefficiencies, mitigate systemic risk via market neutrality, and execute with sub-millisecond precision.

---

## 🧠 Quant Architecture & Mathematics

Unlike directional trading models, this engine generates **Alpha** by exploiting the mean-reversion properties of cointegrated assets. 

### Mathematical Foundations
The model utilizes a **Rolling OLS (Ordinary Least Squares)** regression to determine the optimal hedge ratio ($\beta$), maintaining a delta-neutral portfolio:

$$Spread_t = Y_t - (\beta \cdot X_t)$$

To trigger trades, we calculate the **Z-Score** of the spread, ensuring our entries are based on statistical extremity relative to the rolling mean ($\mu$) and standard deviation ($\sigma$):

$$Z_t = \frac{Spread_t - \mu_t}{\sigma_t}$$

### Institutional Features:
*   **Dynamic Hedge Ratio:** Prevents Look-Ahead Bias by calculating rolling covariance and variance, ensuring the model only uses historically available data.
*   **Integrated Hard Stop-Loss:** A risk-topology safeguard. When $\vert{}Z_t\vert{} > 3.5$, the engine assumes a breakdown in cointegration and triggers an emergency liquidation to protect capital.
*   **Market Neutrality:** By holding offsetting positions in correlated assets (e.g., KO/PEP), the strategy remains insulated from broad market beta.
*   **Vectorized Engine:** Optimized for performance; zero `for` loops. The entire pipeline leverages Pandas matrix transformations for high-frequency backtesting capabilities.

---

## 📊 Strategy Performance Analysis

The engine generates a dual-plot dashboard designed to provide visual evidence of statistical stationarity.

![Statistical Risk Mapping](performance_chart.png)

### 1. Statistical Risk Mapping (Upper Chart)
*   **Z-Score (Blue Line):** Quantifies the deviation from the rolling mean.
*   **Entry Thresholds ($\pm 2.0\sigma$):** Automated signals for opening positions.
*   **Exit Zone (Gray Band):** The target state where the spread reverts to its historical fair value.
*   **Hard Stop-Loss ($\pm 3.5\sigma$):** Critical risk mitigation logic.

### 2. Equity Curve (Lower Chart)
*   **Drawdown Mitigation:** Illustrates how the model survives market shocks (e.g., March 2020) by capping exposure via the Stop-Loss logic.
*   **Real-World Friction:** Performance accounts for 10 Bps transaction costs, proving strategy viability in real broker conditions.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have the required libraries installed:
```bash
pip install -r requirements.txt
