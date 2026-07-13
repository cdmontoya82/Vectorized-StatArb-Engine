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

import stat_validation as sv
import performance_metrics as pm


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
        self.validation = self.config.get('validation_params', {})
        self.df = pd.DataFrame()
        self.benchmark_prices = None

    def fetch_data(self):
        """Downloads adjusted close data for the pair and, if configured, a benchmark ticker."""
        assets = [self.portfolio['asset_y'], self.portfolio['asset_x']]
        benchmark = self.market.get('benchmark_ticker')
        tickers = assets + ([benchmark] if benchmark else [])

        print(f"📡 Downloading market data for: {tickers}...")
        data = yf.download(tickers, start=self.market['start_date'], end=self.market['end_date'])['Close'].dropna()

        self.df = data[assets]
        self.benchmark_prices = data[benchmark] if benchmark else None

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
            return {'z_score': z_score, 'equity_curve': equity_curve, 'pos': pos, 'strat_ret': final_ret}

        return equity_curve.iloc[-1] - 1  # final cumulative return as a scalar

    def validate_pair(self, plot=True):
        """
        Statistical validation layer: confirms the pair is actually cointegrated
        and mean-reverting before trusting the z-score strategy on it. Runs:
          - Engle-Granger cointegration test (full sample)
          - ADF stationarity test on the spread
          - Half-life of mean reversion
          - Rolling cointegration p-value (stability over time)
        """
        y = self.df[self.portfolio['asset_y']]
        x = self.df[self.portfolio['asset_x']]
        window = self.validation.get('rolling_coint_window', 252)
        significance = self.validation.get('significance', 0.05)

        report = sv.validation_report(y, x, window=window, significance=significance)
        eg, adf = report['engle_granger'], report['adf_spread']

        print("=" * 60)
        print("🔬 STATISTICAL VALIDATION REPORT")
        print("=" * 60)
        print(f"Static hedge ratio (full sample OLS): {report['static_hedge_ratio']:.4f}")
        print(f"\nEngle-Granger cointegration test:")
        print(f"  p-value = {eg['p_value']:.4f}  ->  cointegrated: {eg['is_cointegrated']}")
        print(f"\nADF test on spread:")
        print(f"  p-value = {adf['p_value']:.4f}  ->  stationary: {adf['is_stationary']}")
        print(f"\nHalf-life of mean reversion: {report['half_life_days']:.1f} days")

        if not eg['is_cointegrated'] or not adf['is_stationary']:
            print("\n⚠️  WARNING: this pair does not show statistically significant")
            print("    cointegration/stationarity at the full-sample level. Trading it")
            print("    is a bet on correlation persisting, not a validated mean-reversion")
            print("    edge. Consider a different pair or a shorter, more stable subperiod.")
        print("=" * 60)

        if plot:
            rolling_p = sv.rolling_cointegration_pvalue(y, x, window=window)
            plt.figure(figsize=(12, 5))
            plt.plot(rolling_p.index, rolling_p, color='steelblue', linewidth=1, label='Rolling Engle-Granger p-value')
            plt.axhline(significance, color='red', linestyle='--', linewidth=1, label=f'Significance ({significance})')
            plt.fill_between(rolling_p.index, 0, significance, color='green', alpha=0.1)
            plt.title(f"Cointegration Stability Over Time (rolling {window}-day window)")
            plt.xlabel("Date")
            plt.ylabel("Engle-Granger p-value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('cointegration_stability.png', dpi=120)
            plt.show()

        return report

    @staticmethod
    def _sharpe_from_equity(equity_curve, periods_per_year=252):
        """Annualized Sharpe ratio (rf=0) from a cumulative equity curve."""
        daily_ret = equity_curve.pct_change().dropna()
        if daily_ret.std() == 0 or daily_ret.empty:
            return np.nan
        return (daily_ret.mean() / daily_ret.std()) * np.sqrt(periods_per_year)

    def run_walk_forward(self, train_fraction=None):
        """
        Out-of-sample validation: calibrates nothing extra (parameters stay as
        configured), but evaluates the strategy separately on a train segment
        and a held-out test segment. A strategy that performs well in-sample but
        collapses out-of-sample is a red flag for regime dependence / overfitting.
        """
        train_fraction = train_fraction if train_fraction is not None else self.validation.get('train_fraction', 0.7)
        split_idx = int(len(self.df) * train_fraction)
        split_date = self.df.index[split_idx]

        original_df = self.df
        try:
            self.df = original_df.iloc[:split_idx]
            train_series = self.run_backtest(return_series=True)
            train_ret = train_series['equity_curve'].iloc[-1] - 1
            train_sharpe = self._sharpe_from_equity(train_series['equity_curve'])

            self.df = original_df.iloc[split_idx:]
            test_series = self.run_backtest(return_series=True)
            test_ret = test_series['equity_curve'].iloc[-1] - 1
            test_sharpe = self._sharpe_from_equity(test_series['equity_curve'])
        finally:
            self.df = original_df  # always restore, even if something raises

        print("=" * 60)
        print(f"🧪 WALK-FORWARD VALIDATION (split at {split_date.date()})")
        print("=" * 60)
        print(f"{'Segment':<12}{'Period':<28}{'Return':>10}{'Sharpe':>10}")
        print(f"{'Train':<12}{str(original_df.index[0].date())+' -> '+str(original_df.index[split_idx-1].date()):<28}{train_ret:>10.2%}{train_sharpe:>10.2f}")
        print(f"{'Test':<12}{str(original_df.index[split_idx].date())+' -> '+str(original_df.index[-1].date()):<28}{test_ret:>10.2%}{test_sharpe:>10.2f}")

        if np.isfinite(train_sharpe) and np.isfinite(test_sharpe) and train_sharpe > 0 and test_sharpe < train_sharpe * 0.3:
            print("\n⚠️  WARNING: out-of-sample Sharpe collapses relative to train segment.")
            print("    This suggests the edge may be regime-specific rather than structural.")
        print("=" * 60)

        return {
            'split_date': split_date,
            'train': {'return': train_ret, 'sharpe': train_sharpe},
            'test': {'return': test_ret, 'sharpe': test_sharpe},
        }

    def run_monte_carlo(self):
        """
        Validates strategy robustness by perturbing parameters and tracing the
        full equity path (not just the final return) across all simulations.
        Produces a probabilistic equity-path chart with a 10%-90% confidence
        band and the median trajectory — matches monte_carlo_chart.png.
        """
        print(f"🎲 Iniciando Monte Carlo ({self.sim['monte_carlo_runs']} iteraciones)...")
        var = self.sim['param_variation']
        all_curves = []

        for _ in range(self.sim['monte_carlo_runs']):
            # Perturb parameters by +/- var, except window_size (kept fixed/integer)
            rand_params = {
                k: (v if k in self._NON_PERTURBABLE_PARAMS else v * np.random.uniform(1 - var, 1 + var))
                for k, v in self.params.items()
            }
            curve = self.run_backtest(rand_params, return_series=True)['equity_curve']
            all_curves.append(curve)

        # Aggregate trajectories
        df_curves = pd.concat(all_curves, axis=1)
        df_curves = df_curves.apply(pd.to_numeric, errors='coerce').fillna(1.0)

        x_axis = df_curves.index.to_numpy()
        median = df_curves.median(axis=1).to_numpy(dtype=float)
        p10 = df_curves.quantile(0.1, axis=1).to_numpy(dtype=float)
        p90 = df_curves.quantile(0.9, axis=1).to_numpy(dtype=float)

        plt.figure(figsize=(12, 6))
        plt.fill_between(x_axis, p10, p90, color='purple', alpha=0.2, label='Confidence Interval (10%-90%)')
        plt.plot(x_axis, median, color='purple', linewidth=2, label='Median Strategy Path')

        plt.title("Monte Carlo: Probabilistic Equity Path (Strategy Uncertainty Analysis)")
        plt.xlabel("Date")
        plt.ylabel("Equity Growth (1.0 = Base)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('monte_carlo_chart.png', dpi=120)
        plt.show()

    def generate_performance_report(self):
        """
        Computes institutional-style performance/risk metrics (Sharpe, Sortino,
        Calmar, VaR/CVaR, drawdown, beta vs benchmark, trade-level stats) and
        renders a 3-panel dashboard: Z-Score risk map, Equity curve, Drawdown.
        """
        series = self.run_backtest(return_series=True)
        z_score, equity, pos, strat_ret = series['z_score'], series['equity_curve'], series['pos'], series['strat_ret']

        benchmark_ret = None
        if self.benchmark_prices is not None:
            benchmark_ret = self.benchmark_prices.pct_change().reindex(equity.index)

        report = pm.compute_full_report(equity, pos, strat_ret, benchmark_returns=benchmark_ret)

        def pct(v):
            return f"{v:.2%}" if pd.notna(v) else "N/A"

        def num(v):
            return f"{v:.2f}" if pd.notna(v) and np.isfinite(v) else "N/A"

        print("=" * 60)
        print("📈 PERFORMANCE & RISK REPORT")
        print("=" * 60)
        print(f"Annualized Return:        {pct(report['annualized_return'])}")
        print(f"Sharpe Ratio:             {num(report['sharpe_ratio'])}")
        print(f"Sortino Ratio:            {num(report['sortino_ratio'])}")
        print(f"Calmar Ratio:             {num(report['calmar_ratio'])}")
        print(f"Max Drawdown:             {pct(report['max_drawdown'])}  "
              f"(duration: {report['drawdown_duration_days']} days)" if report['drawdown_duration_days'] is not None
              else f"Max Drawdown:             {pct(report['max_drawdown'])}  (not yet recovered)")
        print(f"VaR 95% (daily):          {pct(report['var_95'])}")
        print(f"CVaR 95% (daily):         {pct(report['cvar_95'])}")
        if report['beta_vs_benchmark'] is not None:
            bench_name = self.market.get('benchmark_ticker', 'benchmark')
            print(f"Beta vs {bench_name}:              {num(report['beta_vs_benchmark'])}")
        print(f"\nNumber of Trades:         {report['n_trades']}")
        print(f"Win Rate:                 {pct(report['win_rate'])}")
        print(f"Profit Factor:            {num(report['profit_factor'])}")
        print(f"Avg Win / Avg Loss:       {pct(report['avg_win'])} / {pct(report['avg_loss'])}")
        print(f"Avg Holding Period:       {num(report['avg_holding_days'])} days")
        print(f"Turnover (avg |Δpos|):    {report['turnover']:.3f}")
        print("=" * 60)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # --- Panel 1: Statistical Risk Mapping ---
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

        # --- Panel 2: Equity Curve ---
        ax2.plot(equity.index, equity, color='purple', linewidth=1.5)
        ax2.set_title("Equity Curve")
        ax2.set_ylabel("Equity Growth (1.0 = Base)")
        ax2.grid(True, alpha=0.3)

        # --- Panel 3: Drawdown ---
        dd_series = report['drawdown_series'] * 100
        ax3.fill_between(dd_series.index, dd_series, 0, color='firebrick', alpha=0.4)
        ax3.set_title("Drawdown")
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_xlabel("Date")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('performance_chart.png', dpi=120)
        plt.show()

        return report


if __name__ == "__main__":
    bt = PairsTradingBacktester()
    bt.fetch_data()
    bt.validate_pair()               # confirm cointegration/stationarity BEFORE trusting the strategy
    bt.run_walk_forward()            # out-of-sample sanity check
    bt.generate_performance_report() # institutional metrics + 3-panel dashboard
    bt.run_monte_carlo()
