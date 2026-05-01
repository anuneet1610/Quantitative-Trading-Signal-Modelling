import pandas as pd
import numpy as np


class TechnicalMetricsCalculator:
    """Calculate technical trading metrics WITHOUT data leakage"""

    def __init__(self, df, price_col='close', timestamp_col='datetime', symbol_col='symbol'):
        self.df = df.copy()
        self.price_col = price_col
        self.timestamp_col = timestamp_col
        self.symbol_col = symbol_col

    def calculate_returns(self):
        """Calculate returns WITHOUT leakage - FIXED VERSION"""

        # Group by symbol to prevent mixing stocks
        self.df['prev_close'] = self.df.groupby(self.symbol_col)[self.price_col].shift(1)

        # Log returns for GARCH (historical: from t-1 to t)
        self.df['ret'] = np.log(self.df[self.price_col] / self.df['prev_close'])

        # TARGET: Predict next period's return (from t to t+1)
        # At time t, we know close[t] and want to predict close[t+1]
        self.df['next_close'] = self.df.groupby(self.symbol_col)[self.price_col].shift(-1)
        self.df['pct_change'] = ((self.df['next_close'] - self.df[self.price_col]) /
                                 self.df[self.price_col]) * 100

        return self

    def calculate_ema(self, period, col='close'):
        """Calculate EMA GROUPED by symbol - FIXED"""
        return self.df.groupby(self.symbol_col)[col].transform(
            lambda x: x.ewm(span=period, adjust=False).mean()
        )

    def calculate_all_emas(self):
        """Calculate all EMAs: 12, 20, 26, 50, 200"""
        self.df['ema_12'] = self.calculate_ema(12)
        self.df['ema_20'] = self.calculate_ema(20)
        self.df['ema_26'] = self.calculate_ema(26)
        self.df['ema_50'] = self.calculate_ema(50)
        self.df['ema_200'] = self.calculate_ema(200)
        return self

    def calculate_macd(self):
        """Calculate MACD GROUPED by symbol - FIXED"""
        # MACD line
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']

        # Signal line (9-period EMA of MACD) - GROUPED
        self.df['signal'] = self.df.groupby(self.symbol_col)['macd'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )

        # Histogram
        self.df['histogram'] = self.df['macd'] - self.df['signal']
        self.df['histogram_prev'] = self.df.groupby(self.symbol_col)['histogram'].shift(1)

        # Histogram growing/shrinking
        self.df['histogram_growing'] = self.df['histogram'] > self.df['histogram_prev']
        self.df['histogram_shrinking'] = self.df['histogram'] < self.df['histogram_prev']

        return self

    def calculate_macd_signal(self):
        """Calculate MACD signal classification (-2 to +2)"""
        conditions = [
            (self.df['histogram'] > 0) & self.df['histogram_growing'],
            (self.df['histogram'] > 0) & self.df['histogram_shrinking'],
            (self.df['histogram'] < 0) & self.df['histogram_shrinking'],
            (self.df['histogram'] < 0) & self.df['histogram_growing'],
        ]
        choices = [2, 1, -2, -1]
        self.df['macd_signal'] = np.select(conditions, choices, default=0)
        return self

    def calculate_rsi(self, period=14):
        """Calculate RSI GROUPED by symbol - FIXED"""

        def calc_rsi_group(price_series):
            delta = price_series.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        self.df['rsi'] = self.df.groupby(self.symbol_col)[self.price_col].transform(calc_rsi_group)
        return self

    def calculate_rsi_timing(self):
        """Calculate RSI timing signal"""

        def calc_timing_group(rsi_series):
            rsi_1 = rsi_series
            rsi_2 = rsi_series.shift(1)
            rsi_3 = rsi_series.shift(2)

            rising = (rsi_1 > rsi_2) & (rsi_2 > rsi_3)
            rsi_min_3 = rsi_series.rolling(window=3, min_periods=3).min()
            reversal_buy = rsi_min_3 < 40

            rsi_max = pd.concat([rsi_1, rsi_2, rsi_3], axis=1).max(axis=1)
            reversal_sell = (rsi_max > 70) & (rsi_3 < rsi_2) & (rsi_2 < rsi_1)

            conditions = [rising & reversal_buy, reversal_sell]
            choices = [2, -2]
            return pd.Series(np.select(conditions, choices, default=0), index=rsi_series.index)

        self.df['rsi_timing'] = self.df.groupby(self.symbol_col)['rsi'].transform(calc_timing_group)
        return self

    def calculate_garch_components(self, window=30, omega=0.007, alpha=0.501,
                                   beta=0.499, phi=-0.01, theta=-0.01, init_sigma=0.2):
        """Calculate GARCH(1,1) with ARMA(1,1) - processes each symbol separately"""

        def garch_window(returns):
            if isinstance(returns, pd.Series):
                returns = returns.dropna().values

            if len(returns) < 2:
                return {
                    'sigma_forecast': init_sigma,
                    'arma_forecast': 0.0,
                    'sigma_t': init_sigma,
                    'resid': 0.0
                }

            var = np.var(returns)
            last_sigma2 = var if var > 0 else init_sigma * init_sigma

            prev_ret = returns[0]
            last_resid = 0.0

            for i in range(1, len(returns)):
                cur_ret = returns[i]
                pred = phi * prev_ret + theta * last_resid
                resid = cur_ret - pred
                sigma2 = omega + alpha * (last_resid ** 2) + beta * last_sigma2
                last_sigma2 = sigma2
                last_resid = resid
                prev_ret = cur_ret

            sigma_forecast2 = omega + alpha * (last_resid ** 2) + beta * last_sigma2
            arma_forecast = phi * prev_ret + theta * last_resid

            return {
                'sigma_forecast': np.sqrt(sigma_forecast2),
                'arma_forecast': arma_forecast,
                'sigma_t': np.sqrt(last_sigma2),
                'resid': last_resid
            }

        # Process each symbol separately
        result_list = []

        for symbol in self.df[self.symbol_col].unique():
            symbol_mask = self.df[self.symbol_col] == symbol
            symbol_df = self.df[symbol_mask].copy()

            for idx in range(len(symbol_df)):
                start_idx = max(0, idx - window + 1)
                window_data = symbol_df['ret'].iloc[start_idx:idx + 1]

                if len(window_data) >= 2 and not window_data.isna().all():
                    result_list.append(garch_window(window_data))
                else:
                    result_list.append({
                        'sigma_forecast': init_sigma,
                        'arma_forecast': 0.0,
                        'sigma_t': init_sigma,
                        'resid': 0.0
                    })

        result_df = pd.DataFrame(result_list, index=self.df.index)
        self.df['sigma_forecast'] = result_df['sigma_forecast']
        self.df['arma_forecast'] = result_df['arma_forecast']
        self.df['sigma_t'] = result_df['sigma_t']
        self.df['resid'] = result_df['resid']

        return self

    def calculate_trend_filters(self):
        """Calculate trend filters"""
        self.df['ema_trend_filter_trend_up'] = self.df['ema_20'] > self.df['ema_50']
        self.df['ema_trend_filter_trend_down'] = self.df['ema_20'] < self.df['ema_50']
        self.df['long_term_bias_trend_up'] = self.df[self.price_col] > self.df['ema_200']
        self.df['long_term_bias_trend_down'] = self.df[self.price_col] < self.df['ema_200']
        return self

    def calculate_trading_signals(self):
        """Calculate trading signals"""
        self.df['risk_adj_ret'] = self.df['arma_forecast'] / self.df['sigma_forecast']
        self.df['long_signal'] = self.df['arma_forecast'] > 0
        self.df['short_signal'] = self.df['arma_forecast'] < 0
        return self

    def calculate_all_metrics(self):
        """Calculate all metrics in correct order"""
        print("Calculating returns...")
        self.calculate_returns()

        print("Calculating EMAs...")
        self.calculate_all_emas()

        print("Calculating MACD...")
        self.calculate_macd()
        self.calculate_macd_signal()

        print("Calculating RSI...")
        self.calculate_rsi()
        self.calculate_rsi_timing()

        print("Calculating GARCH...")
        self.calculate_garch_components()

        print("Calculating trend filters...")
        self.calculate_trend_filters()

        print("Calculating trading signals...")
        self.calculate_trading_signals()

        print("✓ All metrics calculated!")
        return self.df

    def get_final_output(self):
        """Get final output columns"""
        output_columns = [
            self.symbol_col,
            self.timestamp_col,
            'close',
            'sigma_forecast',
            'arma_forecast',
            'ema_trend_filter_trend_up',
            'ema_trend_filter_trend_down',
            'long_term_bias_trend_up',
            'long_term_bias_trend_down',
            'macd_signal',
            'risk_adj_ret',
            'long_signal',
            'short_signal',
            'rsi_timing',
            'pct_change'
        ]

        available_columns = [col for col in output_columns if col in self.df.columns]
        return self.df[available_columns]

    def test_for_leakage(self):
        """
        Test if any features can predict the target with suspiciously high accuracy
        This would indicate leakage
        """
        print("\n" + "=" * 80)
        print("LEAKAGE DETECTION TEST")
        print("=" * 80)

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # Drop rows with NaN
        test_df = self.df.dropna()

        if len(test_df) < 100:
            print("Not enough data for leakage test")
            return

        # Test each feature individually
        features_to_test = [
            'close', 'ema_12', 'ema_20', 'macd', 'histogram',
            'rsi', 'arma_forecast', 'sigma_forecast'
        ]

        print("\nTesting if individual features can predict target:")
        print("(High R² indicates potential leakage)")
        print()

        for feature in features_to_test:
            if feature not in test_df.columns:
                continue

            X = test_df[[feature]].values
            y = test_df['pct_change'].values

            # Simple linear regression
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)

            status = "🚨 LEAKAGE!" if r2 > 0.5 else "✅ Safe" if r2 < 0.1 else "⚠️ Check"
            print(f"{feature:20s} → R² = {r2:.4f}  {status}")

        print("\n" + "=" * 80)
        print("INTERPRETATION:")
        print("  R² < 0.1:  ✅ Safe - No obvious leakage")
        print("  R² 0.1-0.5: ⚠️ Check - Might be OK, investigate further")
        print("  R² > 0.5:  🚨 LEAKAGE - Feature contains future information!")
        print("=" * 80)


# USAGE
if __name__ == "__main__":
    df = pd.read_csv('combined_sorted_data.csv')

    # Ensure proper sorting
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

    # Process all symbols together (groupby handles separation)
    calculator = TechnicalMetricsCalculator(df, symbol_col='symbol', timestamp_col='datetime')
    result_df = calculator.calculate_all_metrics()
    calculator.test_for_leakage()

    # Get final output
    final_output = calculator.get_final_output()
    final_output.to_csv('calculated_metrics_5_min.csv', index=False)

    print(f"\n✓ Processed {len(df)} rows for {df['symbol'].nunique()} symbols")
    print(f"✓ Saved to 'calculated_metrics_fixed.csv'")
