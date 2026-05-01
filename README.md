# Quantitative-Trading-Signal-Modelling

This repository implements a hybrid econometric + machine learning framework for predicting short-horizon stock price direction using high-frequency data. Instead of focusing on accuracy, this project focuses on converting weak signals into profitable strategies.

The pipeline combines:
1) Econometric models (ARMA, GARCH) → capture linear trends & volatility
2) Technical indicators → capture momentum & market structure
3) XGBoost classifier → capture nonlinear relationships

The final model predicts: Whether the price will go UP or DOWN in the next 5-minute interval

## Files

### 1) Feature Engineering (calculate_metrics.py)
Computes:
1. Technical indicators (RSI, MACD, EMA, etc.)
2. Lagged returns
3. Rolling volatility
4. ARMA return forecasts
5. GARCH volatility forecasts
6. Time-based features (hour, day, seasonality)

### 2) Model Training (xgboost_model.py)
Model: XGBoost Classifier

Target: Probability of stock price going up

### 3) Backtesting (xgboost_up_down_backtesting.py)
Uses predicted probabilities to take trades only above a confidence threshold

Evaluates:
1. Sharpe Ratio
2. Total PnL
3. Drawdown
4. Trade coverage

## Data
**Source**: Bloomberg (as per research)
**Frequency**: 5-minute intervals
**Stocks**:
1. AAPL (Apple)
2. MSFT (Microsoft)
3. TSLA (Tesla)
4. JPM (JP Morgan Chase)
5. JNJ (Johnson & Johnson)

## Key Results

From the research paper:

**Accuracy**: 51.38%

**ROC-AUC**: ~0.53

**Precision**: 55.44%

**Recall**: 13.54%


With probability thresholding, the model produces a positive trading edge

1. Median Sharpe Ratio: 0.84
2. Stable drawdowns across walk-forward tests
3. Performance improves with longer holding horizons
