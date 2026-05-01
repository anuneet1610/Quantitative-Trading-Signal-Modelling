import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

MODEL_PATH = "xgb_pct_change_model_test.json"

def create_time_features(df):
    """
    FIXED: Extract time features WITHOUT mixing stocks
    """
    df = df.copy()

    # Basic time components
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter
    df['year'] = df['datetime'].dt.year

    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Trading session
    # df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
    # df['is_premarket'] = ((df['hour'] >= 4) & (df['hour'] < 9)).astype(int)
    # df['is_aftermarket'] = ((df['hour'] >= 16) & (df['hour'] < 20)).astype(int)

    # CRITICAL FIX: Calculate timestamp WITHIN each symbol group
    # This prevents leakage across stocks
    # df['timestamp'] = df.groupby('symbol')['datetime'].transform(
    #     lambda x: x.astype(np.int64) / 10 ** 9
    # )

    return df

print("=" * 80)
print("XGBoost Testing - FIXED VERSION")
print("=" * 80)

print("\n1. Loading trained model...")
try:
    model_no_news = xgb.XGBClassifier()
    model_no_news.load_model("xgb_pct_change_model_no_news.json")
    print(f"   ✓ Model loaded from 'xgb_pct_change_model_no_news.json'")
except FileNotFoundError:
    print(f"   ✗ Error: Model file '{MODEL_PATH}' not found!")
    print("   Please train the model first using the training script.")
    exit(1)

# 1. Load data
print("\n1. Loading data from CSV...")
df = pd.read_csv('final_data_5_min.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# CRITICAL: Ensure data is sorted by symbol THEN datetime
df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
print(f"   Loaded {len(df):,} rows")
print(f"   Symbols: {df['symbol'].unique()}")
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# 2. CRITICAL DATA QUALITY CHECK
print("\n2. Data Quality Check...")

# Check for NaN in target
nan_target = df['pct_change'].isna().sum()
print(f"   NaN in pct_change: {nan_target:,} ({nan_target / len(df) * 100:.2f}%)")

# Check for Inf in target
inf_target = np.isinf(df['pct_change']).sum()
print(f"   Inf in pct_change: {inf_target:,}")

# CRITICAL: Remove last row of each stock (has NaN pct_change)
print("\n   Removing last row of each stock (NaN pct_change)...")
df = df.groupby('symbol').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
print(f"   Rows after cleanup: {len(df):,}")

# Verify no more NaN in target
nan_after = df['pct_change'].isna().sum()
if nan_after > 0:
    print(f"   ⚠ WARNING: Still have {nan_after} NaN values in pct_change")
    df = df.dropna(subset=['pct_change'])
    print(f"   Dropped NaN rows. Remaining: {len(df):,}")
else:
    print(f"   ✓ No NaN in pct_change")

# 3. Split by date
print("\n3. Splitting data by date...")
test_df = df[(df['datetime'] >= '2024-01-02') & (df['datetime'] <= '2024-12-30')].copy()

print(f"   Test: {len(test_df):,} rows ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

# 4. Create time features
print("\n4. Creating time features...")
test_df = create_time_features(test_df)
df = create_time_features(df)
print("   ✓ Time features created")

test_df['ret_1'] = test_df.groupby('symbol')['pct_change'].shift(1)
test_df['ret_2'] = test_df.groupby('symbol')['pct_change'].shift(2)
test_df['ret_3'] = test_df.groupby('symbol')['pct_change'].shift(3)
test_df['ret_6'] = test_df.groupby('symbol')['pct_change'].shift(6)

df['ret_1'] = df.groupby('symbol')['pct_change'].shift(1)
df['ret_2'] = df.groupby('symbol')['pct_change'].shift(2)
df['ret_3'] = df.groupby('symbol')['pct_change'].shift(3)
df['ret_6'] = df.groupby('symbol')['pct_change'].shift(6)

test_df['vol_10'] = test_df.groupby('symbol')['pct_change'].rolling(10).std().reset_index(0, drop=True)
df['vol_10'] = test_df.groupby('symbol')['pct_change'].rolling(10).std().reset_index(0, drop=True)

# Create directional target
df['up_down'] = (df['pct_change'] > 0).astype(int)
test_df['up_down'] = (test_df['pct_change'] > 0).astype(int)

# 5. Symbol mapping
print("\n5. Creating symbol mapping...")
try:
    with open("symbol_mapping.pkl", "rb") as f:
        symbol_mapping = pickle.load(f)
    print("   Loaded existing mapping")
except FileNotFoundError:
    unique_symbols = df['symbol'].unique()
    symbol_mapping = {i: sym for i, sym in enumerate(unique_symbols)}
    with open("symbol_mapping.pkl", "wb") as f:
        pickle.dump(symbol_mapping, f)
    print(f"   Created new mapping for {len(unique_symbols)} symbols")

reverse_mapping = {v: k for k, v in symbol_mapping.items()}

# 6. Map symbols to codes
print("\n6. Mapping symbols to codes...")
test_df['symbol'] = test_df['symbol'].map(reverse_mapping).astype(int)
df['symbol'] = df['symbol'].map(reverse_mapping).astype(int)

# Fill sentiment
if 'weighted_avg_sentiment' in test_df.columns:
    test_df['weighted_avg_sentiment'] = test_df['weighted_avg_sentiment'].fillna(0)
    df['weighted_avg_sentiment'] = df['weighted_avg_sentiment'].fillna(0)

print("Final dataset", len(test_df), "samples")

# Sort by symbol and datetime
test_df = test_df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
X_test_no_news = test_df.drop(columns=['datetime', 'pct_change', 'up_down','weighted_avg_sentiment'])
back_df = X_test_no_news.copy()
back_df['prob_up'] = model_no_news.predict_proba(X_test_no_news)[:, 1]
back_df['datetime'] = test_df['datetime']

# CRITICAL: Calculate FUTURE returns correctly
# At time t, we predict direction
# We enter at t+1 (next bar open/close)
# We exit at t+1+h (h bars later)

sharpe_list = []
total_pnl_list = []
max_drawdown_list = []
coverage_list = []
h_vals = []

# for i in range (3, 15):
h = 14

# Entry price: close of next bar
back_df['entry_price'] = back_df.groupby('symbol')['close'].shift(-1)

# Exit price: close h bars after entry
back_df['exit_price'] = back_df.groupby('symbol')['close'].shift(-1 - h)

# Calculate return
back_df['forward_return'] = (
        (back_df['exit_price'] - back_df['entry_price']) / back_df['entry_price']
)

# Trading decision: only trade if confidence >= threshold
t = 0.515
back_df['trade'] = (back_df['prob_up'] >= t).astype(int)
print("Threshold is: ", t)

costs = np.arange(0.0002, 0.00101, 0.0001)   # 0.0002, 0.0003, ... 0.0010
cost_labels = []
h = 14

for COST in costs:
    # Transaction costs

    back_df['gross_pnl'] = back_df['trade'] * back_df['forward_return']
    back_df['net_pnl'] = back_df['trade'] * (back_df['forward_return'] - COST)

    # Filter to actual trades only
    # Filter to actual trades only
    trades = back_df[
        (back_df['trade'] == 1) &
        back_df['net_pnl'].notna()
        ].copy()

    # Calculate metrics
    n_trades = len(trades)
    win_rate = (trades['net_pnl'] > 0).mean()
    avg_return = trades['net_pnl'].mean()

    # Cumulative PnL
    trades = trades.sort_values('datetime')
    trades['cum_pnl'] = trades['net_pnl'].cumsum()

    # Sharpe ratio (annualized)
    # For 5-minute bars: ~78 bars/day, 252 trading days
    periods_per_year = 252 * 78 / h
    sharpe = avg_return / trades['net_pnl'].std() * np.sqrt(periods_per_year)

    # Max drawdown
    trades['cum_max'] = trades['cum_pnl'].cummax()
    trades['drawdown'] = trades['cum_pnl'] - trades['cum_max']
    max_drawdown = trades['drawdown'].min()

    # Win/loss analysis
    winners = trades[trades['net_pnl'] > 0]
    losers = trades[trades['net_pnl'] <= 0]

    avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
    avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0

    # Profit factor
    total_wins = winners['net_pnl'].sum() if len(winners) > 0 else 0
    total_losses = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Calculate metrics by symbol
    by_symbol = trades.groupby('symbol')['net_pnl'].agg([
        ('trades', 'count'),
        ('avg_pnl', 'mean'),
        ('total_pnl', 'sum'),
        ('win_rate', lambda x: (x > 0).mean())
    ])

    # Calculate metrics by time of day
    trades['hour'] = pd.to_datetime(trades['datetime']).dt.hour
    by_hour = trades.groupby('hour')['net_pnl'].agg([
        ('trades', 'count'),
        ('avg_pnl', 'mean'),
        ('win_rate', lambda x: (x > 0).mean())
    ])

    metrics = {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_pnl': trades['net_pnl'].sum(),
        'coverage': n_trades / len(back_df)
    }

    sharpe_list.append(sharpe)
    total_pnl_list.append(trades['net_pnl'].sum())
    max_drawdown_list.append(max_drawdown)
    coverage_list.append(n_trades / len(back_df))
    h_vals.append(h)

    cost_labels.append(round(COST, 4))

    print("h: ",h, "metric:", metrics)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cost_labels, sharpe_list, marker='o', linewidth=2, color='steelblue', markersize=7)
ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Sharpe = 0')

for x, y in zip(cost_labels, sharpe_list):
    if not np.isnan(y):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=9)

ax.set_xlabel('Transaction Cost', fontsize=12)
ax.set_ylabel('Annualised Sharpe Ratio', fontsize=12)
ax.set_title(f'Transaction Cost vs Sharpe Ratio  (h={h}, threshold={t})', fontsize=13)
ax.set_xticks(cost_labels)
ax.set_xticklabels([f'{c:.4f}' for c in cost_labels], rotation=35, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cost_vs_sharpe.png', dpi=150)
plt.show()
print("Plot saved to cost_vs_sharpe.png")


# for i in range(3, 25):
#     h = i
#
#     # Entry price
#     back_df['entry_price'] = back_df.groupby('symbol')['close'].shift(-1)
#
#     # Exit price
#     back_df['exit_price'] = back_df.groupby('symbol')['close'].shift(-1 - h)
#
#     # Forward return
#     back_df['forward_return'] = (
#         (back_df['exit_price'] - back_df['entry_price']) / back_df['entry_price']
#     )
#
#     # Signal
#     t = 0.515
#     back_df['trade'] = (back_df['prob_up'] >= t).astype(int)
#     print("Threshold is:", t)
#
#     COST = 0.0002
#
#     back_df['net_pnl'] = back_df['trade'] * (back_df['forward_return'] - COST)
#
#     # ---------------------------------------------------------
#     # 🔥 NON-OVERLAPPING TRADE FILTER
#     # ---------------------------------------------------------
#
#     non_overlap_trades = []
#
#     for symbol, sym_df in back_df.groupby('symbol'):
#         sym_df = sym_df.sort_values('datetime').reset_index(drop=True)
#
#         in_position = False
#         exit_index = -1
#
#         selected_rows = []
#
#         for idx in range(len(sym_df)):
#
#             # If currently in position, skip until exit
#             if in_position and idx <= exit_index:
#                 continue
#
#             # If signal and valid return
#             if sym_df.loc[idx, 'trade'] == 1 and pd.notna(sym_df.loc[idx, 'net_pnl']):
#                 selected_rows.append(idx)
#                 in_position = True
#                 exit_index = idx + h  # hold for h bars
#
#         if selected_rows:
#             non_overlap_trades.append(sym_df.loc[selected_rows])
#
#     if len(non_overlap_trades) == 0:
#         print("No trades at h =", h)
#         continue
#
#     trades = pd.concat(non_overlap_trades).sort_values('datetime').copy()
#
#     # ---------------------------------------------------------
#     # Metrics
#     # ---------------------------------------------------------
#
#     n_trades = len(trades)
#     win_rate = (trades['net_pnl'] > 0).mean()
#     avg_return = trades['net_pnl'].mean()
#
#     trades['cum_pnl'] = trades['net_pnl'].cumsum()
#
#     periods_per_year = 252 * 78 / h
#     sharpe = avg_return / trades['net_pnl'].std(ddof=1) * np.sqrt(periods_per_year)
#
#     trades['cum_max'] = trades['cum_pnl'].cummax()
#     trades['drawdown'] = trades['cum_pnl'] - trades['cum_max']
#     max_drawdown = trades['drawdown'].min()
#
#     winners = trades[trades['net_pnl'] > 0]
#     losers = trades[trades['net_pnl'] <= 0]
#
#     avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
#     avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0
#
#     total_wins = winners['net_pnl'].sum() if len(winners) > 0 else 0
#     total_losses = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 1
#     profit_factor = total_wins / total_losses if total_losses > 0 else 0
#
#     metrics = {
#         'n_trades': n_trades,
#         'win_rate': win_rate,
#         'avg_return': avg_return,
#         'sharpe': sharpe,
#         'max_drawdown': max_drawdown,
#         'avg_win': avg_win,
#         'avg_loss': avg_loss,
#         'profit_factor': profit_factor,
#         'total_pnl': trades['net_pnl'].sum(),
#         'coverage': n_trades / len(back_df)
#     }
#
#     print("h:", h, "metric:", metrics)
#
#     # Simple t-test (overlap removed, so HAC not necessary)
#     returns = trades['net_pnl'].values
#     t_stat = returns.mean() / (returns.std(ddof=1) / np.sqrt(len(returns)))
#
#     print("Non-overlap t-stat:", t_stat)


import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(12, 8),
    sharex=True
)

# --- Sharpe Ratio ---
axes[0, 0].plot(h_vals, sharpe_list, marker='o')
axes[0, 0].set_title("Sharpe Ratio")
axes[0, 0].set_ylabel("Sharpe Ratio")
axes[0, 0].grid(True)

# --- Total PnL ---
axes[0, 1].plot(h_vals, total_pnl_list, marker='o')
axes[0, 1].set_title("Total PnL")
axes[0, 1].set_ylabel("Total PnL")
axes[0, 1].grid(True)

# --- Max Drawdown ---
axes[1, 0].plot(h_vals, max_drawdown_list, marker='o')
axes[1, 0].set_title("Max Drawdown")
axes[1, 0].set_ylabel("Max Drawdown")
axes[1, 0].set_xlabel("Holding Horizon (h)")
axes[1, 0].grid(True)

# --- Coverage ---
axes[1, 1].plot(h_vals, coverage_list, marker='o')
axes[1, 1].set_title("Coverage")
axes[1, 1].set_ylabel("Coverage")
axes[1, 1].set_xlabel("Holding Horizon (h)")
axes[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





def true_walk_forward_validation(df):
    """
    TRUE walk-forward validation with expanding training window
    (Q1-compliant)
    """

    def clean_xy(X, y):
        mask = (
            ~y.isna() &
            ~np.isinf(y) &
            ~X.isna().any(axis=1) &
            ~np.isinf(X.select_dtypes(include=[np.number])).any(axis=1)
        )
        return X[mask], y[mask]

    print(f"\n{'=' * 70}")
    print("TRUE WALK-FORWARD VALIDATION (EXPANDING WINDOW)")
    print(f"{'=' * 70}\n")

    results = []

    for month in range(1, 13):

        start = pd.Timestamp(2024, month, 1)
        end = start + pd.offsets.MonthBegin(1)

        train_df = df[df['datetime'] < start]
        test_df = df[(df['datetime'] >= start) & (df['datetime'] < end)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df.drop(columns=['datetime','pct_change','up_down','weighted_avg_sentiment'])
        y_train = train_df['up_down']
        X_test = test_df.drop(columns=['datetime', 'pct_change','up_down','weighted_avg_sentiment'])
        y_test = test_df['up_down']

        # 🔒 Critical cleaning
        X_train, y_train = clean_xy(X_train, y_train)
        X_test, y_test = clean_xy(X_test, y_test)

        if len(X_train) < 1000 or len(X_test) < 100:
            print(f"Month {month:2d}: Skipped (insufficient clean data)")
            continue

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=800,
            learning_rate=0.01,
            max_depth=8,
            reg_lambda=1.0,
            reg_alpha=0.3,
            eval_metric='logloss',
            use_label_encoder=False
        )

        # X_train_no_date = X_train.drop(columns=['datetime'])
        # X_test_no_date = X_test.drop(columns=['datetime'])
        model.fit(X_train, y_train)
        back_df = X_test.copy()
        # back_df['datetime'] = X_test['datetime'].values
        # back_df['symbol'] = X_test['symbol'].values

        back_df['prob_up'] = model.predict_proba(X_test)[:,1]


        # acc = accuracy_score(y_test, y_pred)
        # prec = precision_score(y_test, y_pred)
        # rec = recall_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        #
        # results.append({
        #     'month': month,
        #     'samples': len(y_test),
        #     'accuracy': acc,
        #     'precision': prec,
        #     'recall': rec,
        #     'f1': f1,
        # })

        # print(f"Month {month:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, N={len(y_test)}")

        h = 10

        # Entry price: close of next bar
        back_df['entry_price'] = back_df.groupby('symbol')['close'].shift(-1)

        # Exit price: close h bars after entry
        back_df['exit_price'] = back_df.groupby('symbol')['close'].shift(-1 - h)

        # Calculate return
        back_df['forward_return'] = (
                (back_df['exit_price'] - back_df['entry_price']) / back_df['entry_price']
        )

        # Trading decision: only trade if confidence >= threshold
        t = 0.515
        back_df['trade'] = (back_df['prob_up'] >= t).astype(int)
        print("Threshold is: ", t)

        # Transaction costs
        COST = 0.0002

        back_df['gross_pnl'] = back_df['trade'] * back_df['forward_return']
        back_df['net_pnl'] = back_df['trade'] * (back_df['forward_return'] - COST)

        # Filter to actual trades only
        # Filter to actual trades only
        trades = back_df[
            (back_df['trade'] == 1) &
            back_df['net_pnl'].notna()
            ].copy()

        # Calculate metrics
        n_trades = len(trades)
        win_rate = (trades['net_pnl'] > 0).mean()
        avg_return = trades['net_pnl'].mean()

        # Cumulative PnL
        trades['cum_pnl'] = trades['net_pnl'].cumsum()

        # Sharpe ratio (annualized)
        # For 5-minute bars: ~78 bars/day, 252 trading days
        periods_per_year = 252 * 78 / h
        sharpe = avg_return / trades['net_pnl'].std() * np.sqrt(periods_per_year)

        # Max drawdown
        trades['cum_max'] = trades['cum_pnl'].cummax()
        trades['drawdown'] = trades['cum_pnl'] - trades['cum_max']
        max_drawdown = trades['drawdown'].min()

        # Win/loss analysis
        winners = trades[trades['net_pnl'] > 0]
        losers = trades[trades['net_pnl'] <= 0]

        avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0

        # Profit factor
        total_wins = winners['net_pnl'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Calculate metrics by symbol
        by_symbol = trades.groupby('symbol')['net_pnl'].agg([
            ('trades', 'count'),
            ('avg_pnl', 'mean'),
            ('total_pnl', 'sum'),
            ('win_rate', lambda x: (x > 0).mean())
        ])

        # Calculate metrics by time of day
        # trades['hour'] = trades['datetime'].dt.hour
        #
        # by_hour = trades.groupby('hour')['net_pnl'].agg([
        #     ('trades', 'count'),
        #     ('avg_pnl', 'mean'),
        #     ('win_rate', lambda x: (x > 0).mean())
        # ])


        metrics = {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': trades['net_pnl'].sum(),
            'coverage': n_trades / len(back_df)
        }
        results.append(metrics)

    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(results_df.describe())

# true_walk_forward_validation(df)