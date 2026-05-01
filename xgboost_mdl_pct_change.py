import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    # df['is_market_open'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    # df['is_premarket'] = ((df['hour'] >= 8) & (df['hour'] < 13)).astype(int)
    # df['is_aftermarket'] = ((df['hour'] >= 21) & (df['hour'] < 24)).astype(int)

    # CRITICAL FIX: Calculate timestamp WITHIN each symbol group
    # This prevents leakage across stocks
    # df['timestamp'] = df.groupby('symbol')['datetime'].transform(
    #     lambda x: x.astype(np.int64) / 10 ** 9
    # )

    return df

print("=" * 80)
print("XGBoost Training - FIXED VERSION")
print("=" * 80)

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
train_df = df[(df['datetime'] >= '2021-01-04') & (df['datetime'] <= '2023-12-29')].copy()
test_df = df[(df['datetime'] >= '2024-01-02') & (df['datetime'] <= '2024-12-30')].copy()

print(f"   Training: {len(train_df):,} rows ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
print(f"   Test: {len(test_df):,} rows ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

# 4. Create time features
print("\n4. Creating time features...")
train_df = create_time_features(train_df)
test_df = create_time_features(test_df)
print("   ✓ Time features created")

train_df['ret_1'] = train_df.groupby('symbol')['pct_change'].shift(1)
train_df['ret_2'] = train_df.groupby('symbol')['pct_change'].shift(2)
train_df['ret_3'] = train_df.groupby('symbol')['pct_change'].shift(3)
train_df['ret_6'] = train_df.groupby('symbol')['pct_change'].shift(6)

test_df['ret_1'] = test_df.groupby('symbol')['pct_change'].shift(1)
test_df['ret_2'] = test_df.groupby('symbol')['pct_change'].shift(2)
test_df['ret_3'] = test_df.groupby('symbol')['pct_change'].shift(3)
test_df['ret_6'] = test_df.groupby('symbol')['pct_change'].shift(6)

train_df['vol_10'] = train_df.groupby('symbol')['pct_change'].rolling(10).std().reset_index(0, drop=True)
test_df['vol_10'] = test_df.groupby('symbol')['pct_change'].rolling(10).std().reset_index(0, drop=True)

# Create directional target
train_df['up_down'] = (train_df['pct_change'] > 0).astype(int)
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
train_df['symbol'] = train_df['symbol'].map(reverse_mapping).astype(int)
test_df['symbol'] = test_df['symbol'].map(reverse_mapping).astype(int)

# Fill sentiment
if 'weighted_avg_sentiment' in train_df.columns:
    train_df['weighted_avg_sentiment'] = train_df['weighted_avg_sentiment'].fillna(0)
    test_df['weighted_avg_sentiment'] = test_df['weighted_avg_sentiment'].fillna(0)

# 7. Prepare features
print("\n7. Preparing features and target...")

# Drop columns
X_train_full = train_df.drop(columns=['pct_change','up_down','weighted_avg_sentiment'])
y_train_full = train_df['up_down']

# CRITICAL: Use TIME-BASED split for validation (not random!)
# Last 10% of training data chronologically
# Find the 90th percentile date
split_date = X_train_full['datetime'].quantile(0.9)

# Split by date
X_train = X_train_full[X_train_full['datetime'] < split_date]
y_train = y_train_full[X_train_full['datetime'] < split_date]

X_val = X_train_full[X_train_full['datetime'] >= split_date]
y_val = y_train_full[X_train_full['datetime'] >= split_date]

X_train = X_train.drop(columns=['datetime'])
X_val = X_val.drop(columns=['datetime'])

print(f"   Training: {len(X_train):,} samples")
print(f"   Validation: {len(X_val):,} samples")

# 8. Final data cleaning
print("\n8. Final data cleaning...")
print(f"   Before: Train={len(X_train):,}, Val={len(X_val):,}")

# Check for any remaining issues
print(f"   NaN in y_train: {y_train.isna().sum()}")
print(f"   Inf in y_train: {np.isinf(y_train).sum()}")
print(f"   NaN in X_train: {X_train.isna().sum().sum()}")
print(f"   Inf in X_train: {np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()}")

# Remove any remaining bad values
valid_train = ~(y_train.isna() | np.isinf(y_train))
X_train = X_train[valid_train]
y_train = y_train[valid_train]

valid_val = ~(y_val.isna() | np.isinf(y_val))
X_val = X_val[valid_val]
y_val = y_val[valid_val]

# Remove rows with NaN/Inf in features
train_clean = ~(X_train.isna().any(axis=1) |
                np.isinf(X_train.select_dtypes(include=[np.number])).any(axis=1))
X_train = X_train[train_clean]
y_train = y_train[train_clean]

val_clean = ~(X_val.isna().any(axis=1) |
              np.isinf(X_val.select_dtypes(include=[np.number])).any(axis=1))
X_val = X_val[val_clean]
y_val = y_val[val_clean]

print(y_train.value_counts(normalize=True))

print(f"   After: Train={len(X_train):,}, Val={len(X_val):,}")

# 9. DIAGNOSTIC: Check target distribution
print("\n9. Target Variable Distribution:")
print(f"   Training mean: {y_train.mean():.4f}%")
print(f"   Training std: {y_train.std():.4f}%")
print(f"   Training min: {y_train.min():.4f}%")
print(f"   Training max: {y_train.max():.4f}%")

print(X_train.columns)

# 10. Train model
print("\n10. Training XGBoost...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=800,
    learning_rate=0.01,
    max_depth=8,
    reg_lambda=1.0,
    reg_alpha=0.3,
    eval_metric='logloss',
    use_label_encoder=False,
    early_stopping_rounds=50
)


# model.fit(
#     X_train, y_train,
#     eval_set=[(X_train, y_train), (X_val, y_val)],
#     verbose=False
# )
#
# results = model.evals_result()
# # Plot the learning curve
# # Updated Plotting Code
# plt.figure(figsize=(10, 7))
#
# # Use 'validation_0' for the first dataset in eval_set (X_train)
# plt.plot(results['validation_0']['logloss'], label='Training loss')
#
# # Use 'validation_1' for the second dataset in eval_set (X_val)
# plt.plot(results['validation_1']['logloss'], label='Validation loss')
#
# plt.xlabel('Number of trees (Epochs)')
# plt.ylabel('Log Loss')
# plt.title('XGBoost Training and Validation Curve')
# plt.legend()
# plt.show()
#
#
#
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix
# )
#
# # # Probabilities & predictions
# # val_probs = model.predict_proba(X_val)[:, 1]
# # thresholds = np.linspace(0.45, 0.65, 41)
# #
# # best = []
# # for t in thresholds:
# #     preds = (val_probs > t).astype(int)
# #     acc = accuracy_score(y_val, preds)
# #     prec = precision_score(y_val, preds)
# #     rec = recall_score(y_val, preds)
# #     f1 = f1_score(y_val, preds)
# #     best.append((t, acc, prec, rec, f1))
# #
# # best_df = pd.DataFrame(
# #     best, columns=['threshold', 'accuracy', 'precision', 'recall', 'f1']
# # )
# #
# # print(best_df.sort_values('f1', ascending=False).head(10))
#
# model.save_model("xgb_pct_change_model_no_news.json")
# print("\n✓ Model saved!")
#
# # 14. Test set evaluation
# print("\n" + "=" * 80)
# print("TEST SET EVALUATION")
# print("=" * 80)
#
# back_X = test_df
# X_test = test_df.drop(columns=['datetime', 'pct_change', 'up_down','weighted_avg_sentiment'])
# y_test = test_df['up_down']
#
# # Clean test data
# test_clean = ~(y_test.isna() |
#                X_test.isna().any(axis=1) |
#                np.isinf(X_test.select_dtypes(include=[np.number])).any(axis=1))
#
# X_test = X_test[test_clean]
# y_test = y_test[test_clean]
#
# print(f"Test samples: {len(X_test):,}")
#
# train_probs = model.predict_proba(X_train)[:,1]
# t = np.quantile(train_probs, 0.9)
#
# # Predictions
# test_probs = model.predict_proba(X_test)[:, 1]
# test_preds = (test_probs >= t).astype(int)
# print("The threshold is:", t)
#
# print("Final sample lemgth", len(test_preds))
#
# print("\n" + "=" * 80)
# print("TEST SET RESULTS (DIRECTION PREDICTION)")
# print("=" * 80)
#
# print(f"Accuracy:  {accuracy_score(y_test, test_preds):.4f}")
# print(f"Precision: {precision_score(y_test, test_preds):.4f}")
# print(f"Recall:    {recall_score(y_test, test_preds):.4f}")
# print(f"F1-score:  {f1_score(y_test, test_preds):.4f}")
# print(f"ROC-AUC:   {roc_auc_score(y_test, test_probs):.4f}")
#
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, test_preds))
# print("=" * 80)
