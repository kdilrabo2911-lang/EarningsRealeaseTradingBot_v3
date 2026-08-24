"""
Verify that V3 bot and backtest produce identical outputs.

Tests:
1. Bot receives webhook format (like v1/v2)
2. Backtest uses same data
3. Predictions match exactly
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import unittest.mock as mock

# Load environment
load_dotenv()

# Import bot predict function
from predict import predict as bot_predict

# Import backtest components
sys.path.insert(0, str(Path(__file__).parent.parent / "backtesting4_ideas"))
from verify_idea12 import extract_features, ols_fit2

print("="*80)
print("VERIFICATION: Bot vs Backtest Output Match")
print("="*80)

# Load test data (same as backtest)
print("\n1. Loading test data from backtest...")
data_path = Path(__file__).parent.parent / "data" / "train.csv"
df = pd.read_csv(data_path)

# Filter to test set (same as verify_idea12.py)
df = df[df['quarter'].isin(['2026_Q3', '2026_Q4'])]
print(f"   Test set: {len(df)} events")

# Get one sample event
sample = df.iloc[0]
print(f"\n2. Sample event:")
print(f"   Ticker: {sample['ticker']}")
print(f"   Quarter: {sample['quarter']}")
print(f"   Surprise: {sample['surprise']:.6f}")
print(f"   True CAR1: {sample['car1']:.6f}")

# Load summary data for this event
summaries_path = Path(__file__).parent.parent / "data" / "summaries.jsonl"
import json

# Find summary for this ticker/quarter
target_quarter = sample['quarter']
target_ticker = sample['ticker']

summary_data = None
with open(summaries_path) as f:
    for line in f:
        record = json.loads(line)
        if record['ticker'] == target_ticker and record['quarter'] == target_quarter:
            summary_data = record
            break

if not summary_data:
    print(f"❌ Could not find summary for {target_ticker} {target_quarter}")
    sys.exit(1)

print(f"   Found summary with {len(summary_data.get('summary', ''))} chars")

# 3. Test bot with webhook format (like v1/v2)
print("\n3. Testing bot with webhook format...")

# Convert to webhook format (same as v1/v2 bots receive)
webhook_event = {
    "event_id": f"ea_{target_ticker}_{target_quarter}",
    "event_datetime": "2026-10-15T21:00:00Z",  # Mock datetime in Q4 2026
    "event_type": "EARNING_RELEASE",
    "focal_assets": [
        {
            "identifier_value": target_ticker,
            "identifier_type": "ticker"
        }
    ],
    "information_url": f"https://api.explainingmarkets.ai/v1/summaries/events/ea_{target_ticker}_{target_quarter}"
}

# Mock the summary fetch to return our data
from model_v3.v3_predictor import V3Predictor

def mock_fetch_summary(self, event):
    """Return summary in expected format."""
    return {
        "items": [
            {
                "id": "earnings-call-facts",
                "content": [summary_data['summary']]  # Full earnings summary text
            }
        ],
        "metrics": {
            "earnings_surprise": {
                "surprise": float(sample['surprise']),
                "surprise_status": "ok"
            }
        }
    }

# Make bot prediction
with mock.patch.object(V3Predictor, '_fetch_summary', mock_fetch_summary):
    bot_predictions = bot_predict(webhook_event)

bot_percentile = bot_predictions[0]['predicted_percentile']
print(f"   Bot prediction: {bot_percentile:.6f}")

# 4. Calculate what backtest would predict
print("\n4. Calculating backtest prediction...")

# Load backtest data (with all features computed)
# For a fair comparison, we need to compute features the same way

# Actually, let me just verify the backtest R² first to ensure data loads correctly
print("\n5. Running quick backtest to verify R² matches...")

# Load data and compute features (from verify_idea12.py)
df_full = pd.read_csv(data_path)

# Split train/test same as backtest
train_df = df_full[df_full['quarter'].isin(['2024_Q1', '2024_Q2', '2024_Q3', '2024_Q4',
                                             '2025_Q1', '2025_Q2', '2025_Q3', '2025_Q4',
                                             '2026_Q1', '2026_Q2'])]
test_df = df_full[df_full['quarter'].isin(['2026_Q3', '2026_Q4'])]

print(f"   Train: {len(train_df)} events")
print(f"   Test: {len(test_df)} events")

# Load baseline predictions
baseline_path = Path(__file__).parent.parent / "data" / "baseline_predictions.jsonl"
baseline_preds = {}
with open(baseline_path) as f:
    for line in f:
        record = json.loads(line)
        key = (record['ticker'], record['quarter'])
        baseline_preds[key] = record['prediction']

# Add baseline to dataframes
train_df['baseline_gemini'] = train_df.apply(
    lambda x: baseline_preds.get((x['ticker'], x['quarter']), 0.5), axis=1
)
test_df['baseline_gemini'] = test_df.apply(
    lambda x: baseline_preds.get((x['ticker'], x['quarter']), 0.5), axis=1
)

# Extract features
print("\n6. Extracting features for backtest...")
train_features = extract_features(train_df)
test_features = extract_features(test_df)

# Fit model
from sklearn.linear_model import LinearRegression
from scipy.special import expit

X_train = train_features[['baseline_gemini', 'sentiment_positive', 'sentiment_negative',
                           'delta_positive', 'delta_negative', 'car1_lag1', 'surprise',
                           'car1_vs_peers_lag1', 'surprise_vs_peers_lag1', 'r_m_volatility']]
y_train = train_features['car1']

model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test
X_test = test_features[['baseline_gemini', 'sentiment_positive', 'sentiment_negative',
                         'delta_positive', 'delta_negative', 'car1_lag1', 'surprise',
                         'car1_vs_peers_lag1', 'surprise_vs_peers_lag1', 'r_m_volatility']]
y_test = test_features['car1']

car1_pred = model.predict(X_test)

# Convert to percentiles (sigmoid with scale=3)
scale = 3
percentiles = expit(car1_pred * scale)

# Calculate R²
r2_train, _ = ols_fit2(y_train, model.predict(X_train), train_features['surprise'])
r2_test, _ = ols_fit2(y_test, car1_pred, test_features['surprise'])

print(f"\n   Backtest R² (train): {r2_train:.6f}")
print(f"   Backtest R² (test):  {r2_test:.6f}")

# 7. Compare bot vs backtest for our sample event
sample_idx = test_features[test_features['ticker'] == target_ticker].index[0]
backtest_percentile = percentiles[test_features.index.get_loc(sample_idx)]

print(f"\n7. Comparing bot vs backtest for {target_ticker}:")
print(f"   Bot percentile:      {bot_percentile:.6f}")
print(f"   Backtest percentile: {backtest_percentile:.6f}")
print(f"   Difference:          {abs(bot_percentile - backtest_percentile):.6f}")

# Check if they're close (allow small numerical differences)
if abs(bot_percentile - backtest_percentile) < 0.01:
    print("\n✅ MATCH! Bot and backtest produce nearly identical outputs")
else:
    print(f"\n⚠️  MISMATCH! Difference of {abs(bot_percentile - backtest_percentile):.6f}")
    print("   Investigating why...")

    # Compare individual features
    print("\n   Backtest features:")
    backtest_row = test_features.loc[sample_idx]
    for col in X_test.columns:
        print(f"      {col:30} {backtest_row[col]:.6f}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
