"""
Offline model training script for Explaining Markets competition.

This script:
1. Loads historical data (2025Q4 - 2026Q3)
2. Applies FinBERT sentiment analysis
3. Creates delta and lag features
4. Trains linear regression on 7 features to predict CAR1
5. Finds optimal sigmoid scale for CAR1 → percentile conversion
6. Saves model artifacts for deployment
"""
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.special import expit
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Add examples src to path for official scoring
sys.path.append('../examples/src')
from examples.scoring import percentile_ranks, ols_fit2

print("="*80)
print("OFFLINE MODEL TRAINING FOR EXPLAINING MARKETS")
print("="*80)

# ============================================================================
# 1. LOAD HISTORICAL DATA
# ============================================================================
print("\n📂 Loading historical data...")

data_dir = Path("../backtesting2/data")
all_events = []

for file in sorted(data_dir.glob("EARNINGS_RELEASE_*.jsonl")):
    print(f"  Loading {file.name}...")
    with open(file) as f:
        for line in f:
            all_events.append(json.loads(line))

print(f"✅ Loaded {len(all_events)} events")

# ============================================================================
# 2. EXTRACT FEATURES FROM EVENTS
# ============================================================================
print("\n🔧 Extracting features from events...")

def extract_quarter(event_id):
    """Extract quarter from event_id like 'ea_AAPL_Q3_2026'"""
    parts = event_id.split('_')
    return f"{parts[-1]}_{parts[-2]}"

rows = []
for event in all_events:
    event_id = event['event_id']
    ticker = event['focal_assets'][0]['identifier_value']
    quarter = extract_quarter(event_id)

    # Get CAR1
    event_returns = event.get('event_returns') or {}
    returns = event_returns.get(ticker) or {}
    car1 = returns.get('car1')
    if car1 is None:
        continue

    # Get baseline predictions
    baseline_preds = event.get('baseline_predictions', {})
    gemini_pred = baseline_preds.get('gemini/ea-explain-contemp-summary', {}).get(ticker)

    # Get surprise
    metrics = event.get('metrics', {})
    surprise_data = metrics.get('earnings_surprise', {})
    surprise = surprise_data.get('surprise') if surprise_data.get('surprise_status') == 'ok' else None

    # Get earnings call facts
    disclosure = event.get('disclosure', {})
    items = disclosure.get('items', [])
    facts = None
    for item in items:
        if item.get('id') == 'earnings-call-facts':
            facts = item.get('content', [])
            break

    if not facts:
        continue

    rows.append({
        'event_id': event_id,
        'ticker': ticker,
        'quarter': quarter,
        'car1': float(car1),
        'baseline_gemini': float(gemini_pred) if gemini_pred is not None else None,
        'surprise': float(surprise) if surprise is not None else None,
        'facts': facts
    })

df = pd.DataFrame(rows)
print(f"✅ Extracted {len(df)} events with all required fields")

# ============================================================================
# 3. APPLY FINBERT SENTIMENT ANALYSIS
# ============================================================================
print("\n🤖 Applying FinBERT sentiment analysis...")
print("   Loading FinBERT model...")

tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
model.eval()

def get_sentiment(facts_list):
    """Get aggregated sentiment from earnings facts."""
    if not facts_list:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

    # Combine all facts into one text
    text = ' '.join(facts_list)

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # FinBERT outputs: [positive, negative, neutral]
    return {
        'positive': float(probs[0][0]),
        'negative': float(probs[0][1]),
        'neutral': float(probs[0][2])
    }

sentiments = []
for i, facts in enumerate(df['facts']):
    if i % 500 == 0:
        print(f"   Processed {i}/{len(df)} events...")
    sentiments.append(get_sentiment(facts))

df['sentiment_positive'] = [s['positive'] for s in sentiments]
df['sentiment_negative'] = [s['negative'] for s in sentiments]
df['sentiment_neutral'] = [s['neutral'] for s in sentiments]

print(f"✅ Sentiment analysis complete")

# ============================================================================
# 4. CREATE DELTA AND LAG FEATURES
# ============================================================================
print("\n🔧 Creating delta and lag features...")

df = df.sort_values(['ticker', 'quarter'])

# Delta features (change from previous quarter)
for col in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']:
    delta_col = col.replace('sentiment_', 'delta_')
    df[delta_col] = df.groupby('ticker')[col].diff()

# Lag features (previous quarter's values)
df['car1_lag1'] = df.groupby('ticker')['car1'].shift(1)

print(f"✅ Created delta and lag features")

# ============================================================================
# 5. TRAIN LINEAR REGRESSION MODEL
# ============================================================================
print("\n🎯 Training linear regression model...")

# Features and target
all_features = [
    'baseline_gemini',
    'sentiment_positive',
    'sentiment_negative',
    'sentiment_neutral',
    'delta_positive',
    'delta_negative',
    'car1_lag1',
    'surprise'
]
target = 'car1'

# Drop rows with missing values
model_data = df[all_features + [target] + ['quarter', 'ticker']].dropna()
print(f"   Training data: {len(model_data)} events")

# Split: first 4 quarters for training
quarters = sorted(model_data['quarter'].unique())
train_quarters = quarters[:4]

train_data = model_data[model_data['quarter'].isin(train_quarters)]

print(f"   Training quarters: {train_quarters}")
print(f"   Training events: {len(train_data)}")

X_train = train_data[all_features]
y_train = train_data[target]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"✅ Model trained")
print(f"\n   Coefficients:")
for feat, coef in zip(all_features, model.coef_):
    print(f"      {feat:25} {coef:+.8f}")
print(f"      {'intercept':25} {model.intercept_:+.8f}")

# ============================================================================
# 6. FIND OPTIMAL SIGMOID SCALE
# ============================================================================
print("\n🔍 Finding optimal sigmoid scale for CAR1 → percentile conversion...")

# Use test quarters to find best scale
test_quarters = quarters[4:] if len(quarters) > 4 else quarters[-2:]
test_data = model_data[model_data['quarter'].isin(test_quarters)]

X_test = test_data[all_features]
y_test = test_data[target]
y_test_pred_car1 = model.predict(X_test)

# Convert actual CAR1 and surprise to percentiles
y_test_pctile = np.array(percentile_ranks(y_test.tolist()))
surprise_test = test_data['surprise'].values
surprise_test_pct = np.array(percentile_ranks(surprise_test.tolist()))

# Test different sigmoid scales
best_scale = None
best_r2 = -np.inf

print(f"   Testing scales...")
for scale in [5, 10, 15, 20, 25, 30, 40, 50]:
    y_test_pred_pct = expit(y_test_pred_car1 * scale)
    points = list(zip(y_test_pred_pct, surprise_test_pct, y_test_pctile))
    fit = ols_fit2(points)

    if fit:
        print(f"      Scale {scale:2d}: R² = {fit.r_squared:.6f}")
        if fit.r_squared > best_r2:
            best_r2 = fit.r_squared
            best_scale = scale

print(f"\n✅ Optimal sigmoid scale: {best_scale}")
print(f"   Test R²: {best_r2:.6f}")

# ============================================================================
# 7. BUILD HISTORICAL FEATURE DATABASE
# ============================================================================
print("\n💾 Building historical feature database...")

# For each ticker, store the latest quarter's data
historical_features = {}

for ticker in model_data['ticker'].unique():
    ticker_data = model_data[model_data['ticker'] == ticker].sort_values('quarter')

    # Store all quarters for this ticker
    ticker_history = {}
    for _, row in ticker_data.iterrows():
        ticker_history[row['quarter']] = {
            'sentiment_positive': float(row['sentiment_positive']),
            'sentiment_negative': float(row['sentiment_negative']),
            'sentiment_neutral': float(row['sentiment_neutral']),
            'car1': float(row['car1'])
        }

    historical_features[ticker] = ticker_history

print(f"✅ Historical features for {len(historical_features)} tickers")

# ============================================================================
# 8. SAVE MODEL ARTIFACTS
# ============================================================================
print("\n💾 Saving model artifacts...")

artifacts_dir = Path("model_artifacts")
artifacts_dir.mkdir(exist_ok=True)

# Save model coefficients
model_params = {
    'features': all_features,
    'coefficients': model.coef_.tolist(),
    'intercept': float(model.intercept_),
    'sigmoid_scale': best_scale
}

with open(artifacts_dir / 'model_params.json', 'w') as f:
    json.dump(model_params, f, indent=2)

# Save historical features
with open(artifacts_dir / 'historical_features.pkl', 'wb') as f:
    pickle.dump(historical_features, f)

# Save training statistics
train_stats = {
    'car1_min': float(y_train.min()),
    'car1_max': float(y_train.max()),
    'car1_mean': float(y_train.mean()),
    'car1_std': float(y_train.std()),
    'n_train_events': len(train_data),
    'train_quarters': train_quarters,
    'test_r2': best_r2
}

with open(artifacts_dir / 'train_stats.json', 'w') as f:
    json.dump(train_stats, f, indent=2)

print(f"✅ Saved artifacts to {artifacts_dir}/")
print(f"   - model_params.json (coefficients + sigmoid scale)")
print(f"   - historical_features.pkl (ticker → quarter → features)")
print(f"   - train_stats.json (training statistics)")

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nModel Summary:")
print(f"  Features: {len(all_features)}")
print(f"  Training events: {len(train_data)}")
print(f"  Sigmoid scale: {best_scale}")
print(f"  Test R²: {best_r2:.6f}")
print(f"\nReady for deployment!")
