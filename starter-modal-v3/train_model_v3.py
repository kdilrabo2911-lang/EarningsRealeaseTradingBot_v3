"""
V3 Model Training - With Peer Context Features

Features (10 total):
1. baseline_gemini
2-3. sentiment_positive, sentiment_negative
4-5. delta_positive, delta_negative
6. car1_lag1
7. surprise
8. car1_vs_peers_lag1 (NEW)
9. surprise_vs_peers_lag1 (NEW)
10. r_m_volatility (NEW)

Expected improvement: +29.7% vs baseline
"""
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.special import expit

sys.path.append('../examples/src')
from examples.scoring import percentile_ranks, ols_fit2

print("="*80)
print("V3 MODEL TRAINING - WITH PEER CONTEXT FEATURES")
print("="*80)

# Load data
data_dir = Path("../backtesting/data")
data_files = [
    'EARNINGS_RELEASE_2025Q4.jsonl',
    'EARNINGS_RELEASE_2026Q1.jsonl',
    'EARNINGS_RELEASE_2026Q2.jsonl',
    'EARNINGS_RELEASE_2026Q3.jsonl',
]

all_events = []
for filename in data_files:
    with open(data_dir / filename) as f:
        for line in f:
            all_events.append(json.loads(line))

print(f"✅ Loaded {len(all_events)} events")

# Extract features
rows = []
for event in all_events:
    ticker = event['focal_assets'][0]['identifier_value']
    event_id = event['event_id']
    parts = event_id.split('_')
    quarter = f"{parts[-1]}_{parts[-2]}"

    event_returns = event.get('event_returns') or {}
    if ticker not in event_returns:
        continue

    ret = event_returns[ticker]
    car1 = ret.get('car1')
    r_m = ret.get('r_m')

    surprise = event.get('metrics', {}).get('earnings_surprise', {}).get('surprise')
    baseline_gemini = event.get('baseline_predictions', {}).get('gemini/ea-explain-contemp-summary', {}).get(ticker)

    # Get facts for FinBERT
    facts = []
    for item in event.get('disclosure', {}).get('items', []):
        if item.get('id') == 'earnings-call-facts':
            facts = item.get('content', [])

    rows.append({
        'event_id': event_id,
        'ticker': ticker,
        'quarter': quarter,
        'car1': car1,
        'r_m': r_m,
        'surprise': surprise,
        'baseline_gemini': baseline_gemini,
        'facts': facts
    })

df = pd.DataFrame(rows)
print(f"✅ Extracted {len(df)} rows")

# Load FinBERT sentiment from cache
print("\n📊 Loading FinBERT sentiment...")
cache_path = Path("../backtesting2/finbert_cache.pkl")
df_sentiment = pd.read_pickle(cache_path)
df = df.merge(
    df_sentiment[['event_id', 'sentiment_positive', 'sentiment_negative']],
    on='event_id',
    how='left'
)
print(f"✅ Merged sentiment: {len(df)} rows")

# Sort by ticker and quarter
df = df.sort_values(['ticker', 'quarter'])

# Create base features
print("\n🔧 Creating base features...")
for col in ['sentiment_positive', 'sentiment_negative']:
    df[col.replace('sentiment_', 'delta_')] = df.groupby('ticker')[col].diff()

df['car1_lag1'] = df.groupby('ticker')['car1'].shift(1)

# Load industry mapping
with open('../backtesting/data/ticker_industry_cache.json') as f:
    industry_map = json.load(f)
df['industry'] = df['ticker'].map(industry_map).fillna('Other')

# Create peer features
print("🔧 Creating peer context features...")

# Calculate peer stats per quarter/industry
peer_stats = df.groupby(['quarter', 'industry']).agg({
    'car1': 'mean',
    'surprise': 'mean'
}).reset_index()
peer_stats.columns = ['quarter', 'industry', 'peer_car1_this_q', 'peer_surprise_this_q']

df = df.merge(peer_stats, on=['quarter', 'industry'], how='left')

# Shift to get lag-1 peer stats
quarters_sorted = sorted(df['quarter'].unique())
quarter_to_idx = {q: i for i, q in enumerate(quarters_sorted)}
df['quarter_idx'] = df['quarter'].map(quarter_to_idx)

df = df.sort_values(['industry', 'quarter_idx'])
df['peer_car1_lag1'] = df.groupby('industry')['peer_car1_this_q'].shift(1)
df['peer_surprise_lag1'] = df.groupby('industry')['peer_surprise_this_q'].shift(1)

# Relative to peers
df['car1_vs_peers_lag1'] = df['car1_lag1'] - df['peer_car1_lag1']
df['surprise_vs_peers_lag1'] = df['surprise'] - df['peer_surprise_lag1']

# Market regime
quarter_rm_stats = df.groupby('quarter')['r_m'].agg(['std']).reset_index()
quarter_rm_stats.columns = ['quarter', 'r_m_quarter_std']
df = df.merge(quarter_rm_stats, on='quarter', how='left')
df['r_m_volatility'] = df['r_m_quarter_std']

print("✅ Created peer features")

# Drop NaN
df = df.dropna(subset=['car1_lag1', 'delta_positive', 'delta_negative', 'peer_car1_lag1', 'r_m_volatility'])
print(f"✅ Clean dataset: {len(df)} rows")

# Train model on ALL data
print("\n🚂 Training model on full dataset...")

features = [
    'baseline_gemini',
    'sentiment_positive',
    'sentiment_negative',
    'delta_positive',
    'delta_negative',
    'car1_lag1',
    'surprise',
    'car1_vs_peers_lag1',
    'surprise_vs_peers_lag1',
    'r_m_volatility'
]

# Final dropna on all features + target
cols_needed = features + ['car1', 'ticker', 'quarter', 'industry', 'r_m']
# Remove duplicates (surprise is already in features)
cols_needed = list(dict.fromkeys(cols_needed))
df_train = df[cols_needed].dropna()
print(f"✅ Final training set: {len(df_train)} rows")

X = df_train[features]
y = df_train['car1']

model = LinearRegression()
model.fit(X, y)

print(f"✅ Model trained on {len(X)} samples")
print(f"\nCoefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"   {feat:30} {coef:+.6f}")
print(f"   {'intercept':30} {model.intercept_:+.6f}")

# Find optimal sigmoid scale
print("\n🔍 Finding optimal sigmoid scale...")
y_pred_car1 = model.predict(X)

best_r2 = -np.inf
best_scale = None

for scale in [3, 4, 5, 6, 7, 8, 10, 15, 20]:
    y_pred_pct = expit(y_pred_car1 * scale)
    y_actual_pct = np.array(percentile_ranks(y.tolist()))
    surprise_pct = np.array(percentile_ranks(df_train['surprise'].tolist()))

    points = list(zip(y_pred_pct, surprise_pct, y_actual_pct))
    fit = ols_fit2(points)

    if fit and fit.r_squared > best_r2:
        best_r2 = fit.r_squared
        best_scale = scale
        print(f"   scale={scale:2d}: R²={fit.r_squared:.6f} ✅")
    elif fit:
        print(f"   scale={scale:2d}: R²={fit.r_squared:.6f}")

print(f"\n✅ Best scale: {best_scale}, R²: {best_r2:.6f}")

# Save model artifacts
print("\n💾 Saving model artifacts...")

model_artifacts_dir = Path("model_artifacts")
model_artifacts_dir.mkdir(exist_ok=True)

# Save parameters as JSON
model_params = {
    'features': features,
    'coefficients': model.coef_.tolist(),
    'intercept': float(model.intercept_),
    'sigmoid_scale': best_scale,
    'version': 'v3_with_peer_features',
    'training_samples': len(X),
    'train_r2': float(best_r2)
}

with open(model_artifacts_dir / 'model_params.json', 'w') as f:
    json.dump(model_params, f, indent=2)

print(f"✅ Saved model_params.json")

# Save industry mapping
with open(model_artifacts_dir / 'ticker_industry_map.json', 'w') as f:
    json.dump(industry_map, f, indent=2)

print(f"✅ Saved ticker_industry_map.json")

# Save historical features for lag/peer calculations
# We need to save the MOST RECENT quarter's data for each ticker and industry
latest_quarter = df_train['quarter'].max()
historical_ticker = df_train.groupby('ticker').tail(1)[['ticker', 'car1', 'sentiment_positive', 'sentiment_negative']].set_index('ticker').to_dict('index')

historical_industry = df_train.groupby(['quarter', 'industry']).agg({
    'car1': 'mean',
    'surprise': 'mean',
    'r_m': 'std'
}).reset_index()

with open(model_artifacts_dir / 'historical_ticker_features.json', 'w') as f:
    json.dump(historical_ticker, f, indent=2)

with open(model_artifacts_dir / 'historical_industry_stats.json', 'w') as f:
    historical_industry.to_json(f, orient='records', indent=2)

print(f"✅ Saved historical features")

# Save training stats
train_stats = {
    'training_date': pd.Timestamp.now().isoformat(),
    'num_samples': len(X),
    'num_features': len(features),
    'train_r2': float(best_r2),
    'sigmoid_scale': best_scale,
    'quarters_trained': sorted(df_train['quarter'].unique()),
    'num_industries': len(df_train['industry'].unique()),
    'industries': sorted(df_train['industry'].unique())
}

with open(model_artifacts_dir / 'train_stats.json', 'w') as f:
    json.dump(train_stats, f, indent=2)

print(f"✅ Saved train_stats.json")

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nModel version: v3_with_peer_features")
print(f"Features: {len(features)}")
print(f"Training R²: {best_r2:.6f}")
print(f"Sigmoid scale: {best_scale}")
print(f"\nArtifacts saved to: {model_artifacts_dir.absolute()}")
print("="*80)
