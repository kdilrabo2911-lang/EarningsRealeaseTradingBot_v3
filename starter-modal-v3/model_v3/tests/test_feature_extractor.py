"""
Test FeatureExtractor Class

Tests all 10 features using REAL Gemini API (same as v2 bot):
1. baseline_gemini (via Gemini API with exact v2 prompt)
2-3. sentiment_positive, sentiment_negative
4-5. delta_positive, delta_negative
6. car1_lag1
7. surprise
8-9. car1_vs_peers_lag1, surprise_vs_peers_lag1
10. r_m_volatility
"""
import sys
from pathlib import Path
from datetime import datetime
import json
import os

# Load .env file for local testing
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_v3.feature_extractor import FeatureExtractor
from model_v3.peer_tracker import PeerTracker
from model_v3.alpha_vantage_client import AlphaVantageClient

print("="*80)
print("TEST: FeatureExtractor")
print("="*80)

# Load artifacts
artifacts_dir = Path(__file__).parent.parent.parent / "model_artifacts"

with open(artifacts_dir / "ticker_industry_map.json") as f:
    ticker_industry_map = json.load(f)

with open(artifacts_dir / "historical_industry_stats.json") as f:
    historical_industry_stats = json.load(f)

with open(artifacts_dir / "historical_ticker_features.json") as f:
    historical_ticker_features = json.load(f)

# Initialize components
peer_tracker = PeerTracker(ticker_industry_map, historical_industry_stats)
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
if not alpha_vantage_api_key:
    print("[WARN] ALPHA_VANTAGE_API_KEY not set - alpha_vantage calls will fail")
alpha_vantage = AlphaVantageClient(api_key=alpha_vantage_api_key)

# Get Gemini API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("[WARN] GEMINI_API_KEY not set - baseline_gemini will use fallback (0.5)")

extractor = FeatureExtractor(
    alpha_vantage_client=alpha_vantage,
    peer_tracker=peer_tracker,
    gemini_api_key=gemini_api_key,
    historical_ticker_features=historical_ticker_features
)

print("\n1. Testing with mock event (AAPL)...")

# Mock event matching real webhook format
event = {
    "event_id": "ea_AAPL_Q4_2026",
    "event_datetime": "2027-01-15T21:00:00Z",
    "focal_assets": [{"identifier_value": "AAPL"}]
}

# Mock summary JSON (would come from information_url in real usage)
summary_json = {
    "items": [
        {
            "id": "earnings-call-facts",
            "content": [
                "Apple reported record Q4 revenue of $90.1 billion, up 8% year-over-year",
                "iPhone revenue grew 12% to $42.6 billion, driven by strong iPhone 15 sales",
                "Services revenue reached $19.6 billion, a new all-time high",
                "Operating margin improved to 29.3%, up from 28.1% last year",
                "Management guided for revenue growth to accelerate in Q1"
            ]
        }
    ],
    "metrics": {
        "earnings_surprise": {
            "surprise": 0.025,
            "surprise_status": "ok"
        }
    }
}

print(f"\n[INFO] Testing with {len(summary_json['items'][0]['content'])} earnings facts")
print(f"[INFO] Calling REAL Gemini API (same as v2 bot)...")

# Extract features (this will call real Gemini API!)
features = extractor.extract_features(event, "AAPL", summary_json)

if features is None:
    print("❌ Feature extraction failed!")
    sys.exit(1)

print("\n2. Extracted features:")
for feat, val in features.items():
    print(f"  {feat:30} {val:.6f}")

# Verify all 10 features present
expected_features = [
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

print("\n3. Verifying all features present...")
missing = [f for f in expected_features if f not in features]
if missing:
    print(f"❌ Missing features: {missing}")
    sys.exit(1)
else:
    print("✅ All 10 features present!")

# Verify reasonable values
print("\n4. Verifying feature ranges...")
checks = []

checks.append(("baseline_gemini in [0,1]", 0 <= features['baseline_gemini'] <= 1))
checks.append(("sentiment_positive in [0,1]", 0 <= features['sentiment_positive'] <= 1))
checks.append(("sentiment_negative in [0,1]", 0 <= features['sentiment_negative'] <= 1))
checks.append(("delta_positive in [-1,1]", -1 <= features['delta_positive'] <= 1))
checks.append(("delta_negative in [-1,1]", -1 <= features['delta_negative'] <= 1))
checks.append(("surprise matches input", abs(features['surprise'] - 0.025) < 0.001))
checks.append(("r_m_volatility >= 0", features['r_m_volatility'] >= 0))

for name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"  {name:40} {status}")
    if not passed:
        print(f"    Value: {features.get(name.split()[0], 'N/A')}")
        sys.exit(1)

print("\n5. Testing update_historical_features...")

# Update with predicted CAR1
predicted_car1 = 0.025
extractor.update_historical_features("AAPL", features, predicted_car1, "2026_Q4")

# Verify sentiment and car1 were saved
if "AAPL" in extractor.historical_ticker_features:
    saved = extractor.historical_ticker_features["AAPL"]
    print(f"  Saved sentiment_positive: {saved.get('sentiment_positive', 'MISSING'):.6f}")
    print(f"  Saved sentiment_negative: {saved.get('sentiment_negative', 'MISSING'):.6f}")
    print(f"  Saved car1_lag1: {saved.get('car1', 'MISSING'):.6f}")

    if 'sentiment_positive' in saved and 'car1' in saved:
        print("✅ Historical features updated!")
    else:
        print("❌ Update failed!")
        sys.exit(1)
else:
    print("❌ AAPL not in historical features!")
    sys.exit(1)

print("\n6. Testing delta features (2nd event for same ticker)...")

# Extract again - should now have deltas based on previous sentiment
event2 = {
    "event_id": "ea_AAPL_Q1_2027",
    "event_datetime": "2027-04-15T21:00:00Z",
    "focal_assets": [{"identifier_value": "AAPL"}]
}

summary_json2 = {
    "items": [
        {
            "id": "earnings-call-facts",
            "content": [
                "Apple Q1 revenue reached $97.3 billion, exceeding expectations",
                "iPhone revenue continued strong momentum with 15% growth",
                "Services hit $20.2 billion, maintaining high margin profile",
                "Gross margin expanded to 44.5%, beating guidance",
                "Company announced $90 billion buyback program"
            ]
        }
    ],
    "metrics": {
        "earnings_surprise": {
            "surprise": 0.030,
            "surprise_status": "ok"
        }
    }
}

print(f"\n[INFO] Extracting 2nd event (will have deltas and car1_lag1)...")
features2 = extractor.extract_features(event2, "AAPL", summary_json2)

if features2 is None:
    print("❌ 2nd feature extraction failed!")
    sys.exit(1)

print("\nExtracted features (2nd event):")
print(f"  delta_positive:  {features2['delta_positive']:.6f}")
print(f"  delta_negative:  {features2['delta_negative']:.6f}")
print(f"  car1_lag1:       {features2['car1_lag1']:.6f}")

# Verify car1_lag1 matches previous prediction
if abs(features2['car1_lag1'] - predicted_car1) < 0.001:
    print("✅ car1_lag1 correctly uses previous prediction!")
else:
    print(f"❌ car1_lag1 mismatch: expected {predicted_car1:.6f}, got {features2['car1_lag1']:.6f}")
    sys.exit(1)

# Deltas should be difference from previous sentiment
expected_delta_pos = features2['sentiment_positive'] - features['sentiment_positive']
if abs(features2['delta_positive'] - expected_delta_pos) < 0.001:
    print("✅ delta_positive computed correctly!")
else:
    print(f"❌ delta_positive: expected {expected_delta_pos:.6f}, got {features2['delta_positive']:.6f}")

print("\n" + "="*80)
print("✅ FeatureExtractor tests PASSED")
print("="*80)
print("\nNOTE: This test uses REAL Gemini API calls (same as v2 bot)")
print("      Each test run costs ~$0.002 in API fees")
