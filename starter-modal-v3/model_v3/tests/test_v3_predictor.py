"""
Test V3Predictor Class

Integration test for full prediction pipeline.
"""
import sys
from pathlib import Path
import os

# Load .env file for local testing
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_v3.v3_predictor import V3Predictor

print("="*80)
print("TEST: V3Predictor (Full Integration)")
print("="*80)

# Get API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize predictor
artifacts_dir = Path(__file__).parent.parent.parent / "model_artifacts"

print("\n1. Initializing V3Predictor...")
predictor = V3Predictor(
    model_artifacts_dir=str(artifacts_dir),
    gemini_api_key=gemini_api_key,
    alpha_vantage_api_key=alpha_vantage_api_key
)

print("\n2. Testing prediction with mock webhook event...")

# Mock event (matches real webhook format)
event = {
    "event_id": "ea_AAPL_Q4_2026",
    "event_datetime": "2027-01-15T21:00:00Z",
    "information_url": "https://mock-url.com/summary",  # Would fetch real data in production
    "focal_assets": [
        {"identifier_value": "AAPL"}
    ]
}

# For testing, we'll manually inject summary data
# In production, this would be fetched from information_url
import unittest.mock as mock

def mock_fetch_summary(self, event):
    """Mock summary fetch for testing."""
    return {
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

# Patch the _fetch_summary method
with mock.patch.object(V3Predictor, '_fetch_summary', mock_fetch_summary):
    predictions = predictor.predict(event)

print("\n3. Verifying predictions...")

if len(predictions) != 1:
    print(f"❌ Expected 1 prediction, got {len(predictions)}")
    sys.exit(1)

pred = predictions[0]

if pred["identifier_value"] != "AAPL":
    print(f"❌ Wrong ticker: expected AAPL, got {pred['identifier_value']}")
    sys.exit(1)

percentile = pred["predicted_percentile"]

if not (0 <= percentile <= 1):
    print(f"❌ Percentile out of range: {percentile}")
    sys.exit(1)

print(f"\n✅ Prediction successful:")
print(f"   Ticker: {pred['identifier_value']}")
print(f"   Percentile: {percentile:.4f}")

print("\n4. Testing multi-ticker event...")

event_multi = {
    "event_id": "ea_MULTI_Q4_2026",
    "event_datetime": "2027-01-15T21:00:00Z",
    "information_url": "https://mock-url.com/summary",
    "focal_assets": [
        {"identifier_value": "AAPL"},
        {"identifier_value": "MSFT"}
    ]
}

with mock.patch.object(V3Predictor, '_fetch_summary', mock_fetch_summary):
    predictions_multi = predictor.predict(event_multi)

if len(predictions_multi) != 2:
    print(f"❌ Expected 2 predictions, got {len(predictions_multi)}")
    sys.exit(1)

print(f"\n✅ Multi-ticker prediction successful:")
for p in predictions_multi:
    print(f"   {p['identifier_value']}: {p['predicted_percentile']:.4f}")

print("\n" + "="*80)
print("✅ V3Predictor integration tests PASSED")
print("="*80)
print("\nNOTE: Predictor ready for Modal deployment")
print("      Next: Update predict.py to use V3Predictor")
