"""
Test predict.py locally before Modal deployment
"""
import os
from dotenv import load_dotenv
import unittest.mock as mock

# Load .env
load_dotenv()

# Import predict function
from predict import predict

print("="*80)
print("TEST: predict.py (Local)")
print("="*80)

# Mock event
event = {
    "event_id": "ea_AAPL_Q4_2026",
    "event_datetime": "2027-01-15T21:00:00Z",
    "information_url": "https://mock-url.com/summary",
    "focal_assets": [
        {"identifier_value": "AAPL"}
    ]
}

# Mock fetch_summary in V3Predictor
from model_v3.v3_predictor import V3Predictor

def mock_fetch_summary(self, event):
    return {
        "items": [{
            "id": "earnings-call-facts",
            "content": [
                "Apple reported record Q4 revenue of $90.1 billion, up 8% year-over-year",
                "iPhone revenue grew 12% to $42.6 billion",
                "Services revenue reached $19.6 billion",
                "Operating margin improved to 29.3%",
                "Management guided for revenue growth to accelerate"
            ]
        }],
        "metrics": {
            "earnings_surprise": {
                "surprise": 0.025,
                "surprise_status": "ok"
            }
        }
    }

print("\nCalling predict() with mock event...")

with mock.patch.object(V3Predictor, '_fetch_summary', mock_fetch_summary):
    predictions = predict(event)

print("\nPredictions:")
for p in predictions:
    print(f"  {p['identifier_value']}: {p['predicted_percentile']:.4f}")

print("\n" + "="*80)
print("✅ predict.py works locally!")
print("="*80)
print("\nReady for Modal deployment with: modal deploy app.py")
