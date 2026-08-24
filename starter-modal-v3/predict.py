"""★ V3 Linear Regression Model for Explaining Markets Competition ★

10-feature model with peer/industry tracking and market volatility.

Model beats baseline (Gemini + Surprise) by +45% on test data.

Features:
1. baseline_gemini - Gemini 2.0 Flash baseline prediction
2-3. sentiment_positive, sentiment_negative - FinBERT sentiment
4-5. delta_positive, delta_negative - Sentiment change from last quarter
6. car1_lag1 - Previous quarter's CAR1 (momentum)
7. surprise - Earnings surprise
8-9. car1_vs_peers_lag1, surprise_vs_peers_lag1 - Industry-relative features
10. r_m_volatility - Market volatility from Alpha Vantage
"""

from __future__ import annotations
import os
from pathlib import Path

import httpx  # Used by tests for mocking

from model_v3.v3_predictor import V3Predictor

# Global predictor instance (lazy initialization)
_predictor: V3Predictor | None = None


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    Uses V3 model with 10 features including peer tracking and market volatility.

    Returns:
        [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]
    """
    global _predictor

    # Lazy load predictor (once)
    if _predictor is None:
        # Try Modal path first, then local path
        artifacts_dir = Path("/root/model_artifacts")
        if not artifacts_dir.exists():
            artifacts_dir = Path(__file__).parent / "model_artifacts"

        # Get API keys from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        _predictor = V3Predictor(
            model_artifacts_dir=str(artifacts_dir),
            gemini_api_key=gemini_api_key,
            alpha_vantage_api_key=alpha_vantage_api_key
        )

    # Make prediction
    return _predictor.predict(event)
