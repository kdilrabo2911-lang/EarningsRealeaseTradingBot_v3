"""
V3 Predictor - Main orchestrator for v3 model

Coordinates feature extraction and prediction using:
- FeatureExtractor (10 features)
- Linear regression model (trained coefficients)
- Sigmoid conversion (CAR1 → percentile)
"""
from typing import Dict, List, Optional
import json
from pathlib import Path
import httpx
import numpy as np
from scipy.special import expit

from .feature_extractor import FeatureExtractor
from .peer_tracker import PeerTracker
from .alpha_vantage_client import AlphaVantageClient


class V3Predictor:
    """V3 model predictor with 10 features."""

    # Timeout for fetching event summary
    SUMMARY_TIMEOUT_SECONDS = 15.0

    def __init__(self,
                 model_artifacts_dir: str,
                 gemini_api_key: Optional[str] = None,
                 alpha_vantage_api_key: Optional[str] = None):
        """
        Initialize V3 predictor.

        Args:
            model_artifacts_dir: Path to directory with model artifacts
            gemini_api_key: Gemini API key (optional, will use env var if not provided)
            alpha_vantage_api_key: Alpha Vantage API key (required for r_m_volatility)
        """
        artifacts_dir = Path(model_artifacts_dir)

        # Load model parameters
        with open(artifacts_dir / "model_params.json") as f:
            self.model_params = json.load(f)

        # Load ticker industry map
        with open(artifacts_dir / "ticker_industry_map.json") as f:
            ticker_industry_map = json.load(f)

        # Load historical industry stats
        with open(artifacts_dir / "historical_industry_stats.json") as f:
            historical_industry_stats = json.load(f)

        # Load historical ticker features
        with open(artifacts_dir / "historical_ticker_features.json") as f:
            historical_ticker_features = json.load(f)

        # Initialize components
        self.peer_tracker = PeerTracker(ticker_industry_map, historical_industry_stats)
        self.alpha_vantage = AlphaVantageClient(api_key=alpha_vantage_api_key)
        self.feature_extractor = FeatureExtractor(
            alpha_vantage_client=self.alpha_vantage,
            peer_tracker=self.peer_tracker,
            gemini_api_key=gemini_api_key,
            historical_ticker_features=historical_ticker_features
        )

        print(f"[INFO] V3 Predictor initialized")
        print(f"       Features: {self.model_params['features']}")
        print(f"       Sigmoid scale: {self.model_params['sigmoid_scale']}")
        print(f"       Training R²: {self.model_params.get('r2_score', 'N/A')}")

    def predict(self, event: dict) -> List[Dict[str, float]]:
        """
        Make predictions for an Explaining Markets event.

        Args:
            event: Event dict from webhook

        Returns:
            List of predictions: [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]
        """
        print("=" * 80)
        print("🤖 V3 LINEAR REGRESSION MODEL (+45% vs baseline)")
        print(f"   Features: {len(self.model_params['features'])} features")
        print(f"   Sigmoid scale: {self.model_params['sigmoid_scale']}")
        print("=" * 80)

        predictions = []

        for asset in event["focal_assets"]:
            ticker = asset["identifier_value"]

            try:
                # Fetch event summary
                summary_json = self._fetch_summary(event)
                if summary_json is None:
                    print(f"[WARN] Could not fetch summary for {ticker}, using fallback (0.5)")
                    predictions.append({
                        "identifier_value": ticker,
                        "predicted_percentile": 0.5
                    })
                    continue

                # Extract features
                features = self.feature_extractor.extract_features(event, ticker, summary_json)

                if features is None:
                    print(f"[WARN] Feature extraction failed for {ticker}, using fallback (0.5)")
                    predictions.append({
                        "identifier_value": ticker,
                        "predicted_percentile": 0.5
                    })
                    continue

                # Build feature vector in correct order
                feature_vector = np.array([
                    features[feat] for feat in self.model_params['features']
                ])

                # Linear regression: y = X · β + intercept
                predicted_car1 = (
                    np.dot(feature_vector, self.model_params['coefficients']) +
                    self.model_params['intercept']
                )

                # Convert CAR1 to percentile using sigmoid
                scale = self.model_params['sigmoid_scale']
                predicted_percentile = float(expit(predicted_car1 * scale))

                # Clip to [0, 1] (should already be, but safety)
                predicted_percentile = float(np.clip(predicted_percentile, 0.0, 1.0))

                print(f"[INFO] {ticker}: CAR1={predicted_car1:.4f}, percentile={predicted_percentile:.4f}")

                # Update historical features for next time
                quarter = self.feature_extractor._extract_quarter(event)
                self.feature_extractor.update_historical_features(
                    ticker, features, predicted_car1, quarter
                )

                predictions.append({
                    "identifier_value": ticker,
                    "predicted_percentile": predicted_percentile
                })

            except Exception as e:
                print(f"[ERROR] Prediction failed for {ticker}: {e}")
                predictions.append({
                    "identifier_value": ticker,
                    "predicted_percentile": 0.5
                })

        return predictions

    def _fetch_summary(self, event: dict) -> Optional[dict]:
        """Fetch event summary from information_url."""
        try:
            response = httpx.get(event["information_url"], timeout=self.SUMMARY_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[ERROR] Failed to fetch summary: {e}")
            return None
