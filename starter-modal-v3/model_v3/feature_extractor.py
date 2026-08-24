"""
Feature Extractor for V3 Model

Extracts all 10 features needed for prediction:
1. baseline_gemini (via Gemini API)
2-3. sentiment_positive, sentiment_negative (via FinBERT)
4-5. delta_positive, delta_negative (change from last quarter)
6. car1_lag1 (previous quarter's CAR1)
7. surprise (earnings surprise)
8. car1_vs_peers_lag1 (CAR1 vs industry peers)
9. surprise_vs_peers_lag1 (surprise vs industry peers)
10. r_m_volatility (market volatility from Alpha Vantage)
"""
from typing import Dict, Optional, List
from datetime import datetime
import os
import httpx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .alpha_vantage_client import AlphaVantageClient
from .peer_tracker import PeerTracker


class FeatureExtractor:
    """Extracts all 10 features for V3 model."""

    # Gemini configuration (matching v2 baseline)
    GEMINI_MODEL = "gemini-flash-latest"  # Stable Gemini 2.0 Flash
    GEMINI_MAX_RETRIES = 2
    GEMINI_TIMEOUT_SECONDS = 30.0

    def __init__(self,
                 alpha_vantage_client: AlphaVantageClient,
                 peer_tracker: PeerTracker,
                 gemini_api_key: Optional[str] = None,
                 historical_ticker_features: Optional[Dict] = None):
        """
        Args:
            alpha_vantage_client: Client for market data
            peer_tracker: Tracker for peer statistics
            gemini_api_key: Gemini API key for baseline_gemini feature
            historical_ticker_features: Pre-loaded historical features from training
        """
        self.av_client = alpha_vantage_client
        self.peer_tracker = peer_tracker
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')

        # FinBERT model (lazy load)
        self._finbert_tokenizer = None
        self._finbert_model = None

        # Historical features for lag/delta calculation
        # Format: {ticker: {'car1': float, 'sentiment_positive': float, ...}}
        self.historical_ticker_features = historical_ticker_features or {}

    def extract_features(self,
                        event: dict,
                        ticker: str,
                        summary_json: dict) -> Optional[Dict[str, float]]:
        """
        Extract all 10 features from an event.

        Args:
            event: Raw event dict from webhook
            ticker: Company ticker
            summary_json: Event summary JSON from information_url

        Returns:
            Dict with 10 features, or None if extraction fails
        """
        try:
            # Extract earnings facts
            facts = self._extract_facts(summary_json)
            if not facts:
                print(f"[WARN] No facts found for {ticker}")
                return None

            # Get surprise from summary JSON
            surprise = self._get_surprise(summary_json, ticker)

            # Get quarter from event
            quarter = self._extract_quarter(event)

            # Feature 1: baseline_gemini
            baseline_gemini = self._generate_baseline_gemini(facts)

            # Features 2-3: FinBERT sentiment
            sentiment_positive, sentiment_negative = self._get_finbert_sentiment(facts)

            # Features 4-6: Delta and lag features
            historical = self.historical_ticker_features.get(ticker, {})
            delta_positive = sentiment_positive - historical.get('sentiment_positive', sentiment_positive)
            delta_negative = sentiment_negative - historical.get('sentiment_negative', sentiment_negative)
            car1_lag1 = historical.get('car1', 0.0)

            # Features 8-9: Peer features
            peer_car1_lag1, peer_surprise_lag1 = self.peer_tracker.get_peer_stats_lag1(ticker, quarter)
            car1_vs_peers_lag1 = car1_lag1 - peer_car1_lag1
            surprise_vs_peers_lag1 = surprise - peer_surprise_lag1

            # Feature 10: Market volatility
            event_datetime = datetime.fromisoformat(event['event_datetime'].replace('Z', '+00:00'))
            _, r_m_volatility = self.av_client.get_market_data(event_datetime)

            # Assemble features dict
            features = {
                'baseline_gemini': baseline_gemini,
                'sentiment_positive': sentiment_positive,
                'sentiment_negative': sentiment_negative,
                'delta_positive': delta_positive,
                'delta_negative': delta_negative,
                'car1_lag1': car1_lag1,
                'surprise': surprise,
                'car1_vs_peers_lag1': car1_vs_peers_lag1,
                'surprise_vs_peers_lag1': surprise_vs_peers_lag1,
                'r_m_volatility': r_m_volatility
            }

            return features

        except Exception as e:
            print(f"[ERROR] Feature extraction failed for {ticker}: {e}")
            return None

    def update_historical_features(self, ticker: str, features: Dict[str, float],
                                   car1_predicted: float, quarter: str):
        """
        Update historical features after prediction for next time.

        Args:
            ticker: Company ticker
            features: Extracted features dict
            car1_predicted: Predicted CAR1 value
            quarter: Quarter string (e.g., "2026_Q4")
        """
        self.historical_ticker_features[ticker] = {
            'car1': car1_predicted,
            'sentiment_positive': features['sentiment_positive'],
            'sentiment_negative': features['sentiment_negative']
        }

        # Update peer tracker
        self.peer_tracker.update(ticker, quarter, car1_predicted, features['surprise'])

    def _extract_facts(self, summary_json: dict) -> List[str]:
        """Extract earnings call facts from summary JSON."""
        items = summary_json.get('items', [])
        for item in items:
            if item.get('id') == 'earnings-call-facts':
                return item.get('content', [])
        return []

    def _get_surprise(self, summary_json: dict, ticker: str) -> float:
        """Get earnings surprise from summary JSON.

        NOTE: The actual webhook information_url response format is uncertain.
        We try to get it from summary_json['metrics']['earnings_surprise'],
        falling back to 0.0 if not available.
        """
        try:
            metrics = summary_json.get('metrics', {})
            surprise_data = metrics.get('earnings_surprise', {})

            if surprise_data.get('surprise_status') == 'ok':
                surprise = surprise_data.get('surprise', 0.0)
                print(f"[INFO] Got earnings_surprise for {ticker}: {surprise:.6f}")
                return float(surprise)
            else:
                status = surprise_data.get('surprise_status', 'unknown')
                print(f"[WARN] Earnings surprise status '{status}' for {ticker} - using 0.0")
                return 0.0

        except Exception as e:
            print(f"[WARN] Could not get earnings_surprise for {ticker}: {e}")
            return 0.0

    def _extract_quarter(self, event: dict) -> str:
        """Extract quarter from event datetime."""
        event_datetime = datetime.fromisoformat(event['event_datetime'].replace('Z', '+00:00'))
        year = event_datetime.year
        month = event_datetime.month

        # Determine quarter
        if month <= 3:
            quarter = 'Q1'
        elif month <= 6:
            quarter = 'Q2'
        elif month <= 9:
            quarter = 'Q3'
        else:
            quarter = 'Q4'

        return f"{year}_{quarter}"

    def _generate_baseline_gemini(self, facts: List[str]) -> float:
        """Generate baseline_gemini prediction using Gemini API.

        Matches v2 implementation exactly - same prompt, same parsing.
        """
        if not GEMINI_AVAILABLE or not self.gemini_api_key:
            print("[WARN] Gemini not available, using fallback (0.5)")
            return 0.5

        if not facts:
            print("[WARN] No facts provided, using fallback (0.5)")
            return 0.5

        # Format facts as bullet points (matching v2)
        facts_str = "\n".join(f"- {fact}" for fact in facts)

        # Configure Gemini API
        genai.configure(api_key=self.gemini_api_key)

        # Exact prompt from v2 baseline implementation
        prompt = f"""Predict the unexpected stock return following an earnings call.

You are given key facts from a company's earnings call transcript.
Predict the stock's unexpected return as a class and percentile.

Base rates — calibrate your predictions to these proportions:
  - ~25% of stocks go UP (price increases 5%+ after the call)
  - ~50% of stocks are NEUTRAL (price moves less than 5%)
  - ~25% of stocks go DOWN (price decreases 5%+ after the call)

Consistency constraints between class and percentile:
  - "down"    → percentile in [0.00, 0.25]
  - "neutral" → percentile in [0.25, 0.75]
  - "up"      → percentile in [0.75, 1.00]

Your rationale must reference substantive evidence directly
(e.g., "Revenue grew 18% year-over-year…"). Never reference fact
numbers (e.g., never say "fact 3 shows…" or "according to fact 7").

---

Key facts from earnings call:
{facts_str}

---

Please provide:
1. predict_class: Exactly one of: "up" (5%+ increase), "neutral" (<5% move), "down" (5%+ decrease)
2. predict_percentile: Percentile rank of unexpected return: 0.0 (worst) to 1.0 (best)
3. rationale: 2-3 sentence explanation justifying the prediction using substantive evidence

Format your response as:
predict_class: [up/neutral/down]
predict_percentile: [0.0-1.0]
rationale: [your explanation]
"""

        # Retry loop (matching v2)
        for attempt in range(1, self.GEMINI_MAX_RETRIES + 1):
            try:
                model = genai.GenerativeModel(self.GEMINI_MODEL)
                response = model.generate_content(
                    prompt,
                    request_options={"timeout": self.GEMINI_TIMEOUT_SECONDS}
                )

                # Parse response (matching v2)
                text = response.text

                # Extract percentile from response
                # Look for "predict_percentile: X.XX" pattern
                for line in text.split('\n'):
                    line = line.strip()
                    if line.startswith('predict_percentile:'):
                        percentile_str = line.split(':', 1)[1].strip()
                        try:
                            percentile = float(percentile_str)
                            # Normalize to [0, 1]
                            if percentile > 1.0:
                                percentile = percentile / 100.0
                            percentile = max(0.0, min(1.0, percentile))
                            print(f"[INFO] Gemini baseline prediction: {percentile:.4f}")
                            return percentile
                        except ValueError:
                            continue

                # If we couldn't parse, fallback
                print(f"[WARN] Could not parse Gemini response, using fallback (0.5)")
                return 0.5

            except Exception as e:
                if attempt == self.GEMINI_MAX_RETRIES:
                    print(f"[ERROR] Gemini API failed after {self.GEMINI_MAX_RETRIES} attempts: {e}")
                    print(f"[WARN] Using fallback (0.5)")
                    return 0.5
                else:
                    print(f"[WARN] Gemini API error (attempt {attempt}/{self.GEMINI_MAX_RETRIES}): {e}, retrying...")
                    continue

        return 0.5

    def _get_finbert_sentiment(self, facts: List[str]) -> tuple[float, float]:
        """Run FinBERT on facts to get sentiment scores."""
        if not facts:
            return 0.33, 0.33

        # Lazy load FinBERT
        if self._finbert_tokenizer is None:
            self._finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            self._finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self._finbert_model.eval()

        # Combine facts into one text
        text = " ".join(facts)

        # Tokenize
        inputs = self._finbert_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )

        # Get predictions
        with torch.no_grad():
            outputs = self._finbert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT classes: [positive, negative, neutral]
        sentiment_positive = float(probs[0][0])
        sentiment_negative = float(probs[0][1])

        return sentiment_positive, sentiment_negative
