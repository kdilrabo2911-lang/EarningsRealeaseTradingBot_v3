"""
Peer Tracker for Industry-Level Statistics

Tracks CAR1 and surprise by industry/quarter to calculate peer-relative features:
- car1_vs_peers_lag1
- surprise_vs_peers_lag1
"""
from typing import Dict, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path


class PeerTracker:
    """Tracks industry-level peer statistics across quarters."""

    def __init__(self, ticker_industry_map: Dict[str, str],
                 historical_stats: Optional[Dict] = None):
        """
        Args:
            ticker_industry_map: Maps ticker -> industry (e.g., {"AAPL": "Technology"})
            historical_stats: Pre-loaded historical stats from training
        """
        self.ticker_industry_map = ticker_industry_map

        # Structure: {quarter: {industry: {'car1_sum': float, 'car1_count': int, ...}}}
        self._stats = defaultdict(lambda: defaultdict(lambda: {
            'car1_sum': 0.0,
            'car1_count': 0,
            'surprise_sum': 0.0,
            'surprise_count': 0
        }))

        # Load historical stats if provided
        if historical_stats:
            self._load_historical_stats(historical_stats)

    def _load_historical_stats(self, historical_stats: Dict):
        """Load pre-computed historical stats from training."""
        # Format: [{quarter, industry, car1, surprise}, ...]
        # (column names might vary - support both formats)
        for record in historical_stats:
            quarter = record['quarter']
            industry = record['industry']

            # Get means - try multiple column name formats
            car1_mean = record.get('peer_car1_this_q') or record.get('car1', 0.0)
            surprise_mean = record.get('peer_surprise_this_q') or record.get('surprise', 0.0)

            # Assume 10 samples per industry/quarter (approximate)
            count = 10

            self._stats[quarter][industry] = {
                'car1_sum': car1_mean * count,
                'car1_count': count,
                'surprise_sum': surprise_mean * count,
                'surprise_count': count
            }

    def get_industry(self, ticker: str) -> str:
        """
        Get industry for a ticker. If unknown, classify as 'Other'.

        Args:
            ticker: Company ticker

        Returns:
            Industry string

        Note:
            Unknown tickers are classified as "Other" and use cross-industry peer averages.
            Could enhance with Yahoo Finance API for real-time industry lookup.
        """
        if ticker in self.ticker_industry_map:
            return self.ticker_industry_map[ticker]

        # Unknown ticker - classify as "Other"
        self.ticker_industry_map[ticker] = 'Other'
        print(f"[WARN] Unknown ticker '{ticker}' - using 'Other' industry")

        return 'Other'

    def update(self, ticker: str, quarter: str, car1: float, surprise: float):
        """
        Update industry stats with new observation.

        Args:
            ticker: Company ticker
            quarter: Quarter string (e.g., "2026_Q4")
            car1: Predicted or actual CAR1
            surprise: Earnings surprise
        """
        industry = self.get_industry(ticker)

        stats = self._stats[quarter][industry]
        stats['car1_sum'] += car1
        stats['car1_count'] += 1
        stats['surprise_sum'] += surprise
        stats['surprise_count'] += 1

    def get_peer_stats_lag1(self, ticker: str, current_quarter: str) -> Tuple[float, float]:
        """
        Get peer stats from PREVIOUS quarter.

        Args:
            ticker: Company ticker
            current_quarter: Current quarter string (e.g., "2026_Q4")

        Returns:
            (peer_car1_lag1, peer_surprise_lag1) tuple
            - Both are industry averages from previous quarter
        """
        industry = self.get_industry(ticker)
        prev_quarter = self._get_previous_quarter(current_quarter)

        if prev_quarter not in self._stats or industry not in self._stats[prev_quarter]:
            # No peer data available, return zeros
            return 0.0, 0.0

        stats = self._stats[prev_quarter][industry]

        # Calculate means
        car1_mean = (
            stats['car1_sum'] / stats['car1_count']
            if stats['car1_count'] > 0 else 0.0
        )
        surprise_mean = (
            stats['surprise_sum'] / stats['surprise_count']
            if stats['surprise_count'] > 0 else 0.0
        )

        return car1_mean, surprise_mean

    def _get_previous_quarter(self, current_quarter: str) -> str:
        """
        Get previous quarter string.

        Examples:
            2026_Q4 -> 2026_Q3
            2026_Q1 -> 2025_Q4
        """
        parts = current_quarter.split('_')
        if len(parts) != 2:
            return ""

        year = int(parts[0])
        quarter = parts[1]  # e.g., "Q4"

        if quarter == "Q1":
            return f"{year-1}_Q4"
        elif quarter == "Q2":
            return f"{year}_Q1"
        elif quarter == "Q3":
            return f"{year}_Q2"
        elif quarter == "Q4":
            return f"{year}_Q3"
        else:
            return ""

    def get_stats_summary(self) -> Dict:
        """Get summary of tracked stats for debugging."""
        total_quarters = len(self._stats)
        total_industries = sum(len(industries) for industries in self._stats.values())
        total_observations = sum(
            stats['car1_count']
            for quarter_stats in self._stats.values()
            for stats in quarter_stats.values()
        )

        return {
            'quarters_tracked': total_quarters,
            'total_industry_quarters': total_industries,
            'total_observations': total_observations
        }

    @classmethod
    def from_json_files(cls, industry_map_path: Path, historical_stats_path: Path):
        """
        Factory method to create PeerTracker from JSON files.

        Args:
            industry_map_path: Path to ticker_industry_map.json
            historical_stats_path: Path to historical_industry_stats.json

        Returns:
            PeerTracker instance
        """
        # Load ticker -> industry map
        with open(industry_map_path) as f:
            ticker_industry_map = json.load(f)

        # Load historical stats
        with open(historical_stats_path) as f:
            historical_stats = json.load(f)

        return cls(ticker_industry_map, historical_stats)
