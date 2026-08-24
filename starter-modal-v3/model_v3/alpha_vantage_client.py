"""
Alpha Vantage Client for Market Data

Fetches SPY (S&P 500) daily data and calculates:
- r_m: Daily market return
- r_m_volatility: 60-day rolling standard deviation
"""
from datetime import datetime, timedelta
from typing import Tuple, Optional
import httpx
import numpy as np


class AlphaVantageClient:
    """Client for fetching market data from Alpha Vantage."""

    def __init__(self, api_key: str, cache_duration_hours: int = 1):
        """
        Args:
            api_key: Alpha Vantage API key
            cache_duration_hours: How long to cache SPY data before refreshing
        """
        self.api_key = api_key
        self.cache_duration_hours = cache_duration_hours
        self.base_url = "https://www.alphavantage.co/query"

        # Cache
        self._spy_returns = {}  # {date_str: daily_return}
        self._spy_volatility = {}  # {date_str: rolling_60d_std}
        self._cache_timestamp: Optional[datetime] = None

    def get_market_data(self, event_date: datetime) -> Tuple[float, float]:
        """
        Get market return and volatility for a given date.

        Args:
            event_date: The datetime of the earnings event

        Returns:
            (r_m, r_m_volatility) tuple
            - r_m: Market return on event date (0.0 if unavailable)
            - r_m_volatility: 60-day rolling std of returns
        """
        # Refresh cache if stale
        if self._is_cache_stale():
            try:
                self._fetch_spy_data()
            except Exception as e:
                print(f"[WARN] Alpha Vantage fetch failed: {e}")
                # Use fallback values
                return 0.0, 0.01

        # Find closest trading day
        r_m, r_m_volatility = self._get_data_for_date(event_date)
        return r_m, r_m_volatility

    def _is_cache_stale(self) -> bool:
        """Check if cache needs refresh."""
        if self._cache_timestamp is None:
            return True

        age = datetime.now() - self._cache_timestamp
        return age.total_seconds() > (self.cache_duration_hours * 3600)

    def _fetch_spy_data(self):
        """Fetch SPY daily data from Alpha Vantage and populate cache."""
        print("[INFO] Fetching SPY data from Alpha Vantage...")

        if not self.api_key:
            raise ValueError("Alpha Vantage API key not provided")

        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': 'SPY',
            'outputsize': 'full',  # Get ~20 years of data
            'apikey': self.api_key
        }

        response = httpx.get(self.base_url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        if 'Time Series (Daily)' not in data:
            error_msg = data.get('Note') or data.get('Error Message') or 'Unknown error'
            raise ValueError(f"Alpha Vantage API error: {error_msg}")

        time_series = data['Time Series (Daily)']

        # Parse and sort by date
        dates_sorted = sorted(time_series.keys(), reverse=True)  # Most recent first

        # Calculate daily returns
        returns = {}
        for i in range(len(dates_sorted) - 1):
            current_date = dates_sorted[i]
            prev_date = dates_sorted[i + 1]

            current_close = float(time_series[current_date]['4. close'])
            prev_close = float(time_series[prev_date]['4. close'])

            daily_return = (current_close - prev_close) / prev_close
            returns[current_date] = daily_return

        # Calculate rolling 60-day volatility
        volatility = {}
        for i, date in enumerate(dates_sorted):
            # Get last 60 days of returns
            window_dates = dates_sorted[i:min(i + 60, len(dates_sorted))]
            window_returns = [returns.get(d, 0.0) for d in window_dates if d in returns]

            if len(window_returns) >= 10:
                vol = float(np.std(window_returns))
            else:
                vol = 0.01  # Fallback

            volatility[date] = vol

        self._spy_returns = returns
        self._spy_volatility = volatility
        self._cache_timestamp = datetime.now()

        print(f"[INFO] Cached {len(returns)} days of SPY data")

    def _get_data_for_date(self, event_date: datetime) -> Tuple[float, float]:
        """Get r_m and volatility for a specific date."""
        if not self._spy_returns:
            # No data available
            return 0.0, 0.01

        # Try to find exact date or closest previous trading day (within 5 days)
        event_date_str = event_date.strftime('%Y-%m-%d')

        # Try exact match
        if event_date_str in self._spy_returns:
            r_m = self._spy_returns[event_date_str]
            vol = self._spy_volatility.get(event_date_str, 0.01)
            return float(r_m), float(vol)

        # Look back up to 5 days for closest trading day
        for i in range(1, 6):
            check_date = (event_date - timedelta(days=i)).strftime('%Y-%m-%d')
            if check_date in self._spy_returns:
                r_m = self._spy_returns[check_date]
                vol = self._spy_volatility.get(check_date, 0.01)
                return float(r_m), float(vol)

        # No data found, return fallback
        return 0.0, 0.01

    def get_cache_stats(self) -> dict:
        """Get cache statistics for debugging."""
        return {
            'cached_days': len(self._spy_returns),
            'cache_age_hours': (
                (datetime.now() - self._cache_timestamp).total_seconds() / 3600
                if self._cache_timestamp else None
            ),
            'cache_stale': self._is_cache_stale()
        }
