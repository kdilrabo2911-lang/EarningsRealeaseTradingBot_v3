"""
Test Alpha Vantage Client
"""
import sys
sys.path.append('..')

from datetime import datetime
from model_v3.alpha_vantage_client import AlphaVantageClient
import os

def test_alpha_vantage_client():
    """Test that AlphaVantageClient fetches real data."""
    print("="*80)
    print("TEST: AlphaVantageClient")
    print("="*80)

    # Get API key from environment
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'D3VPNUZTLFCZVH45')

    # Create client
    client = AlphaVantageClient(api_key=api_key, cache_duration_hours=1)

    # Test with a known date
    test_date = datetime(2026, 1, 27, 21, 0, 0)

    print(f"\nTesting with date: {test_date}")

    # Fetch data
    r_m, r_m_volatility = client.get_market_data(test_date)

    print(f"\nResults:")
    print(f"  r_m (market return):      {r_m:.6f}")
    print(f"  r_m_volatility (60d std): {r_m_volatility:.6f}")

    # Get cache stats
    stats = client.get_cache_stats()
    print(f"\nCache stats:")
    print(f"  Cached days: {stats['cached_days']}")
    print(f"  Cache age (hours): {stats['cache_age_hours']:.2f}" if stats['cache_age_hours'] else "  Cache age: Not cached")

    # Assertions
    assert isinstance(r_m, float), "r_m should be float"
    assert isinstance(r_m_volatility, float), "r_m_volatility should be float"
    assert stats['cached_days'] > 0, "Should have cached data"

    # Test cache hit (should be fast)
    print(f"\n Testing cache hit...")
    r_m2, vol2 = client.get_market_data(test_date)
    assert r_m2 == r_m, "Cached values should match"

    print("\n" + "="*80)
    print("✅ AlphaVantageClient tests PASSED")
    print("="*80)


if __name__ == "__main__":
    test_alpha_vantage_client()
