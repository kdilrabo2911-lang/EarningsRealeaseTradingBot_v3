"""
Test PeerTracker
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_v3.peer_tracker import PeerTracker


def test_peer_tracker():
    """Test PeerTracker basic functionality."""
    print("="*80)
    print("TEST: PeerTracker")
    print("="*80)

    # Create simple industry map
    ticker_industry_map = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'JPM': 'Financial Services',
        'GS': 'Financial Services'
    }

    # Create tracker
    tracker = PeerTracker(ticker_industry_map)

    print("\n1. Testing update and retrieval...")

    # Add some Q3 data
    tracker.update('AAPL', '2026_Q3', car1=0.05, surprise=0.01)
    tracker.update('MSFT', '2026_Q3', car1=0.03, surprise=0.02)
    tracker.update('GOOGL', '2026_Q3', car1=0.04, surprise=0.015)

    tracker.update('JPM', '2026_Q3', car1=-0.02, surprise=-0.01)
    tracker.update('GS', '2026_Q3', car1=-0.01, surprise=-0.005)

    # Get Q4 peer stats (should use Q3 data)
    tech_car1, tech_surprise = tracker.get_peer_stats_lag1('AAPL', '2026_Q4')

    print(f"\nTechnology industry Q3 stats (for Q4 lag):")
    print(f"  peer_car1_lag1:     {tech_car1:.6f}")
    print(f"  peer_surprise_lag1: {tech_surprise:.6f}")

    # Verify calculations
    expected_tech_car1 = (0.05 + 0.03 + 0.04) / 3  # 0.04
    expected_tech_surprise = (0.01 + 0.02 + 0.015) / 3  # 0.015

    assert abs(tech_car1 - expected_tech_car1) < 0.0001, f"Expected {expected_tech_car1}, got {tech_car1}"
    assert abs(tech_surprise - expected_tech_surprise) < 0.0001

    print(f"  ✅ Calculations correct!")

    # Test financial services
    fin_car1, fin_surprise = tracker.get_peer_stats_lag1('JPM', '2026_Q4')
    expected_fin_car1 = (-0.02 + -0.01) / 2
    assert abs(fin_car1 - expected_fin_car1) < 0.0001

    print(f"\nFinancial Services industry Q3 stats:")
    print(f"  peer_car1_lag1:     {fin_car1:.6f}")
    print(f"  ✅ Correct!")

    # Test quarter transition
    print("\n2. Testing quarter transitions...")
    test_cases = [
        ('2026_Q4', '2026_Q3'),
        ('2026_Q1', '2025_Q4'),
        ('2027_Q1', '2026_Q4'),
        ('2026_Q2', '2026_Q1'),
    ]

    for current, expected_prev in test_cases:
        result = tracker._get_previous_quarter(current)
        assert result == expected_prev, f"{current} -> expected {expected_prev}, got {result}"
        print(f"  {current} -> {result} ✅")

    # Test stats summary
    print("\n3. Testing stats summary...")
    summary = tracker.get_stats_summary()
    print(f"\n  Quarters tracked: {summary['quarters_tracked']}")
    print(f"  Total observations: {summary['total_observations']}")

    assert summary['quarters_tracked'] == 1, "Should have 1 quarter"
    assert summary['total_observations'] == 5, "Should have 5 observations"

    print("\n" + "="*80)
    print("✅ PeerTracker tests PASSED")
    print("="*80)


def test_peer_tracker_with_files():
    """Test loading from actual model artifacts."""
    print("\n" + "="*80)
    print("TEST: PeerTracker from Files")
    print("="*80)

    artifacts_dir = Path(__file__).parent.parent.parent / "model_artifacts"

    if not artifacts_dir.exists():
        print("⚠️  Skipping file test - model_artifacts not found")
        return

    industry_map_path = artifacts_dir / "ticker_industry_map.json"
    historical_stats_path = artifacts_dir / "historical_industry_stats.json"

    if not industry_map_path.exists() or not historical_stats_path.exists():
        print("⚠️  Skipping file test - required files not found")
        return

    # Load tracker from files
    tracker = PeerTracker.from_json_files(industry_map_path, historical_stats_path)

    summary = tracker.get_stats_summary()
    print(f"\nLoaded from files:")
    print(f"  Quarters tracked: {summary['quarters_tracked']}")
    print(f"  Total observations: {summary['total_observations']}")

    assert summary['quarters_tracked'] > 0, "Should have historical quarters"

    # Test getting peer stats for a known ticker
    test_ticker = 'AAPL' if 'AAPL' in tracker.ticker_industry_map else list(tracker.ticker_industry_map.keys())[0]
    peer_car1, peer_surprise = tracker.get_peer_stats_lag1(test_ticker, '2027_Q1')

    print(f"\nPeer stats for {test_ticker} in 2027_Q1 (using 2026_Q4 data):")
    print(f"  peer_car1_lag1:     {peer_car1:.6f}")
    print(f"  peer_surprise_lag1: {peer_surprise:.6f}")

    print("\n✅ File loading tests PASSED")


if __name__ == "__main__":
    test_peer_tracker()
    test_peer_tracker_with_files()
