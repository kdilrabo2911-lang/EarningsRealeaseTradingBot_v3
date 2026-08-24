# V3 Model Implementation Plan

## Model Performance
- **Baseline (surprise only)**: R² = 0.1092
- **V2 (7 features + Gemini)**: R² = 0.1132 (+3.7%)
- **V3 (10 features + Gemini + Alpha Vantage + Peers)**: R² = 0.1586 (+45.2%)

## New Features in V3 (beyond V2's 7 features)

### Feature 8-10: Peer Context Features
8. **car1_vs_peers_lag1**: Company's lag-1 CAR1 vs industry peers' average
   - Coefficient: -0.672 (mean reversion from peers)
   - Requires: Historical CAR1 by industry/quarter

9. **surprise_vs_peers_lag1**: Surprise relative to industry average
   - Coefficient: +0.009
   - Requires: Historical surprise by industry/quarter

10. **r_m_volatility**: Market return volatility (60-day rolling std)
    - Coefficient: -0.600
    - Requires: Alpha Vantage SPY data

## predict.py Changes Required

### 1. Add Alpha Vantage Integration
```python
# New global cache
_spy_data_cache = None
_spy_cache_timestamp = None

def _fetch_spy_data():
    """Fetch SPY from Alpha Vantage, cache for 1 hour"""
    # Call Alpha Vantage TIME_SERIES_DAILY
    # Calculate daily returns
    # Calculate 60-day rolling std
    pass

def _get_market_volatility(event_date):
    """Get r_m_volatility for event date"""
    # Refresh cache if stale
    # Find closest trading day to event_date
    # Return volatility
    pass
```

### 2. Add Industry Tracking
```python
# New global state
_industry_stats = {}  # {quarter: {industry: {car1_mean, surprise_mean}}}
_ticker_industry_map = {}  # loaded from model_artifacts

def _load_model_artifacts():
    # Existing code...
    # ADD: Load ticker_industry_map.json
    # ADD: Load historical_industry_stats.json
    pass

def _update_industry_stats(ticker, quarter, car1, surprise):
    """Update rolling industry statistics after each prediction"""
    industry = _ticker_industry_map.get(ticker, 'Other')
    # Update _industry_stats with new data point
    pass

def _get_peer_stats_lag1(ticker, current_quarter):
    """Get previous quarter's peer stats for industry"""
    industry = _ticker_industry_map.get(ticker, 'Other')
    prev_quarter = _get_previous_quarter(current_quarter)
    return _industry_stats.get(prev_quarter, {}).get(industry, {})
    pass
```

### 3. Update Feature Extraction
```python
def _extract_features(event: dict, ticker: str) -> dict | None:
    # Existing v2 features 1-7...

    # NEW Feature 8-9: Peer features
    current_quarter = _extract_quarter_from_event(event)
    peer_stats = _get_peer_stats_lag1(ticker, current_quarter)

    car1_lag1 = historical_features.get(ticker, {}).get('car1', 0.0)
    peer_car1_lag1 = peer_stats.get('car1_mean', 0.0)
    peer_surprise_lag1 = peer_stats.get('surprise_mean', 0.0)

    features['car1_vs_peers_lag1'] = car1_lag1 - peer_car1_lag1
    features['surprise_vs_peers_lag1'] = surprise - peer_surprise_lag1

    # NEW Feature 10: Market volatility
    event_date = datetime.fromisoformat(event['event_datetime'])
    features['r_m_volatility'] = _get_market_volatility(event_date)

    return features
```

### 4. Update Historical Feature Saving
```python
def _save_historical_features(ticker: str, features: dict, car1_predicted: float):
    # Existing v2 code...

    # NEW: Update industry stats
    current_quarter = _extract_quarter_from_event(...)
    _update_industry_stats(ticker, current_quarter, car1_predicted, features['surprise'])
```

## State Management Challenges

### Production Constraints
- Modal containers are stateless between requests
- Need to persist industry stats somewhere

### Options:
1. **Use Modal Dicts** (persistent key-value store)
   - Pros: Built-in persistence
   - Cons: Adds latency (~50-100ms per request)

2. **In-memory only** (reset on cold start)
   - Pros: Fast
   - Cons: Lose state on container restart
   - Mitigation: Initialize from historical_industry_stats.json

### Recommended: Hybrid Approach
- Load historical stats from JSON on startup
- Track new stats in-memory during container lifetime
- Accept that state resets on cold starts (rare in production)

## Testing Plan
1. ✅ Verify backtest with Alpha Vantage data
2. ⏳ Test Alpha Vantage API in predict.py locally
3. ⏳ Test peer tracking logic locally
4. ⏳ Deploy to Modal and test with portal
5. ⏳ Monitor first 10 production predictions

## Deployment Checklist
- [ ] Alpha Vantage API key in .env
- [ ] Update model_params.json features list
- [ ] Add ticker_industry_map.json to artifacts
- [ ] Add historical_industry_stats.json to artifacts
- [ ] Update modal_app.py imports if needed
- [ ] Test locally first
- [ ] Deploy with `modal deploy`
- [ ] Test webhook with portal

## Estimated Complexity
- **Lines of code to add/modify**: ~200 lines
- **Time estimate**: 45-60 minutes
- **Risk level**: Medium (state management complexity)
