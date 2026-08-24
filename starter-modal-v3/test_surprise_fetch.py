"""
Test fetching earnings_surprise from the metrics API endpoint.
"""
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

api_key = os.getenv('EM_API_KEY')
api_base = os.getenv('EM_API_BASE_URL', 'https://api.explainingmarkets.ai/v1')

print("="*80)
print("TEST: Fetching earnings_surprise from metrics endpoint")
print("="*80)

if not api_key:
    print("❌ EM_API_KEY not set in .env")
    exit(1)

print(f"\n1. API Key: {api_key[:20]}...")
print(f"   API Base: {api_base}")

# Get a real event from the calendar
print("\n2. Fetching recent events from calendar...")
try:
    response = httpx.get(
        f"{api_base}/events",
        headers={'X-API-Key': api_key},
        timeout=10.0
    )
    response.raise_for_status()
    events = response.json()

    if not events:
        print("❌ No events found")
        exit(1)

    # Get first event
    event = events[0]
    event_id = event['event_id']
    ticker = event['focal_assets'][0]['identifier_value']

    print(f"   Found event: {event_id}")
    print(f"   Ticker: {ticker}")
    print(f"   Datetime: {event['event_datetime']}")

except Exception as e:
    print(f"❌ Failed to fetch events: {e}")
    exit(1)

# Now fetch metrics for this event
print(f"\n3. Fetching metrics for event {event_id}...")
metrics_url = f"{api_base}/events/{event_id}/metrics"
print(f"   URL: {metrics_url}")

try:
    response = httpx.get(
        metrics_url,
        headers={'X-API-Key': api_key},
        timeout=10.0
    )

    print(f"   Status code: {response.status_code}")

    if response.status_code == 200:
        metrics = response.json()
        print(f"   Response: {metrics}")

        # Check for earnings_surprise
        surprise_data = metrics.get('earnings_surprise', {})
        if surprise_data:
            print(f"\n4. Earnings surprise data:")
            print(f"   Status: {surprise_data.get('surprise_status')}")
            print(f"   Surprise: {surprise_data.get('surprise')}")

            if surprise_data.get('surprise_status') == 'ok':
                print(f"\n✅ SUCCESS! Got earnings_surprise: {surprise_data.get('surprise')}")
            else:
                print(f"\n⚠️  Earnings surprise status is '{surprise_data.get('surprise_status')}'")
        else:
            print("\n⚠️  No earnings_surprise in response")
            print(f"   Full response: {metrics}")
    else:
        print(f"❌ Non-200 status: {response.status_code}")
        print(f"   Response: {response.text}")

except Exception as e:
    print(f"❌ Failed to fetch metrics: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
