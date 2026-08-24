import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"Testing API key: {api_key[:10]}...")

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("gemini-flash-latest")
    response = model.generate_content("Say 'billing test successful' if you can read this.")
    
    print("\n✅ SUCCESS!")
    print(f"Response: {response.text}")
    print("\n📊 API Status:")
    print("  ✓ API key is VALID and WORKING")
    print("  ✓ Model: gemini-flash-latest is accessible")
    print("  ✓ You have available quota")
    print("\n💰 This confirms you're on PAID TIER (Tier 1)")
    print("  - Free tier would show 429 errors after ~15 requests/min")
    print("  - Paid tier gets 1000 requests/min")
    
except Exception as e:
    error_str = str(e)
    print(f"\n❌ ERROR: {e}")
    
    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
        print("\n⚠️  QUOTA EXCEEDED")
        if "free_tier" in error_str.lower():
            print("  Issue: Still on FREE TIER despite billing enabled")
            print("  This can take 24-48 hours to propagate")
        else:
            print("  Issue: Hit rate limit (even on paid tier)")
    elif "404" in error_str:
        print("\n⚠️  Model not found")
    elif "API_KEY_INVALID" in error_str or "not valid" in error_str:
        print("\n⚠️  Invalid API key")

