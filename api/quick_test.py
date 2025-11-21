"""
Quick API test - waits for server to be ready before running tests
"""

import requests
import time
import sys

BASE_URL = "http://localhost:8000"

print("⏳ Waiting for API server to start...")
print()

# Wait up to 30 seconds for server to start
for i in range(30):
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API server is ready!")
            print()
            break
    except requests.exceptions.RequestException:
        pass
    
    sys.stdout.write(f"\r{'.' * (i + 1)}")
    sys.stdout.flush()
    time.sleep(1)
else:
    print("\n❌ Server did not start within 30 seconds")
    print("Please check the server logs")
    sys.exit(1)

print("="  * 80)
print("  QUICK API TEST")
print("=" * 80)

# Test 1: Health Check
print("\n1️⃣  Testing health endpoint...")
response = requests.get(f"{BASE_URL}/health")
print(f"   Status: {response.status_code}")
data = response.json()
print(f"   Models loaded: {data.get('models_loaded', False)}")

# Test 2: Root endpoint
print("\n2️⃣  Testing root endpoint...")
response = requests.get(BASE_URL)
print(f"   Status: {response.status_code}")
data = response.json()
print(f"   API: {data.get('name', 'N/A')}")

# Test 3: Categories
print("\n3️⃣  Testing categories endpoint...")
response = requests.get(f"{BASE_URL}/api/categories")
print(f"   Status: {response.status_code}")
data = response.json()
print(f"   Categories available: {data.get('count', 0)}")

# Test 4: Compliance Check
print("\n4️⃣  Testing compliance check...")
test_contract = """
SUPPLY AGREEMENT

This Agreement is made between ABC Corp (Buyer) and XYZ Inc (Supplier).

1. DELIVERY TERMS
Supplier shall deliver the goods within ten (10) business days of receiving the purchase order.

2. PAYMENT TERMS  
Buyer shall pay the invoice amount within thirty (30) days of delivery.

3. PURCHASE QUANTITY
Buyer agrees to purchase minimum 1000 units per order.
"""

test_query = "Supplier delivered goods 15 days after receiving order"

response = requests.post(
    f"{BASE_URL}/api/compliance/check",
    json={
        "contract_text": test_contract,
        "query": test_query
    }
)

print(f"   Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"   Query: {test_query}")
    print(f"   Result: {data.get('status', 'N/A').upper()}")
    print(f"   Confidence: {data.get('confidence', 0):.0%}")
    
    if data.get('breach_details'):
        breach = data['breach_details']
        print(f"   Required: {breach.get('required', 'N/A')}")
        print(f"   Actual: {breach.get('actual', 'N/A')}")

print("\n" + "=" * 80)
print("  ✅ QUICK TEST COMPLETE!")
print("=" * 80)
print(f"\n📖 View full API docs at: {BASE_URL}/docs")
print(f"📖 View alternative docs at: {BASE_URL}/redoc")
print()
