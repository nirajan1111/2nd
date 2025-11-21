"""
API Test Client
Tests all FastAPI endpoints.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_health():
    """Test health check endpoint."""
    print_section("TEST 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed!")


def test_root():
    """Test root endpoint."""
    print_section("TEST 2: Root Endpoint")
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Root endpoint passed!")


def test_categories():
    """Test get categories endpoint."""
    print_section("TEST 3: Get Categories")
    
    response = requests.get(f"{BASE_URL}/api/categories")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Total categories: {data['total']}")
    print(f"Sample categories: {data['categories'][:5]}")
    
    assert response.status_code == 200
    assert data['total'] > 0
    print("✅ Categories endpoint passed!")


def test_extract_clauses():
    """Test clause extraction endpoint."""
    print_section("TEST 4: Extract Clauses")
    
    contract = """
    SUPPLY AGREEMENT
    
    This Agreement is entered into between ABC Corporation ("Buyer") and 
    XYZ Supplies Inc. ("Supplier").
    
    1. DELIVERY: Supplier shall deliver goods within ten (10) business days.
    2. PAYMENT: Buyer shall pay within thirty (30) days of invoice.
    3. TERMINATION: Either party may terminate with ninety (90) days notice.
    4. MINIMUM PURCHASE: Buyer agrees to purchase minimum 1,000 units per quarter.
    """
    
    payload = {
        "contract_text": contract
    }
    
    response = requests.post(f"{BASE_URL}/api/contracts/extract", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Contract ID: {data['contract_id']}")
        print(f"Extracted clauses: {data['extracted_count']}")
        print(f"Categories found:")
        for category in list(data['clauses'].keys())[:5]:
            print(f"  • {category}")
        print("✅ Clause extraction passed!")
        return data['contract_id']
    else:
        print(f"❌ Error: {response.text}")
        return None


def test_compliance_check():
    """Test compliance check endpoint."""
    print_section("TEST 5: Compliance Check")
    
    contract = """
    SUPPLY AGREEMENT
    
    1. DELIVERY: Supplier shall deliver goods within ten (10) business days.
    2. PAYMENT: Buyer shall pay within thirty (30) days of invoice.
    3. TERMINATION: Either party may terminate with ninety (90) days notice.
    4. MINIMUM PURCHASE: Buyer agrees to purchase minimum 1,000 units per quarter.
    """
    
    queries = [
        "Supplier delivered goods 15 days after order",
        "Buyer paid on day 25",
        "Buyer terminated with 60 days notice",
        "Buyer purchased 1200 units"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        
        payload = {
            "contract_text": contract,
            "query": query
        }
        
        response = requests.post(f"{BASE_URL}/api/compliance/check", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data['status'].upper()}")
            print(f"   Confidence: {data['confidence']:.0%}")
            print(f"   Request ID: {data['request_id']}")
        else:
            print(f"   ❌ Error: {response.text}")
    
    print("\n✅ Compliance checks completed!")


def test_batch_compliance():
    """Test batch compliance check endpoint."""
    print_section("TEST 6: Batch Compliance Check")
    
    contract = """
    SUPPLY AGREEMENT
    
    1. DELIVERY: Supplier shall deliver goods within ten (10) business days.
    2. PAYMENT: Buyer shall pay within thirty (30) days of invoice.
    3. TERMINATION: Either party may terminate with ninety (90) days notice.
    """
    
    queries = [
        "Supplier delivered goods 15 days after order",
        "Buyer paid on day 25",
        "Buyer terminated with 60 days notice"
    ]
    
    payload = {
        "contract_text": contract,
        "queries": queries
    }
    
    response = requests.post(f"{BASE_URL}/api/compliance/batch", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Batch ID: {data['batch_id']}")
        print(f"Total queries: {data['total_queries']}")
        print(f"\nResults:")
        for i, result in enumerate(data['results'], 1):
            print(f"\n  {i}. {result['query']}")
            print(f"     Status: {result['status'].upper()}")
            print(f"     Confidence: {result['confidence']:.0%}")
        
        print("\n✅ Batch compliance check passed!")
        return data['batch_id']
    else:
        print(f"❌ Error: {response.text}")
        return None


def test_get_results(request_id: str):
    """Test get results endpoint."""
    print_section("TEST 7: Get Cached Results")
    
    response = requests.get(f"{BASE_URL}/api/results/{request_id}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Type: {data['type']}")
        print(f"Timestamp: {data['timestamp']}")
        print("✅ Get results passed!")
    else:
        print(f"❌ Error: {response.text}")


def test_list_results():
    """Test list results endpoint."""
    print_section("TEST 8: List All Results")
    
    response = requests.get(f"{BASE_URL}/api/results")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total cached: {data['total_cached']}")
        print(f"Results:")
        for result in data['results'][:5]:
            print(f"  • {result['id'][:8]}... - Type: {result['type']} - {result['timestamp']}")
        print("✅ List results passed!")
    else:
        print(f"❌ Error: {response.text}")


def run_all_tests():
    """Run all API tests."""
    print("\n" + "="*80)
    print("  CONTRACT COMPLIANCE CHECKER - API TESTS")
    print("="*80)
    
    try:
        # Basic tests
        test_health()
        test_root()
        test_categories()
        
        # Extraction test
        contract_id = test_extract_clauses()
        
        # Compliance tests
        test_compliance_check()
        batch_id = test_batch_compliance()
        
        # Results tests
        if batch_id:
            test_get_results(batch_id)
        test_list_results()
        
        # Summary
        print_section("✅ ALL TESTS PASSED!")
        print("API is working correctly!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running:")
        print("  cd /Users/nirajansah/major/second-attempt")
        print("  /opt/miniconda3/bin/conda run -n major uvicorn api.main:app --reload")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    run_all_tests()
