"""Quick diagnostic script to check if API can be imported."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("  API DIAGNOSTICS")
print("=" * 80)
print(f"\n📂 Project root: {project_root}")
print(f"🐍 Python version: {sys.version}")
print(f"📍 Python path:")
for p in sys.path[:5]:
    print(f"   - {p}")

print("\n" + "=" * 80)
print("  Checking imports...")
print("=" * 80)

try:
    print("\n1. Importing FastAPI...")
    import fastapi
    print(f"   ✅ FastAPI {fastapi.__version__}")
except ImportError as e:
    print(f"   ❌ FastAPI not found: {e}")
    sys.exit(1)

try:
    print("\n2. Importing Pydantic...")
    import pydantic
    print(f"   ✅ Pydantic {pydantic.VERSION}")
except ImportError as e:
    print(f"   ❌ Pydantic not found: {e}")
    sys.exit(1)

try:
    print("\n3. Importing compliance checker...")
    from cuad_integration.simple_compliance_checker import SimpleComplianceChecker
    print("   ✅ SimpleComplianceChecker imported")
except ImportError as e:
    print(f"   ❌ SimpleComplianceChecker failed: {e}")
    sys.exit(1)

try:
    print("\n4. Importing clause extractor...")
    from cuad_integration.clause_extractor import CUADClauseExtractor
    print("   ✅ CUADClauseExtractor imported")
except ImportError as e:
    print(f"   ❌ CUADClauseExtractor failed: {e}")
    sys.exit(1)

try:
    print("\n5. Importing API main module...")
    from api.main import app
    print("   ✅ FastAPI app created successfully")
    print(f"   📝 Title: {app.title}")
    print(f"   📝 Version: {app.version}")
    print(f"   📝 Endpoints: {len(app.routes)}")
except Exception as e:
    print(f"   ❌ API main import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("  ✅ ALL DIAGNOSTICS PASSED!")
print("=" * 80)
print("\nYou can now start the server with:")
print("  /opt/miniconda3/bin/conda run -n major uvicorn api.main:app --reload")
print("\n or use the startup script:")
print("  ./api/start_server.sh")
print()
