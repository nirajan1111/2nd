"""
FastAPI Backend for Contract Compliance Checker
Main application file with all API endpoints.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import uuid
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cuad_integration.simple_compliance_checker import SimpleComplianceChecker, ComplianceStatus
from cuad_integration.clause_extractor import CUADClauseExtractor

# Initialize FastAPI app
app = FastAPI(
    title="Contract Compliance Checker API",
    description="API for checking contract compliance using CUAD extraction and rule-based reasoning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
compliance_checker: Optional[SimpleComplianceChecker] = None
clause_extractor: Optional[CUADClauseExtractor] = None
results_cache: Dict[str, Dict] = {}

# Pydantic models for request/response
class ContractUploadResponse(BaseModel):
    contract_id: str
    message: str
    contract_length: int
    timestamp: str

class ClauseExtractionRequest(BaseModel):
    contract_text: str = Field(..., description="Full contract text")
    categories: Optional[List[str]] = Field(None, description="Specific categories to extract")

class ClauseExtractionResponse(BaseModel):
    contract_id: str
    clauses: Dict[str, str]
    extracted_count: int
    timestamp: str

class ComplianceCheckRequest(BaseModel):
    contract_text: str = Field(..., description="Full contract text")
    query: str = Field(..., description="Compliance question (e.g., 'Supplier delivered 15 days late')")

class ComplianceCheckResponse(BaseModel):
    request_id: str
    status: str
    confidence: float
    explanation: str
    relevant_clauses: List[Dict]
    parsed_action: Optional[str]
    breach_details: Optional[Dict]
    timestamp: str

class BatchComplianceRequest(BaseModel):
    contract_text: str = Field(..., description="Full contract text")
    queries: List[str] = Field(..., description="List of compliance questions")

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    models_loaded: bool


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global compliance_checker, clause_extractor
    
    print("🚀 Starting Contract Compliance Checker API...")
    print("📦 Loading models...")
    
    try:
        # Initialize compliance checker (includes clause extractor)
        compliance_checker = SimpleComplianceChecker("Rakib/roberta-base-on-cuad")
        clause_extractor = compliance_checker.clause_extractor
        
        print("✅ Models loaded successfully!")
        print("🌐 API is ready to accept requests")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Contract Compliance Checker API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /api/contracts/extract": "Extract clauses from contract",
            "POST /api/compliance/check": "Check compliance",
            "GET /api/results/{request_id}": "Get cached results",
            "GET /health": "Health check"
        }
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    models_loaded = compliance_checker is not None and clause_extractor is not None
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        message="API is running" if models_loaded else "Models not loaded",
        version="1.0.0",
        models_loaded=models_loaded
    )


# Extract clauses endpoint
@app.post("/api/contracts/extract", response_model=ClauseExtractionResponse, tags=["Contracts"])
async def extract_clauses(request: ClauseExtractionRequest):
    """
    Extract clauses from contract text.
    
    - **contract_text**: Full contract text
    - **categories**: Optional list of specific categories to extract
    
    Returns extracted clauses for all or specified categories.
    """
    if not clause_extractor:
        raise HTTPException(status_code=503, detail="Clause extractor not initialized")
    
    try:
        # Generate contract ID
        contract_id = str(uuid.uuid4())
        
        # Extract clauses
        if request.categories:
            # Extract specific categories
            clauses = {}
            all_clauses = clause_extractor.extract_and_format(request.contract_text)
            for category in request.categories:
                if category in all_clauses:
                    clauses[category] = all_clauses[category]
        else:
            # Extract all clauses
            clauses = clause_extractor.extract_and_format(request.contract_text)
        
        # Filter out "Not found" entries
        found_clauses = {k: v for k, v in clauses.items() if v != "Not found"}
        
        # Cache result
        results_cache[contract_id] = {
            "type": "extraction",
            "contract_text": request.contract_text,
            "clauses": found_clauses,
            "timestamp": datetime.now().isoformat()
        }
        
        return ClauseExtractionResponse(
            contract_id=contract_id,
            clauses=found_clauses,
            extracted_count=len(found_clauses),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting clauses: {str(e)}")


# Compliance check endpoint
@app.post("/api/compliance/check", response_model=ComplianceCheckResponse, tags=["Compliance"])
async def check_compliance(request: ComplianceCheckRequest):
    """
    Check if an action complies with contract terms.
    
    - **contract_text**: Full contract text
    - **query**: Compliance question (e.g., "Supplier delivered 15 days late")
    
    Returns compliance status, confidence, and detailed explanation.
    """
    if not compliance_checker:
        raise HTTPException(status_code=503, detail="Compliance checker not initialized")
    
    try:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Check compliance
        result = compliance_checker.check_compliance(
            contract_text=request.contract_text,
            user_query=request.query
        )
        
        # Cache result
        results_cache[request_id] = {
            "type": "compliance",
            "contract_text": request.contract_text,
            "query": request.query,
            "result": result.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        return ComplianceCheckResponse(
            request_id=request_id,
            status=result.status.value,
            confidence=result.confidence,
            explanation=result.explanation,
            relevant_clauses=result.relevant_clauses,
            parsed_action=str(result.parsed_action) if result.parsed_action else None,
            breach_details=result.breach_details,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking compliance: {str(e)}")


# Get cached results endpoint
@app.get("/api/results/{request_id}", tags=["Results"])
async def get_results(request_id: str):
    """
    Retrieve cached results by request ID.
    
    - **request_id**: The ID returned from a previous request
    
    Returns cached compliance or extraction results.
    """
    if request_id not in results_cache:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    return results_cache[request_id]


# List all cached results
@app.get("/api/results", tags=["Results"])
async def list_results():
    """
    List all cached results.
    
    Returns a summary of all cached compliance checks and extractions.
    """
    return {
        "total_cached": len(results_cache),
        "results": [
            {
                "id": request_id,
                "type": data["type"],
                "timestamp": data["timestamp"]
            }
            for request_id, data in results_cache.items()
        ]
    }


# Clear cache endpoint
@app.delete("/api/results", tags=["Results"])
async def clear_cache():
    """Clear all cached results."""
    global results_cache
    count = len(results_cache)
    results_cache = {}
    return {"message": f"Cleared {count} cached results"}


# Upload contract file endpoint
@app.post("/api/contracts/upload", response_model=ContractUploadResponse, tags=["Contracts"])
async def upload_contract(file: UploadFile = File(...)):
    """
    Upload a contract file (TXT or PDF).
    
    - **file**: Contract file (TXT or PDF format)
    
    Returns contract ID and metadata.
    """
    # Check file extension
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Only TXT and PDF files are supported")
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse content based on file type
        if file.filename.endswith('.txt'):
            contract_text = content.decode('utf-8')
        elif file.filename.endswith('.pdf'):
            # PDF parsing would go here (requires PyPDF2 or pdfplumber)
            raise HTTPException(status_code=501, detail="PDF support not yet implemented")
        
        # Generate contract ID
        contract_id = str(uuid.uuid4())
        
        # Cache contract
        results_cache[contract_id] = {
            "type": "upload",
            "filename": file.filename,
            "contract_text": contract_text,
            "timestamp": datetime.now().isoformat()
        }
        
        return ContractUploadResponse(
            contract_id=contract_id,
            message=f"Contract '{file.filename}' uploaded successfully",
            contract_length=len(contract_text),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading contract: {str(e)}")


# Batch compliance check endpoint
@app.post("/api/compliance/batch", tags=["Compliance"])
async def batch_compliance_check(request: BatchComplianceRequest):
    """
    Check multiple compliance queries against the same contract.
    
    - **contract_text**: Full contract text
    - **queries**: List of compliance questions
    
    Returns list of compliance results.
    """
    if not compliance_checker:
        raise HTTPException(status_code=503, detail="Compliance checker not initialized")
    
    try:
        results = []
        
        for query in request.queries:
            result = compliance_checker.check_compliance(
                contract_text=request.contract_text,
                user_query=query
            )
            
            results.append({
                "query": query,
                "status": result.status.value,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "breach_details": result.breach_details
            })
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Cache results
        results_cache[batch_id] = {
            "type": "batch",
            "contract_text": request.contract_text,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "batch_id": batch_id,
            "total_queries": len(request.queries),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch check: {str(e)}")


# Get available categories endpoint
@app.get("/api/categories", tags=["General"])
async def get_categories():
    """
    Get list of available CUAD categories for extraction.
    
    Returns all supported clause categories.
    """
    if not clause_extractor:
        raise HTTPException(status_code=503, detail="Clause extractor not initialized")
    
    return {
        "categories": list(clause_extractor.EXTRACTION_CATEGORIES.keys()),
        "total": len(clause_extractor.EXTRACTION_CATEGORIES)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
