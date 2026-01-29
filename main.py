import logging
import os
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import modules from the phases
from claim_extraction.extractor import ClaimExtractor
from evidence_retrieval.retriever import retrieve_evidence
from verification.claim_verification import verify_claims, get_nli_model
from aggregation.aggregate import aggregate_results

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HallucinationDetectorAPI")

# Initialize FastAPI app
app = FastAPI(
    title="Hallucination Detection System",
    description="Orchestrator for Hallucination Detection Phases",
    version="1.0.0"
)

# Add CORS Middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Data Models ---

class AnalysisRequest(BaseModel):
    llm_output: str

class EvidenceItem(BaseModel):
    text: str
    source_url: str

class VerifiedClaim(BaseModel):
    claim: str
    verdict: str
    evidence: List[EvidenceItem]

class AnalysisResponse(BaseModel):
    hallucinated: bool
    confidence: float
    claims: List[VerifiedClaim]

# --- Lifecycle Events ---

@app.on_event("startup")
async def startup_event():
    """
    Initialize resources on startup.
    """
    logger.info("Initializing API...")
    # Load NLI model once at startup as required
    try:
        get_nli_model()
        logger.info("NLI Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load NLI model: {e}")

# --- API Endpoints ---

@app.post("/check_hallucination", response_model=AnalysisResponse)
async def check_hallucination(request: AnalysisRequest):
    """
    Main endpoint to check for hallucinations in LLM output.
    """
    llm_text = request.llm_output
    if not llm_text.strip():
        # Handle empty input
        return AnalysisResponse(
            hallucinated=False,
            confidence=0.0,
            claims=[]
        )

    logger.info("Starting hallucination check pipeline.")

    # --- Phase 1: Claim Extraction ---
    logger.info("Phase 1: Extracting claims...")
    try:
        extractor = ClaimExtractor()
        extracted_claims = extractor.extract_claims_from_text(llm_text)
        
        if not extracted_claims or len(extracted_claims) == 0:
            logger.info("No claims extracted.")
            return AnalysisResponse(
                hallucinated=False,
                confidence=0.0,
                claims=[]
            )
            
        logger.info(f"Extracted {len(extracted_claims)} claims.")
        
    except Exception as e:
        logger.error(f"Error in Claim Extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Claim Extraction Phase failed: {str(e)}")

    # --- Phase 2 & 3: Evidence Retrieval (Wiki + SerpAPI Fallback) ---
    logger.info("Phase 2 & 3: Retrieving evidence...")
    try:
        # retrieve_evidence handles both Wikipedia and SerpAPI fallback
        retrieved_results_list = retrieve_evidence(extracted_claims)
        
        # Create a map for easy lookup
        retrieved_map = {item["claim"]: item for item in retrieved_results_list}
        
        # Prepare list for verification, ensuring all claims are present
        # If evidence retrieval fails (or finds nothing), we must keep the claim to mark as UNSUPPORTED later
        claims_for_verification = []
        for claim in extracted_claims:
            if claim in retrieved_map:
                claims_for_verification.append(retrieved_map[claim])
            else:
                # Missing evidence -> Empty evidence list
                claims_for_verification.append({
                    "claim": claim,
                    "evidence": [],
                    "source": "none"
                })
                
    except Exception as e:
        logger.error(f"Error in Evidence Retrieval: {e}")
        # If retrieval fails completely, treat all claims as having no evidence
        claims_for_verification = [{
            "claim": claim, 
            "evidence": [], 
            "source": "none"
        } for claim in extracted_claims]

    # --- Phase 4: Claim Verification ---
    logger.info("Phase 4: Verifying claims...")
    try:
        # verify_claims will return a list of dicts with 'verifications'
        verified_results = verify_claims(claims_for_verification)
    except Exception as e:
        logger.error(f"Error in Claim Verification: {e}")
        # If verification fails, skip verification details
        verified_results = [{
            "claim": item["claim"],
            "verifications": []
        } for item in claims_for_verification]

    # --- Phase 5: Aggregation ---
    logger.info("Phase 5: Aggregating results...")
    try:
        aggregation_output = aggregate_results(verified_results)
    except Exception as e:
         logger.error(f"Error in Aggregation: {e}")
         raise HTTPException(status_code=500, detail=f"Aggregation Phase failed: {str(e)}")

    # --- Final Result Formatting ---
    # We need to map NEUTRAL to UNSUPPORTED if there was no evidence.
    # The aggregator defaults to NEUTRAL if no contradictions/entailments are found.
    # But for our API requirement: "If evidence retrieval fails → mark claim as UNSUPPORTED"
    
    final_claims = []
    
    for claim_res in aggregation_output.get("claims", []):
        verdict = claim_res.get("verdict")
        evidence = claim_res.get("evidence", [])
        
        # Post-processing logic for UNSUPPORTED
        if not evidence or len(evidence) == 0:
            verdict = "UNSUPPORTED"
            
        final_claims.append({
            "claim": claim_res.get("claim"),
            "verdict": verdict,
            "evidence": evidence
        })
    
    # Re-calculate overall hallucination status if needed? 
    # The current aggregation logic in Phase 5 bases decision on Contradictions and Neutral ratio.
    # If we change NEUTRAL to UNSUPPORTED, does it change the overall halluciation status?
    # Phase 5 logic: 
    #  - If any CONTRADICTION -> Hallucinated=True
    #  - Else if majority NEUTRAL -> Hallucinated=True (Unverifiable claims represent risk)
    #  - Else -> False
    #
    # If we output UNSUPPORTED, semantically it's similar to NEUTRAL (Unverifiable). 
    # So the Phase 5 decision (Hallucinated=True for majority Neutral) stands correct for UNSUPPORTED too.
    # We don't need to change the top-level 'hallucinated' boolean, just the claim label for clarity.
    
    return AnalysisResponse(
        hallucinated=aggregation_output.get("hallucinated", False),
        confidence=aggregation_output.get("confidence", 0.0),
        claims=final_claims
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
