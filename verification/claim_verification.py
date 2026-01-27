from typing import List, Dict
import logging
from verification.nli_model import NLIModel

logger = logging.getLogger(__name__)

# Global instance to load only once
_nli_model = None

def get_nli_model():
    global _nli_model
    if _nli_model is None:
        try:
            _nli_model = NLIModel()
        except Exception as e:
            logger.error(f"Could not initialize NLI model: {e}")
            # Depending on requirements, we might want to re-raise or handle gracefully.
            # But the prompt says "Never crashes the pipeline". 
            # If model fails to load, we can't verify.
            pass
    return _nli_model

def verify_claims(claim_evidence_pairs: List[Dict]) -> List[Dict]:
    """
    Verifies claims against evidence using an NLI model.
    
    Args:
        claim_evidence_pairs: List of dicts with 'claim' and 'evidence' (list of dicts).
        
    Returns:
        List of dicts with verification results.
    """
    nli_model = get_nli_model()
    results = []

    # If model failed to load, return empty verifications or handle gracefully
    if nli_model is None:
        logger.error("NLI Model not available. Returning empty verifications.")
        for item in claim_evidence_pairs:
            results.append({
                "claim": item.get("claim"),
                "verifications": []
            })
        return results

    for item in claim_evidence_pairs:
        claim_text = item.get("claim")
        evidence_list = item.get("evidence", [])
        
        verifications = []

        # Assuming evidence_list is a list of dicts: {"text": "...", "source_url": "..."}
        if evidence_list:
            for ev in evidence_list:
                ev_text = ev.get("text")
                source_url = ev.get("source_url")
                
                if not ev_text:
                    continue
                    
                # Run NLI
                # Premise = Evidence, Hypothesis = Claim
                label, confidence = nli_model.predict(premise=ev_text, hypothesis=claim_text)
                
                if label:
                    verifications.append({
                        "evidence_text": ev_text,
                        "source_url": source_url,
                        "nli_label": label,
                        "confidence": round(confidence, 4)
                    })
                # If inference failed (label is None), we skip this evidence as per "Skip that evidence"
        
        results.append({
            "claim": claim_text,
            "verifications": verifications
        })
    
    return results
