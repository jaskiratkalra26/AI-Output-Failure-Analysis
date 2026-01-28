from typing import List, Dict
import logging
import re
from verification.nli_model import NLIModel
from config.config import CONFIG

logger = logging.getLogger(__name__)

# Load config
verification_config = CONFIG.get("verification", {})
model_config = CONFIG.get("model_config", {})
retrieval_config = CONFIG.get("retrieval", {})

# Global instance to load only once
_nli_model = None

def get_nli_model():
    global _nli_model
    if _nli_model is None:
        try:
            model_name = model_config.get("nli_model_name", "roberta-large-mnli")
            max_len = verification_config.get("nli_max_length", 512)
            _nli_model = NLIModel(model_name=model_name, max_length=max_len)
        except Exception as e:
            logger.error(f"Could not initialize NLI model: {e}")
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

    keyword_min_len = retrieval_config.get("keyword_min_length", 3)
    overlap_threshold = verification_config.get("keyword_overlap_threshold", 0.7)
    supportive_conf = verification_config.get("supportive_entailment_confidence", 0.9)

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
                
                # FIX 2: Supportive entailment override
                if label == "NEUTRAL":
                    # Check overlap: if significant claim keywords are in evidence
                    folder_keywords = [w.lower() for w in re.findall(r'\w+', claim_text) if len(w) > keyword_min_len]
                    if folder_keywords:
                        found_count = sum(1 for k in folder_keywords if k in ev_text.lower())
                        if (found_count / len(folder_keywords)) >= overlap_threshold:
                            label = "ENTAILMENT"
                            confidence = max(confidence, supportive_conf)

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
