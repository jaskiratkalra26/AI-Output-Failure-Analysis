from typing import List, Dict, Any
import logging
from config.config import CONFIG

logger = logging.getLogger(__name__)

# Load config
verification_config = CONFIG.get("verification", {})

def aggregate_results(verified_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates claim-level verification results into a final hallucination decision.

    Args:
        verified_claims: List of dicts, each containing 'claim' and 'verifications'.

    Returns:
        Structured dictionary with final 'hallucinated' status, 'confidence', and detailed claim verdicts.
    """
    
    if not verified_claims:
        return {
            "hallucinated": False, 
            "confidence": 0.0,
            "claims": []
        }

    processed_claims = []
    
    label_contradiction = "CONTRADICTION"
    label_entailment = "ENTAILMENT"
    label_neutral = "NEUTRAL"
    
    contra_threshold = verification_config.get("contradiction_threshold", 0.80)

    # Counters for overall decision
    total_claims = 0
    neutral_verdict_count = 0
    has_contradiction = False
    
    # Trackers for confidence calculation
    max_contradiction_conf = 0.0
    all_entailment_confs = []

    for claim_obj in verified_claims:
        claim_text = claim_obj.get("claim", "")
        verifications = claim_obj.get("verifications", [])
        
        # 1. Determine Claim-level verdict
        # Priority: CONTRADICTION > ENTAILMENT > NEUTRAL
        
        final_verdict = label_neutral # Default
        evidences = []
        
        found_contradiction = False
        found_entailment = False
        
        # We need to collect evidence and check labels
        # Also keeping track of confidences for potential output use
        
        current_claim_contradiction_confs = []
        current_claim_entailment_confs = []

        for v in verifications:
            label = v.get("nli_label", label_neutral)
            conf = v.get("confidence", 0.0)
            text = v.get("evidence_text", "")
            url = v.get("source_url", "")
            
            # Add to evidence list for output
            if text:
                evidences.append({
                    "text": text,
                    "source_url": url
                })

            if label == label_contradiction:
                # Only count CONTRADICTION if confidence >= threshold
                if conf >= contra_threshold:
                    found_contradiction = True
                    current_claim_contradiction_confs.append(conf)
                    # Update global max contradiction confidence
                    if conf > max_contradiction_conf:
                        max_contradiction_conf = conf
            elif label == label_entailment:
                found_entailment = True
                current_claim_entailment_confs.append(conf)

        if found_contradiction:
            final_verdict = label_contradiction
        elif found_entailment:
            final_verdict = label_entailment
        else:
            final_verdict = label_neutral

        # Add to processed claims list
        processed_claims.append({
            "claim": claim_text,
            "verdict": final_verdict,
            "evidence": evidences
        })
        
        # Update global counters
        total_claims += 1
        if final_verdict == label_contradiction:
            has_contradiction = True
        elif final_verdict == label_neutral:
            neutral_verdict_count += 1
        elif final_verdict == label_entailment:
             all_entailment_confs.extend(current_claim_entailment_confs)

    # 2. Output-level hallucination decision
    is_hallucinated = False
    final_confidence = 0.0

    if has_contradiction:
        # If ANY claim verdict == CONTRADICTION -> hallucinated = True
        is_hallucinated = True
        # Confidence = max contradiction confidence
        final_confidence = max_contradiction_conf
        
    elif total_claims > 0 and (neutral_verdict_count > total_claims / 2):
        # Else if majority of claims == NEUTRAL -> hallucinated = True
        is_hallucinated = True
        # Confidence = ratio of neutral claims
        final_confidence = neutral_verdict_count / total_claims
        
    else:
        # Else -> hallucinated = False
        is_hallucinated = False
        # Confidence = average entailment confidence
        if all_entailment_confs:
            final_confidence = sum(all_entailment_confs) / len(all_entailment_confs)
        else:
            final_confidence = 0.0

    # Clamp confidence (already generally safe, but ensures bounds)
    final_confidence = max(0.0, min(1.0, final_confidence))

    return {
        "hallucinated": is_hallucinated,
        "confidence": round(final_confidence, 2),
        "claims": processed_claims
    }

