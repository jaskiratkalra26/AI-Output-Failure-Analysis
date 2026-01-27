import pandas as pd
import logging

logger = logging.getLogger(__name__)

def aggregate_results(verification_results):
    """
    Aggregates verification results into a summary.
    
    Args:
        verification_results: List of dicts, where each dict contains a claim and its verifications.
        
    Returns:
        DataFrame containing the aggregated results.
    """
    aggregated_data = []

    for item in verification_results:
        claim = item.get("claim")
        verifications = item.get("verifications", [])
        
        # Simple aggregation strategy:
        # If any 'ENTAILMENT' -> Likely True (unless conflicted)
        # If 'CONTRADICTION' and no 'ENTAILMENT' -> Likely False
        # If only 'NEUTRAL' -> Not Verified
        
        entailment_count = 0
        contradiction_count = 0
        neutral_count = 0
        
        scores = []

        for v in verifications:
            label = v.get("nli_label", "NEUTRAL")
            conf = v.get("confidence", 0.0)
            
            if label == "ENTAILMENT":
                entailment_count += 1
                scores.append(conf)
            elif label == "CONTRADICTION":
                contradiction_count += 1
                scores.append(-conf) # Penalize
            else:
                neutral_count += 1
        
        # Determine strict status
        if entailment_count > 0 and contradiction_count == 0:
            status = "Supported"
            score = sum(scores) / len(scores) if scores else 0
        elif contradiction_count > 0 and entailment_count == 0:
            status = "Refuted"
            score = sum(scores) / len(scores) if scores else 0
        elif entailment_count > 0 and contradiction_count > 0:
            status = "Conflicting"
            score = 0
        elif neutral_count > 0:
            status = "Not Enough Info"
            score = 0
        else:
            status = "No Evidence"
            score = 0

        aggregated_data.append({
            "claim": claim,
            "status": status,
            "score": round(score, 2),
            "evidence_count": len(verifications),
            "entailments": entailment_count,
            "contradictions": contradiction_count
        })
    
    df = pd.DataFrame(aggregated_data)
    return df
