from typing import List, Dict
from evidence_retrieval.search import duckduckgo_search

def retrieve_evidence(claims: List[str]) -> List[Dict]:
    """
    Retrieves evidence for a list of factual claims.
    
    Args:
        claims: A list of claim strings.
        
    Returns:
        A list of dictionaries where each dictionary contains the claim and its evidence.
    """
    results = []
    
    for claim in claims:
        if not claim or not isinstance(claim, str):
            continue
            
        # Use the claim text directly as the search query
        evidences = duckduckgo_search(claim)
        
        results.append({
            "claim": claim,
            "evidence": evidences
        })
        
    return results
