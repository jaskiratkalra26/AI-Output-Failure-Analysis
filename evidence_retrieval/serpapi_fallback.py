import os
import requests
import logging
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Try to import helpers from local retriever, fallback if not possible
try:
    from .retriever import extract_key_entity, get_claim_keywords
    # Also attempt to import calculate_relevance_score if available, otherwise define locally
    from .retriever import calculate_relevance_score
except ImportError:
    # Fallback implementations if import fails or running standalone
    def extract_key_entity(claim: str) -> str:
        match = re.search(r'\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\b', claim)
        return match.group(0) if match else claim

    def get_claim_keywords(claim: str) -> List[str]:
        stopwords = {"the", "was", "is", "in", "of", "and", "a", "to"}
        return [w.lower() for w in re.findall(r'\w+', claim) 
                if w.lower() not in stopwords and len(w) > 3]

    def calculate_relevance_score(sentence: str, keywords: List[str], entity: str) -> int:
        s_lower = sentence.lower()
        score = 0
        if entity and entity.lower() in s_lower:
            score += 1
        for k in keywords:
            if k in s_lower:
                score += 1
        return score

# Removed strict domain allowlist as per user request to allow all domains.
# ALLOWED_DOMAINS = {...}

def is_recent(result_item: Dict) -> bool:
    """
    Checks recency of the result.
    Prefer results published in the last 5–10 years.
    Ignore publication date if unavailable (i.e., return True).
    """
    date_str = result_item.get("date")
    if not date_str:
        return True # Ignore if unavailable
    
    # Simple regex to extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        try:
            year = int(year_match.group(0))
            current_year = datetime.now().year
            # Strict filtering: if we know the year, and it's older than 10 years, maybe we should warn?
            # Prompt says "Prefer ... Ignore if unavailable".
            # I will accept it if it's within last 15 years to be safe, or just accept all if vague.
            # "Prefer" suggests ranking, but we are filtering.
            # Use 10 year cutoff for strictness as per "3 Recency handling".
            if current_year - year <= 10:
                return True
            else:
                return False # Too old
        except ValueError:
            return True
            
    return True

def fetch_serpapi_evidence(claim: str, api_key: str) -> List[Dict]:
    """
    Core function to fetch evidence from SerpAPI for a single claim.
    """
    logger.info(f"Querying SerpAPI for claim: {claim}")
        
    # Construct Query
    entity = extract_key_entity(claim)
    keywords = get_claim_keywords(claim)
    
    query_parts = []
    if entity and len(entity.split()) > 0:
        query_parts.append(f'"{entity}"')
    
    query_parts.extend(keywords)
    
    if not query_parts:
        query = claim
    else:
        query = " ".join(query_parts)

    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": 5,              # Fetch top 5
            "hl": "en",            # Language: English
            "gl": "us",            # Region: US 
            "google_domain": "google.com"
        }
        
        # Execute Search
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        response.raise_for_status()
        results_json = response.json()
        
        if "error" in results_json:
            logger.error(f"SerpAPI returned error: {results_json['error']}")
            return []
            
        organic_results = results_json.get("organic_results", [])
        
        valid_evidence = []
        seen_urls = set()
        
        for item in organic_results:
            if len(valid_evidence) >= 2:
                break
                
            url = item.get("link")
            snippet = item.get("snippet", "")
            
            if not url or not snippet:
                continue
                
            # Deduplication
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # 1. Language Filter (ASCII check)
            if not snippet.isascii():
                continue
                
            # 2. Domain Allowlist - REMOVED
                
            # 3. Recency handling
            if not is_recent(item):
                continue
            
            # 4. Relevance Check (ensure snippet matches claim/entity)
            if calculate_relevance_score(snippet, keywords, entity) <= 0:
                continue
                
            valid_evidence.append({
                "text": snippet,
                "source_url": url
            })
            
        if not valid_evidence:
             logger.warning(f"SerpAPI found no valid evidence for claim: {claim}")
             
        return valid_evidence

    except Exception as e:
        logger.error(f"Failed to retrieve fallback evidence for '{claim}': {e}")
        return []

