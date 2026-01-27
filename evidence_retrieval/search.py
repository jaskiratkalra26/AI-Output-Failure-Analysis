import logging
import time
from typing import List, Dict
try:
    from duckduckgo_search import DDGS
except ImportError:
    # Fallback or instructions if package isn't installed
    DDGS = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def duckduckgo_search(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Performs a DuckDuckGo search for the given query.
    
    Args:
        query: The search query string.
        num_results: Number of results to retrieve (default: 3).
        
    Returns:
        A list of dictionaries containing 'text' (snippet) and 'source_url'.
    """
    try:
        return _execute_search(query, num_results)
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for query '{query}': {e}. Retrying once...")
        time.sleep(1)
        try:
            return _execute_search(query, num_results)
        except Exception as e:
            logger.error(f"Second DuckDuckGo search attempt failed for query '{query}': {e}. Skipping.")
            return []

def _execute_search(query: str, num_results: int) -> List[Dict[str, str]]:
    """Helper to execute the search."""
    results = []
    # logger.info(f"Searching for: {query}")
    
    # Priority: lite (usually more robust for scraping), then html, then auto
    backends = ["lite", "html", "api"]
    
    found_results = []
    
    with DDGS() as ddgs:
        for backend in backends:
            try:
                # Add delay to avoid rate limiting
                time.sleep(2)
                
                # logger.info(f"Trying backend: {backend}")
                found_results = list(ddgs.text(query, max_results=num_results, backend=backend))
                
                if found_results:
                    # logger.info(f"Backend '{backend}' returned {len(found_results)} results.")
                    break # Success
                else:
                    # logger.warning(f"Backend '{backend}' returned 0 results.")
                    pass
                    
            except Exception as e:
                logger.warning(f"Error with backend '{backend}': {e}")
                time.sleep(1)

    if found_results:
        seen_links = set()
        for result in found_results:
            snippet = result.get("body", "").strip()
            link = result.get("href", "").strip()
            
            if snippet and link and link not in seen_links:
                results.append({
                    "text": snippet,
                    "source_url": link
                })
                seen_links.add(link)
                
    return results
