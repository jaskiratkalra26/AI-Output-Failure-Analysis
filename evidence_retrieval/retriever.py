import logging
import re
import wikipedia
import os
from typing import List, Dict, Optional
from config.config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Load config
retrieval_config = CONFIG.get("retrieval", {})

# Try to import SerpAPI helper
try:
    from .serpapi_fallback import fetch_serpapi_evidence
except ImportError:
    logger.warning("Could not import fetch_serpapi_evidence. Fallback will not be active.")
    def fetch_serpapi_evidence(claim, api_key): return []

# Set Wikipedia language
try:
    lang = retrieval_config.get("wikipedia_language", "en")
    wikipedia.set_lang(lang)
except Exception as e:
    logger.error(f"Failed to set Wikipedia language: {e}")

def extract_key_entity(claim: str) -> str:
    """
    Extracts the key entity from the claim using an improved heuristic:
    Finds all capitalized phrases, filters out common starting stopwords, 
    and picks the longest remaining candidate.
    """
    if not claim:
        return ""
        
    # 1. Find all sequences of capitalized words
    # matches e.g. "Isaac Newton", "The", "George Orwell"
    matches = re.finditer(r'\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\b', claim)
    candidates = [m.group(0) for m in matches]
    
    if not candidates:
        return claim  # Fallback: use whole claim
        
    # 2. Define stopwords to filter (if they appear as the *entire* candidate)
    # We load from config and also add common question starters
    config_stopwords = retrieval_config.get("stopwords", [])
    # Capitalize them for comparison
    stopwords = set(s.capitalize() for s in config_stopwords)
    # Add question words and others that might start a sentence
    stopwords.update({"When", "Where", "Who", "What", "Why", "How", "There", "Here"})
    
    filtered_candidates = []
    for c in candidates:
        # If the candidate matches a stopword exactly, skip it
        # But if it's "The Beatles", "The" is part of the phrase, so regex handles it?
        # Our regex `\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\b` grabs "The Beatles" as one group.
        # So checking `if c in stopwords` is safe for "The" vs "The Beatles".
        if c in stopwords:
            continue
        filtered_candidates.append(c)
        
    # If we filtered everything (e.g. claim is just "The"), revert to all candidates
    final_candidates = filtered_candidates if filtered_candidates else candidates
    
    # 3. Pick the longest candidate by character length
    # This favors "George Orwell" (13) over "The" (3) or "Novel" (5) if customized
    best_entity = max(final_candidates, key=len)
    
    return best_entity

def split_text_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using regex.
    """
    # Split by period, question mark, exclamation mark followed by whitespace.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def get_claim_keywords(claim: str) -> List[str]:
    """
    Extracts keywords from claim (words > 3 chars, excluding common stopwords).
    """
    stopwords = set(retrieval_config.get("stopwords", []))
    if not stopwords:
        stopwords = {
            "the", "was", "is", "in", "of", "and", "a", "to", "he", "she", "it", 
            "they", "for", "on", "with", "as", "at", "by", "an", "that", "this", 
            "from", "which", "are", "were"
        }
    
    words = re.findall(r'\w+', claim)
    min_len = retrieval_config.get("keyword_min_length", 3)
    keywords = [w.lower() for w in words if w.lower() not in stopwords and len(w) > min_len]
    return keywords

def simple_stem(word: str) -> str:
    """
    Rudimentary stemmer to handle pluralization and common suffixes.
    """
    word = word.lower()
    suffixes = ["ation", "tion", "sion", "ment", "ing", "ed", "es", "al", "ly", "y", "s"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

def calculate_relevance_score(sentence: str, keywords: List[str], entity: str) -> int:
    """
    Calculates a relevance score for a sentence based on keyword overlap.
    """
    s_lower = sentence.lower()
    s_words = set(re.findall(r'\w+', s_lower))
    score = 0
    
    # Get weights from config
    entity_score = retrieval_config.get("entity_match_score", 1)
    keyword_score = retrieval_config.get("keyword_match_score", 1)

    # Check entity (strict string match)
    if entity and entity.lower() in s_lower:
        score += entity_score
    elif entity:
        # Partial entity match (e.g. "Newton" in "Isaac Newton")
        parts = entity.split()
        if len(parts) > 1:
            surname = parts[-1].lower()
            if surname in s_words:
                score += (entity_score * 0.5)
        
    # Check keywords with stemming and fuzzy matching
    matched_keywords = set()
    for k in keywords:
        k_stem = simple_stem(k)
        
        # 1. Exact/Substring match
        if k in s_lower:
            matched_keywords.add(k)
            continue
            
        # 2. Stem match against words
        for w in s_words:
            if simple_stem(w) == k_stem:
                matched_keywords.add(k)
                break
    
    score += len(matched_keywords) * keyword_score
            
    return score

def retrieve_evidence(claims: List[str]) -> List[Dict]:
    """
    Retrieves high-quality evidence from Wikipedia, falling back to SerpAPI if needed.
    """
    results = []
    
    for claim in claims:
        if not claim or not isinstance(claim, str):
            continue
            
        logger.info(f"Processing claim: {claim}")
        
        entity = extract_key_entity(claim)
        keywords = get_claim_keywords(claim)
        
        found_evidence = []
        max_score = 0
        
        # --- Phase 1: Try Wikipedia ---
        try:
            search_term = entity if entity else claim
            max_search = retrieval_config.get("max_search_results", 1)
            search_results = wikipedia.search(search_term, results=max_search)
            
            if not search_results:
                logger.warning(f"No Wikipedia results found for claim/entity: {search_term}")
            else:
                page_title = search_results[0]
                try:
                    # We use auto_suggest=False because search() already gives us the best match title.
                    # Enabling it can cause weird redirects (e.g. 'Australia' -> 'List of...').
                    page = wikipedia.page(page_title, auto_suggest=False)
                    
                    if "en.wikipedia.org" in page.url:
                        # Process Content
                        potential_text_blocks = []
                        if page.summary:
                            potential_text_blocks.append(page.summary)
                        content = page.content
                        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
                        max_paras = retrieval_config.get("max_paragraphs", 5)
                        potential_text_blocks.extend(paragraphs[:max_paras])
                        
                        candidate_sentences = []
                        seen_sentences = set()
                        
                        for block in potential_text_blocks:
                            sentences = split_text_into_sentences(block)
                            for sent in sentences:
                                clean_sent = sent.replace('\n', ' ').strip()
                                if not clean_sent or clean_sent in seen_sentences: continue
                                min_sent_len = retrieval_config.get("min_sentence_length", 20)
                                if len(clean_sent) < min_sent_len: continue
                                
                                score = calculate_relevance_score(clean_sent, keywords, entity)
                                if score > 0:
                                    candidate_sentences.append({
                                        "text": clean_sent,
                                        "source_url": page.url,
                                        "score": score
                                    })
                                    seen_sentences.add(clean_sent)
                        
                        candidate_sentences.sort(key=lambda x: x["score"], reverse=True)
                        
                        if candidate_sentences:
                            max_score = candidate_sentences[0]["score"]
                            # Add top N
                            top_n = retrieval_config.get("document_top_n_sentences", 2)
                            entries_to_add = candidate_sentences[:top_n]
                            for entry in entries_to_add:
                                found_evidence.append({
                                    "text": entry["text"],
                                    "source_url": entry["source_url"]
                                })
                
                except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                    logger.warning(f"Wikipedia page retrieval failed: {e}")

        except Exception as e:
            logger.error(f"Error during Wikipedia retrieval: {e}")

        # --- Phase 2: Fallback Logic ---
        # Activate if: No evidence found OR evidence is weak (max_score < threshold)
        fallback_threshold = retrieval_config.get("fallback_threshold_score", 2)
        if not found_evidence or max_score < fallback_threshold:
            reason = "No evidence found" if not found_evidence else f"Weak evidence (score {max_score})"
            logger.info(f"Triggering SerpAPI fallback for '{claim}'. Reason: {reason}")
            
            serp_env_var = retrieval_config.get("serpapi_api_key_env", "SERPAPI_API_KEY")
            api_key = os.environ.get(serp_env_var)
            if api_key:
                try:
                    serp_evidence = fetch_serpapi_evidence(claim, api_key)
                    if serp_evidence:
                        # Append valid SerpAPI evidence
                        # Note: We rely on fetch_serpapi_evidence to filter irrelevant results (score <= 0)
                        found_evidence.extend(serp_evidence)
                except Exception as e:
                    logger.error(f"Error during fallback retrieval: {e}")
            else:
                logger.warning("SERPAPI_API_KEY missing, skipping fallback.")

        # --- Phase 3: Build Result ---
        if found_evidence:
            # Determine source label
            sources = [e.get("source_url", "") for e in found_evidence]
            wiki_domain = retrieval_config.get("wikipedia_domain", "wikipedia.org")
            has_wiki = any(wiki_domain in s for s in sources)
            has_other = any(wiki_domain not in s for s in sources)
            
            if has_wiki and has_other:
                source_label = "mixed"
            elif has_wiki:
                source_label = "wikipedia"
            else:
                source_label = "serpapi"

            results.append({
                "claim": claim,
                "evidence": found_evidence,
                "source": source_label
            })
        else:
            logger.warning(f"No valid evidence found for claim: {claim}")
            
    return results
