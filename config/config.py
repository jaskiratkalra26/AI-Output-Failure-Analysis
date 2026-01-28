
import os
from dotenv import load_dotenv

# Load environment variables once in config
load_dotenv()

CONFIG = {
    "model_config": {
        "gemini_model": "gemini-2.5-flash",
        "gemini_api_key_env": "GEMINI_API_KEY",
        "nli_model_name": "roberta-large-mnli"
    },
    "paths": {
        "input_data_path": "phase1_llm_outputs.csv",
        "pipeline_output_path": "final_pipeline_output.csv"
    },
    "retrieval": {
        "wikipedia_language": "en",
        "max_search_results": 1,
        "max_paragraphs": 5,
        "min_sentence_length": 20,
        "document_top_n_sentences": 2,
        "fallback_threshold_score": 2,
        "entity_match_score": 1,
        "keyword_match_score": 1,
        "wikipedia_domain": "wikipedia.org",
        "serpapi_api_key_env": "SERPAPI_API_KEY",
        "stopwords": [
            "the", "was", "is", "in", "of", "and", "a", "to", "he", "she", "it", 
            "they", "for", "on", "with", "as", "at", "by", "an", "that", "this", 
            "from", "which", "are", "were"
        ],
        "keyword_min_length": 3
    },
    "verification": {
        "nli_max_length": 512,
        "contradiction_threshold": 0.80,
        "keyword_overlap_threshold": 0.7,
        "supportive_entailment_confidence": 0.9
    },
    "extraction": {
        "naive_fallback_min_length": 10
    },
    "prompts": {
        "claim_extraction_prompt": """
You are an expert fact-checker and claim extractor.
Your task is to identify and extract atomic, verifiable claims from the provided text.
A "claim" is a statement that asserts a fact which can be proven true or false.

Instructions:
1. Extract independent sentences or clauses that contain factual assertions.
2. Ignore opinions, questions, greetings, or subjective statements.
3. Ensure each extracted claim is self-contained (resolve pronouns like "he", "it" to the entities they refer to if possible based on context).
4. Output strict JSON format: a list of strings.

Input Text:
"{LLM_ANSWER}"

Output ONLY the JSON list. Example:
["The earth revolves around the sun.", "Water boils at 100 degrees Celsius."]
"""
    }
}
