import os
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from config.config import CONFIG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ClaimExtractor:
    def __init__(self):
        self.config = CONFIG
        self.model = self._get_gemini_model()

    def _get_gemini_model(self):
        """Configure and return the Gemini model."""
        model_config = self.config.get("model_config", {})
        api_key_env = model_config.get("gemini_api_key_env", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            logging.error(f"{api_key_env} environment variable not found.")
            raise ValueError(f"{api_key_env} not set")
        
        genai.configure(api_key=api_key)
        model_name = model_config.get("gemini_model", "gemini-2.5-flash")
        return genai.GenerativeModel(model_name)

    def extract_claims_from_text(self, text):
        """
        Send text to Gemini and parse JSON response.
        Returns: list of strings (claims) or None if error.
        """
        prompt_template = self.config.get("prompts", {}).get("claim_extraction_prompt", "")
        if not prompt_template:
            # Fallback if config prompt is empty (should not happen)
            logging.error("Claim extraction prompt missing from config.")
            return None
            
        prompt = prompt_template.replace("{LLM_ANSWER}", text)
        
        try:
            response = self.model.generate_content(prompt)
            clean_text = response.text.strip()
            logging.debug(f"Raw model response: {clean_text}")
            
            # Clean up JSON if it contains markdown code blocks
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
                
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            clean_text = clean_text.strip()
            
            claims = json.loads(clean_text)
            
            if not isinstance(claims, list):
                logging.warning(f"Invalid JSON format (not a list): {clean_text}")
                return []
                
            return [str(c) for c in claims]

        except Exception as e:
            logging.warning(f"Error extracting claims: {e}")
            return None

