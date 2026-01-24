import os
import json
import logging
import sqlite3
import time
import pandas as pd
import yaml
import google.generativeai as genai
from dotenv import load_dotenv
from .prompts import CLAIM_EXTRACTION_PROMPT

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ClaimExtractor:
    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)
        self.db_path = self.config.get("database_path", "data/hallucination.db")
        self.setup_database()
        self.model = self._get_gemini_model()

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def setup_database(self):
        """Create the SQLite table if it doesn't exist."""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER,
                claim_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logging.info(f"Database initialized: {self.db_path}")

    def _get_gemini_model(self):
        """Configure and return the Gemini model."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY environment variable not found.")
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        model_name = self.config.get("gemini_model", "gemini-pro")
        return genai.GenerativeModel(model_name)

    def extract_claims_from_text(self, text):
        """
        Send text to Gemini and parse JSON response.
        Returns: list of strings (claims) or None if error.
        """
        prompt = CLAIM_EXTRACTION_PROMPT.replace("{LLM_ANSWER}", text)
        
        try:
            response = self.model.generate_content(prompt)
            clean_text = response.text.strip()
            
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

    def run(self):
        input_csv = self.config.get("input_data_path", "phase1_llm_outputs.csv")
        
        if not os.path.exists(input_csv):
            logging.error(f"Input file {input_csv} not found.")
            return

        df = pd.read_csv(input_csv)
        
        if 'id' not in df.columns or 'llm_answer' not in df.columns:
            logging.error("CSV missing required columns: id, llm_answer")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        total_claims = 0
        logging.info(f"Starting claim extraction for {len(df)} rows...")

        for index, row in df.iterrows():
            sample_id = row['id']
            llm_answer = row['llm_answer']
            
            # Skip if answer is missing/NaN
            if pd.isna(llm_answer) or str(llm_answer).strip() == "":
                logging.warning(f"Skipping row {sample_id}: Empty LLM answer")
                continue

            logging.info(f"Processing ID {sample_id}...")
            
            # 1. First Attempt
            claims = self.extract_claims_from_text(llm_answer)
            
            # 2. Retry Attempt if failed
            if claims is None:
                logging.info(f"Retrying ID {sample_id} after failure...")
                time.sleep(2)
                claims = self.extract_claims_from_text(llm_answer)
            
            # 3. Handle Final Failure
            if claims is None:
                logging.error(f"Failed to process ID {sample_id} after retry. Skipping.")
                continue
                
            # 4. Insert Claims
            if not claims:
                logging.info(f"No claims found for ID {sample_id}")
            else:
                rows_to_insert = [(int(sample_id), claim) for claim in claims]
                cursor.executemany("INSERT INTO claims (sample_id, claim_text) VALUES (?, ?)", rows_to_insert)
                conn.commit()
                total_claims += len(claims)
                logging.info(f"  -> Extracted {len(claims)} claims.")
            
            # Rate limiting pause
            time.sleep(20)

        conn.close()
        logging.info(f"Extraction complete. Total claims stored: {total_claims}")

if __name__ == "__main__":
    extractor = ClaimExtractor()
    extractor.run()
