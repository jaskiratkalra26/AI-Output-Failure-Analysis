import os
import sys
import pandas as pd
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import CONFIG
from claim_extraction.extractor import ClaimExtractor
from evidence_retrieval.retriever import retrieve_evidence
from verification.claim_verification import verify_claims
from aggregation.aggregate import aggregate_results

import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
paths_config = CONFIG.get("paths", {})
extraction_config = CONFIG.get("extraction", {})

def run(input_path=None, output_path=None, column_name=None):
    logger.info("Starting Hallucination Detection Pipeline...")
    
    # 1. Determine Input File
    # Prioritize function argument (CLI), then config, then default
    input_file = input_path or paths_config.get("input_data_path", "phase1_llm_outputs.csv")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found.")
        return

    logger.info(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return
    
    # 2. Determine Text Column
    text_column = None
    
    if column_name:
        if column_name in df.columns:
            text_column = column_name
        else:
            logger.error(f"Provided column '{column_name}' not found in CSV. Available columns: {list(df.columns)}")
            return
    elif len(df.columns) == 1:
        text_column = df.columns[0]
        logger.info(f"Detected single column CSV. Using column '{text_column}' as input.")
    elif "llm_answer" in df.columns:
        text_column = "llm_answer"
        logger.info(f"Using default column '{text_column}' as input.")
    else:
        logger.error("Could not determine input column. Please provide one with --column or use a CSV with a single column or 'llm_answer' column.")
        return
    
    # Initialize components
    extractor = None
    try:
        extractor = ClaimExtractor()
    except ValueError as e:
        logger.warning(f"ClaimExtractor initialization failed: {e}. Using naive sentence splitting fallback.")
        # We will handle the fallback inside the loop
    except Exception as e:
        logger.error(f"Unexpected error initializing ClaimExtractor: {e}")
        return

    all_results = []
    
    # Processing all rows
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        llm_answer = row[text_column]
        # Use existing ID if available, otherwise use index
        row_id = row.get("id", index)
        
        # 2. Extract Claims
        claims = []
        if extractor:
            try:
                claims = extractor.extract_claims_from_text(str(llm_answer))
            except Exception as e:
                logger.error(f"Error extracting claims for row {row_id}: {e}")
        
        # Fallback if extractor failed or not initialized
        if not claims:
            # Naive fallback: split by periods if the answer is long, else take the whole thing
            # This is just for testing verification when extractor is down
            if isinstance(llm_answer, str):
                min_len = extraction_config.get("naive_fallback_min_length", 10)
                claims = [s.strip() for s in llm_answer.split('.') if len(s.strip()) > min_len]
                if not claims: 
                    claims = [llm_answer]
            else:
                claims = []

        if not claims:
            logger.warning(f"No claims extracted for row {row_id}. Skipping.")
            continue
            
        logger.info(f"Row {row_id}: Extracted {len(claims)} claims.")

        # 3. Retrieve Evidence
        claim_evidence_pairs = retrieve_evidence(claims)
        
        # 4. Verify Claims
        verification_results = verify_claims(claim_evidence_pairs)
        
        # 5. Aggregate Results
        agg_df = aggregate_results(verification_results)
        
        # Add metadata
        agg_df["row_id"] = row_id
        agg_df["original_answer"] = llm_answer
        
        all_results.append(agg_df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # Determine output path
        if output_path:
            final_output_file = output_path
        else:
            final_output_file = paths_config.get("pipeline_output_path", "final_pipeline_output.csv")
            
        final_df.to_csv(final_output_file, index=False)
        logger.info(f"Pipeline completed. Results saved to {final_output_file}")
        print(final_df.head())
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hallucination Detection Pipeline")
    parser.add_argument("--input", "-i", type=str, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, help="Path to output CSV file")
    parser.add_argument("--column", "-c", type=str, help="Name of the column containing the LLM output text")
    
    args = parser.parse_args()
    
    run(input_path=args.input, output_path=args.output, column_name=args.column)
