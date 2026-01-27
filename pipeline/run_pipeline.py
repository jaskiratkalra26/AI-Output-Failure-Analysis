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

from claim_extraction.extractor import ClaimExtractor
from evidence_retrieval.retriever import retrieve_evidence
from verification.claim_verification import verify_claims
from aggregation.aggregate import aggregate_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run():
    logger.info("Starting Hallucination Detection Pipeline...")
    
    # 1. Load Input Data
    input_file = "phase1_llm_outputs.csv"
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found.")
        return

    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    if "llm_answer" not in df.columns:
        logger.error("Column 'llm_answer' not found in input CSV.")
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
        llm_answer = row["llm_answer"]
        row_id = row.get("id", index)
        
        # 2. Extract Claims
        claims = []
        if extractor:
            try:
                claims = extractor.extract_claims_from_text(llm_answer)
            except Exception as e:
                logger.error(f"Error extracting claims for row {row_id}: {e}")
        
        # Fallback if extractor failed or not initialized
        if not claims:
            # Naive fallback: split by periods if the answer is long, else take the whole thing
            # This is just for testing verification when extractor is down
            if isinstance(llm_answer, str):
                claims = [s.strip() for s in llm_answer.split('.') if len(s.strip()) > 10]
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
        output_file = "final_pipeline_output.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"Pipeline completed. Results saved to {output_file}")
        print(final_df.head())
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    run()
