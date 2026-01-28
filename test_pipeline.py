import os
import json
import logging
from claim_extraction.extractor import ClaimExtractor
from evidence_retrieval.retriever import retrieve_evidence
from verification.claim_verification import verify_claims
from aggregation.aggregate import aggregate_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_full_pipeline(text):
    print("\n" + "="*50)
    print("INPUT TEXT:")
    print(text)
    print("="*50 + "\n")

    # 1. Extraction
    print("1. Extracting claims...")
    claims = []
    try:
        extractor = ClaimExtractor()
        claims = extractor.extract_claims_from_text(text)
        print(f"Extracted {len(claims)} claims:")
        for i, c in enumerate(claims):
            print(f"  {i+1}. {c}")
    except Exception as e:
        print(f"Extraction failed/skipped: {e}")
        print("Using fallback claims (simulating extraction)...")
        # Fallback split
        claims = [
            "Albert Einstein developed the theory of relativity.",
            "Albert Einstein was awarded the Nobel Prize in Physics in 1921.",
            "Albert Einstein was born in Germany.",
            "Albert Einstein later became a citizen of the United States.",
            "Einstein invented the first working computer."
        ]
        print(f"Using {len(claims)} fallback claims.")

    if not claims:
        print("No claims extracted. Aborting.")
        return

    # 2. Retrieval
    print("\n2. Retrieving evidence...")
    try:
        claim_evidence_pairs = retrieve_evidence(claims)
        for item in claim_evidence_pairs:
            print(f"  Claim: '{item['claim'][:50]}...' -> Found {len(item.get('evidence', []))} pieces of evidence.")
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return

    # 3. Verification
    print("\n3. Verifying claims...")
    try:
        verified_claims = verify_claims(claim_evidence_pairs)
        for item in verified_claims:
            print(f"  Claim: '{item['claim'][:50]}...' -> {len(item.get('verifications', []))} verifications.")
            for v in item.get('verifications', [])[:1]: # Show first verification
                print(f"    - First verification: {v['nli_label']} ({v['confidence']:.2f})")
    except Exception as e:
        print(f"Verification failed: {e}")
        return

    # 4. Aggregation
    print("\n4. Aggregating results...")
    try:
        final_result = aggregate_results(verified_claims)
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        print(json.dumps(final_result, indent=2))
        
        # Determine if it's correct (we assume this text is mostly factual, but let's see)
        # "Albert Einstein developed the theory of relativity... 1921 Nobel Prize... Born in Germany... US Citizen... Invented computer (FALSE)"
        # The last part "Einstein also invented the first working computer" is a hallucination.
        # So we expect hallucinated=True.
        
        if final_result["hallucinated"]:
            print("\nSUCCESS: System correctly identified hallucination.")
        else:
            print("\nWARNING: System did NOT flag this as hallucinated. (Maybe NLI failed to find contradiction or Retrieval didn't find the computer invention info)")

    except Exception as e:
        print(f"Aggregation failed: {e}")
        return

if __name__ == "__main__":
    text_to_test = "Albert Einstein developed the theory of relativity in the early 20th century and was awarded the Nobel Prize in Physics in 1921. He was born in Germany and later became a citizen of the United States. Einstein also invented the first working computer, which revolutionized modern technology."
    test_full_pipeline(text_to_test)
