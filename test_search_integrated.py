from evidence_retrieval.retriever import retrieve_evidence
import logging

logging.basicConfig(level=logging.INFO)

def test_retrieval():
    claims = ["The Eiffel Tower was completed in 1889"]
    print(f"Testing retrieval for: {claims}")
    results = retrieve_evidence(claims)
    print(f"Evidence found: {len(results[0]['evidence'])}")
    if results[0]['evidence']:
        print("First evidence snippet:", results[0]['evidence'][0]['text'][:50])
    else:
        print("No evidence found.")

if __name__ == "__main__":
    test_retrieval()
