from duckduckgo_search import DDGS
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search():
    query = "The Eiffel Tower was completed in 1889"
    print(f"Testing query: {query}")
    
    with DDGS() as ddgs:
        print("\n--- Trying Default (Auto) ---")
        try:
            results = list(ddgs.text(query, max_results=3))
            print(f"Results: {len(results)}")
            if results: print(results[0].keys())
        except Exception as e:
            print(f"Error: {e}")

        print("\n--- Trying Lite ---")
        try:
            results = list(ddgs.text(query, max_results=3, backend='lite'))
            print(f"Results: {len(results)}")
        except Exception as e:
            print(f"Error: {e}")

        print("\n--- Trying HTML ---")
        try:
            results = list(ddgs.text(query, max_results=3, backend='html'))
            print(f"Results: {len(results)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_search()
