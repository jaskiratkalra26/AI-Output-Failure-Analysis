from duckduckgo_search import DDGS
import logging
import json

logging.basicConfig(level=logging.INFO)

def debug_search():
    query = "Elon Musk is the CEO of Tesla"
    print(f"Query: {query}")
    with DDGS() as ddgs:
        # Try lite
        print("Trying lite...")
        results = list(ddgs.text(query, max_results=1, backend="lite"))
        print(f"Count: {len(results)}")
        if results:
            print("Keys:", results[0].keys())
            print("Item:", json.dumps(results[0], indent=2))
            
        # Try html
        print("Trying html...")
        results = list(ddgs.text(query, max_results=1, backend="html"))
        print(f"Count: {len(results)}")
        if results:
            print("Keys:", results[0].keys())

debug_search()
