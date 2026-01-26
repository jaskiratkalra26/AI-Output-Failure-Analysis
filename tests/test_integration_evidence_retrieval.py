import unittest
import sys
import os

# Add the project root to the python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evidence_retrieval.search import duckduckgo_search

class TestEvidenceRetrievalIntegration(unittest.TestCase):

    def test_duckduckgo_search_live(self):
        """
        Integration test to verify that duckduckgo_search actually works with the live DDGS API.
        Requires internet connection.
        """
        query = "Python programming language"
        print(f"\nRunning live search for: '{query}'...")
        results = duckduckgo_search(query, num_results=2)
        
        # We expect at least some results if the API is working
        self.assertGreater(len(results), 0, "Live search returned no results. Check internet connection or API status.")
        
        first_result = results[0]
        self.assertIn('text', first_result)
        self.assertIn('source_url', first_result)
        print(f"Live search successful. Retrieved {len(results)} results.")
        print(f"First result title/snippet: {first_result['text'][:50]}...")

if __name__ == '__main__':
    unittest.main()
