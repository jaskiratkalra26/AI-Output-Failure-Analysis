import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evidence_retrieval.search import duckduckgo_search
from evidence_retrieval.retriever import retrieve_evidence

class TestEvidenceRetrieval(unittest.TestCase):

    @patch('evidence_retrieval.search.DDGS')
    def test_duckduckgo_search_success(self, mock_ddgs_cls):
        """Test duckduckgo_search with valid results."""
        # Setup mock
        mock_ddgs_instance = mock_ddgs_cls.return_value
        mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
        
        mock_results = [
            {'body': 'Snippet 1', 'href': 'http://example.com/1'},
            {'body': 'Snippet 2', 'href': 'http://example.com/2'}
        ]
        mock_ddgs_instance.text.return_value = mock_results

        # Execute
        results = duckduckgo_search("test query", num_results=2)

        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['text'], 'Snippet 1')
        self.assertEqual(results[0]['source_url'], 'http://example.com/1')

    @patch('evidence_retrieval.search.DDGS')
    def test_duckduckgo_search_handled_exception(self, mock_ddgs_cls):
        """Test duckduckgo_search handles exceptions gracefully (retries and returns empty list eventually)."""
        # Setup mock to raise exception
        mock_ddgs_instance = mock_ddgs_cls.return_value
        mock_ddgs_instance.__enter__.side_effect = Exception("Connection error")
        
        # We need to mock the time.sleep to avoid waiting during tests
        with patch('time.sleep'):
            results = duckduckgo_search("failure case")
        
        # Assert
        self.assertEqual(results, [])

    @patch('evidence_retrieval.retriever.duckduckgo_search')
    def test_retrieve_evidence_flow(self, mock_search):
        """Test retrieve_evidence orchestrates the search correctly."""
        # Setup mock
        mock_search.return_value = [{'text': 'Proof', 'source_url': 'url'}]

        claims = ["The sky is blue", "Water is wet"]
        results = retrieve_evidence(claims)

        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['claim'], "The sky is blue")
        self.assertEqual(results[0]['evidence'][0]['text'], 'Proof')
        self.assertEqual(mock_search.call_count, 2)

    def test_retrieve_evidence_empty_input(self):
        """Test retrieve_evidence with empty input."""
        results = retrieve_evidence([])
        self.assertEqual(results, [])

    @patch('evidence_retrieval.retriever.duckduckgo_search')
    def test_retrieve_evidence_invalid_input(self, mock_search):
        """Test retrieve_evidence filters out invalid inputs.
           Also ensures we don't accidentally call the real API in unit tests.
        """
        # Setup mock to return something purely to verify it IS called for the valid one
        mock_search.return_value = []
        
        results = retrieve_evidence([None, "Valid", ""])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['claim'], "Valid")
        
        # Should only be called once for "Valid"
        mock_search.assert_called_once_with("Valid")

if __name__ == '__main__':
    unittest.main()
