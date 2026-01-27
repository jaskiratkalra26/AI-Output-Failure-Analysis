import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from verification.claim_verification import verify_claims

class TestClaimVerification(unittest.TestCase):
    
    @patch('verification.claim_verification.get_nli_model')
    def test_verify_claims_success(self, mock_get_model):
        """Test verifying claims with valid inputs and model."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        # Setup model prediction: (Label, Confidence)
        # First call: ENTAILMENT, Second call: CONTRADICTION
        mock_model.predict.side_effect = [
            ("ENTAILMENT", 0.95),
            ("CONTRADICTION", 0.85)
        ]
        
        claims_data = [
            {
                "claim": "Claim 1",
                "evidence": [
                    {"text": "Evidence 1A", "source_url": "url1"},
                    {"text": "Evidence 1B", "source_url": "url2"}
                ]
            }
        ]
        
        results = verify_claims(claims_data)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["claim"], "Claim 1")
        self.assertEqual(len(results[0]["verifications"]), 2)
        
        # Check first verification
        v1 = results[0]["verifications"][0]
        self.assertEqual(v1["nli_label"], "ENTAILMENT")
        self.assertEqual(v1["confidence"], 0.95)
        self.assertEqual(v1["evidence_text"], "Evidence 1A")
        
        # Check second verification
        v2 = results[0]["verifications"][1]
        self.assertEqual(v2["nli_label"], "CONTRADICTION")

    @patch('verification.claim_verification.get_nli_model')
    def test_verify_claims_model_not_loaded(self, mock_get_model):
        """Test behavior when NLI model fails to load (returns None)."""
        mock_get_model.return_value = None
        
        claims_data = [{"claim": "Claim 1", "evidence": [{"text": "Ev 1"}]}]
        
        results = verify_claims(claims_data)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["claim"], "Claim 1")
        self.assertEqual(results[0]["verifications"], [])

    @patch('verification.claim_verification.get_nli_model')
    def test_verify_claims_no_evidence(self, mock_get_model):
        """Test verifying claims with no evidence provided."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        claims_data = [{"claim": "Claim 1", "evidence": []}]
        
        results = verify_claims(claims_data)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["verifications"], [])
        mock_model.predict.assert_not_called()

    @patch('verification.claim_verification.get_nli_model')
    def test_verify_claims_inference_failure(self, mock_get_model):
        """Test behavior when model.predict returns None (error)."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        # Predict returns None, 0.0
        mock_model.predict.return_value = (None, 0.0)
        
        claims_data = [{"claim": "Claim 1", "evidence": [{"text": "Ev 1"}]}]
        
        results = verify_claims(claims_data)
        
        # Should have empty verifications because the failure is skipped
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["verifications"], [])

if __name__ == '__main__':
    unittest.main()
