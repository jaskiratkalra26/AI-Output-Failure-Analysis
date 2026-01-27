import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import torch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from verification.nli_model import NLIModel

class TestNLIModel(unittest.TestCase):
    
    @patch('verification.nli_model.AutoTokenizer')
    @patch('verification.nli_model.AutoModelForSequenceClassification')
    def test_initialization(self, mock_model_cls, mock_tokenizer_cls):
        """Test that the model and tokenizer are loaded correctly."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        # Fix: .to() needed to return the model itself or the chained object
        mock_model.to.return_value = mock_model
        
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        
        model = NLIModel(model_name="test-model")
        
        mock_tokenizer_cls.from_pretrained.assert_called_with("test-model")
        mock_model_cls.from_pretrained.assert_called_with("test-model")
        # Check chain
        self.assertEqual(model.tokenizer, mock_tokenizer)
        self.assertEqual(model.model, mock_model)

    @patch('verification.nli_model.AutoTokenizer')
    @patch('verification.nli_model.AutoModelForSequenceClassification')
    def test_initialization_failure(self, mock_model_cls, mock_tokenizer_cls):
        """Test initialization failure."""
        mock_tokenizer_cls.from_pretrained.side_effect = Exception("Download failed")
        
        with self.assertRaises(Exception):
            NLIModel(model_name="test-model")

    @patch('verification.nli_model.AutoTokenizer')
    @patch('verification.nli_model.AutoModelForSequenceClassification')
    def test_predict_entailment(self, mock_model_cls, mock_tokenizer_cls):
        """Test predict method for ENTAILMENT result."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model # Fix chain
        
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Configure model config id2label
        mock_model.config.id2label = {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
        
        # Setup tokenizer return value
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Setup model output (logits)
        # Class 2 (ENTAILMENT) has highest score
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 0.9]]) 
        mock_model.return_value = mock_output
        
        nli_model = NLIModel()
        label, confidence = nli_model.predict("premise", "hypothesis")
        
        self.assertEqual(label, "ENTAILMENT")
        # Softmax of [0.1, 0.2, 0.9] -> exp(0.9) / sum... roughly high prob
        self.assertTrue(confidence > 0.33) 

    @patch('verification.nli_model.AutoTokenizer')
    @patch('verification.nli_model.AutoModelForSequenceClassification')
    def test_predict_contradiction(self, mock_model_cls, mock_tokenizer_cls):
        """Test predict method for CONTRADICTION result."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model # Fix chain
        
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        
        mock_model.config.id2label = {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
        
        mock_tokenizer.return_value = {'input_ids': torch.tensor([[1]]), 'attention_mask': torch.tensor([[1]])}
        
        # Class 0 (CONTRADICTION) highest
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.9, 0.1, 0.1]])
        mock_model.return_value = mock_output
        
        nli_model = NLIModel()
        label, confidence = nli_model.predict("premise", "hypothesis")
        
        self.assertEqual(label, "CONTRADICTION")

    @patch('verification.nli_model.AutoTokenizer')
    @patch('verification.nli_model.AutoModelForSequenceClassification')
    def test_predict_exception(self, mock_model_cls, mock_tokenizer_cls):
        """Test predict method handles exceptions gracefully."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model  # Fix chain
        
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Tokenizer raises exception
        mock_tokenizer.side_effect = Exception("Tokenization error")
        
        nli_model = NLIModel()
        label, confidence = nli_model.predict("premise", "hypothesis")
        
        self.assertIsNone(label)
        self.assertEqual(confidence, 0.0)

if __name__ == '__main__':
    unittest.main()
