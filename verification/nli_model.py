import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class NLIModel:
    def __init__(self, model_name="roberta-large-mnli", max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        print(f"Loading NLI model: {model_name} on {self.device}... (This may take a few minutes for the first download)")
        logger.info(f"Loading NLI model: {model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("NLI model loaded successfully.")
            logger.info("NLI model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise e

    def predict(self, premise: str, hypothesis: str):
        """
        Runs NLI inference on (premise, hypothesis).
        Returns: (label, confidence_score) where label is one of ENTAILMENT, CONTRADICTION, NEUTRAL.
        """
        try:
            # Inputs: Premise (Evidence) first, Hypothesis (Claim) second is standard for NLI
            inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).squeeze()
                
                predicted_class_id = torch.argmax(probs).item()
                confidence = probs[predicted_class_id].item()

                # Get label from model config
                # roberta-large-mnli config: {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'} usually, or similar
                # We trust id2label
                if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                    model_label = self.model.config.id2label[predicted_class_id].upper()
                else:
                    # Fallback for standard MNLI mapping if id2label is missing
                    # 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
                    labels = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
                    model_label = labels[predicted_class_id]

                # Map to required output format purely for safety
                if "ENTAILMENT" in model_label:
                    final_label = "ENTAILMENT"
                elif "CONTRADICTION" in model_label:
                    final_label = "CONTRADICTION"
                elif "NEUTRAL" in model_label:
                    final_label = "NEUTRAL"
                else:
                    final_label = "NEUTRAL" # Fallback
                
                return final_label, confidence

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None, 0.0
