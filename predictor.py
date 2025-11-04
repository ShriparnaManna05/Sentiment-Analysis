import torch
import numpy as np
from transformers import AutoTokenizer
import re
import spacy
from typing import Dict, List, Tuple, Optional, Union

class DimABSAPredictor:
    """
    Predictor class for Dimensional Aspect-Based Sentiment Analysis
    """
    def __init__(
        self,
        model,
        tokenizer="xlm-roberta-base",
        device=None,
        task="DimASR",
        spacy_model="en_core_web_sm"
    ):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Load spaCy model for aspect extraction
        try:
            self.nlp = spacy.load(spacy_model)
        except:
            print(f"SpaCy model {spacy_model} not found. Using default English model.")
            self.nlp = spacy.load("en_core_web_sm")
    
    def predict(self, text, aspects=None):
        """
        Predict VA scores for aspects in the given text
        
        Args:
            text: Input text
            aspects: List of aspects to predict VA scores for (optional)
                    If not provided, aspects will be extracted automatically
        
        Returns:
            List of (aspect, valence, arousal) tuples
        """
        # Extract aspects if not provided
        if aspects is None:
            aspects = self.extract_aspects(text)
        
        results = []
        
        for aspect in aspects:
            # Encode text and aspect
            encoding = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Find aspect indices
            aspect_indices = self.find_aspect_indices(text, aspect, encoding)
            
            if aspect_indices:
                encoding["aspect_indices"] = torch.tensor([aspect_indices], device=self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoding)
                
                if isinstance(outputs, dict) and "va_logits" in outputs:
                    va_scores = outputs["va_logits"][0].cpu().numpy()
                else:
                    va_scores = outputs[0].cpu().numpy()
            
            # Format results
            valence, arousal = va_scores
            results.append((aspect, valence, arousal))
        
        return results
    
    def extract_aspects(self, text):
        """
        Extract aspects from text using spaCy
        
        Args:
            text: Input text
            
        Returns:
            List of extracted aspects
        """
        doc = self.nlp(text)
        aspects = []
        
        # Extract noun chunks as potential aspects
        for chunk in doc.noun_chunks:
            aspects.append(chunk.text)
        
        return aspects
    
    def find_aspect_indices(self, text, aspect, encoding):
        """
        Find indices of aspect tokens in the encoded text
        
        Args:
            text: Input text
            aspect: Aspect to find
            encoding: Tokenized encoding
            
        Returns:
            List of aspect token indices
        """
        # Tokenize aspect
        aspect_encoding = self.tokenizer(
            aspect,
            add_special_tokens=False
        )
        aspect_tokens = aspect_encoding["input_ids"]
        
        # Find aspect tokens in the full text
        input_ids = encoding["input_ids"][0].tolist()
        aspect_indices = []
        
        # Simple substring matching (can be improved with more sophisticated methods)
        for i in range(len(input_ids) - len(aspect_tokens) + 1):
            if input_ids[i:i+len(aspect_tokens)] == aspect_tokens:
                aspect_indices.extend(list(range(i, i+len(aspect_tokens))))
                break
        
        # If aspect not found, use CLS token
        if not aspect_indices:
            aspect_indices = [0]
        
        # Pad aspect indices to fixed length
        max_aspect_len = 10
        aspect_indices = aspect_indices[:max_aspect_len]
        aspect_indices = aspect_indices + [-1] * (max_aspect_len - len(aspect_indices))
        
        return aspect_indices
    
    def format_output(self, results):
        """
        Format results as a string
        
        Args:
            results: List of (aspect, valence, arousal) tuples
            
        Returns:
            Formatted string
        """
        output = []
        
        for aspect, valence, arousal in results:
            output.append(f"({aspect}, {valence:.1f}#{arousal:.1f})")
        
        return " ".join(output)
    
    def predict_and_format(self, text, aspects=None):
        """
        Predict VA scores and format the output
        
        Args:
            text: Input text
            aspects: List of aspects to predict VA scores for (optional)
            
        Returns:
            Formatted string of results
        """
        results = self.predict(text, aspects)
        return self.format_output(results)


# Example usage
if __name__ == "__main__":
    from src.models.model import DimABSAModel
    
    # Load model
    model = DimABSAModel()
    model.load_state_dict(torch.load("models/dimabsa_model.pt"))
    
    # Create predictor
    predictor = DimABSAPredictor(model)
    
    # Predict
    text = "The food was amazing but the service was slow"
    result = predictor.predict_and_format(text)
    print(result)  # Expected: (food, 8.2#6.7) (service, 3.9#4.5)