"""
Download script for DimABSA project resources.
This script downloads all necessary language models and resources.
"""
import os
import sys
import nltk
import spacy
from transformers import AutoTokenizer, AutoModel
import torch

def download_nltk_resources():
    """Download required NLTK resources."""
    print("Downloading NLTK resources...")
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        nltk.download(resource)
    print("✓ NLTK resources downloaded successfully")

def download_spacy_models():
    """Download required spaCy language models."""
    print("Downloading spaCy language models...")
    # Core languages for the multilingual DimABSA system
    languages = ['en_core_web_sm', 'es_core_news_sm']
    
    for lang in languages:
        print(f"Downloading {lang}...")
        os.system(f"{sys.executable} -m spacy download {lang}")
    
    print("✓ spaCy models downloaded successfully")

def download_transformer_models():
    """Download pretrained transformer models."""
    print("Downloading transformer models...")
    
    # XLM-RoBERTa is a good multilingual model for our DimABSA system
    model_name = 'xlm-roberta-base'
    print(f"Downloading {model_name}...")
    
    # This will cache the model locally
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print(f"✓ {model_name} downloaded successfully")
    
    # Save a small test tensor to verify PyTorch is working
    test_tensor = torch.tensor([1.0, 2.0, 3.0])
    torch.save(test_tensor, "torch_test.pt")
    print("✓ PyTorch test successful")

def main():
    """Main function to download all resources."""
    print("Starting download of resources for DimABSA project...")
    
    download_nltk_resources()
    download_spacy_models()
    download_transformer_models()
    
    print("\nAll resources downloaded successfully!")
    print("The DimABSA project is ready to use.")

if __name__ == "__main__":
    main()