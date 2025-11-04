import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class DimABSADataset(Dataset):
    """
    Dataset class for Dimensional Aspect-Based Sentiment Analysis
    Handles multilingual data across multiple domains
    """
    def __init__(
        self,
        data: Union[pd.DataFrame, List[Dict]],
        tokenizer: str = "xlm-roberta-base",
        max_length: int = 128,
        task: str = "DimASR"  # One of: DimASR, DimASTE, DimASQP
    ):
        self.data = data if isinstance(data, list) else data.to_dict('records')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.task = task
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        
        # Basic encoding for all tasks
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add VA scores
        if "valence" in item and "arousal" in item:
            encoding["va_scores"] = torch.tensor([item["valence"], item["arousal"]], dtype=torch.float)
        
        # Task-specific processing
        if self.task == "DimASR":
            # For predefined aspects
            if "aspect" in item and "aspect_indices" not in item:
                aspect = item["aspect"]
                # Find aspect indices in tokenized text
                aspect_encoding = self.tokenizer(
                    aspect,
                    add_special_tokens=False
                )
                aspect_tokens = aspect_encoding["input_ids"]
                
                # Find aspect tokens in the full text
                input_ids = encoding["input_ids"].tolist()
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
                encoding["aspect_indices"] = torch.tensor(aspect_indices, dtype=torch.long)
            elif "aspect_indices" in item:
                encoding["aspect_indices"] = torch.tensor(item["aspect_indices"], dtype=torch.long)
        
        elif self.task == "DimASTE":
            # For aspect-opinion extraction
            if "aspect_labels" in item and "opinion_labels" in item:
                encoding["aspect_labels"] = torch.tensor(
                    item["aspect_labels"][:self.max_length] + [0] * (self.max_length - len(item["aspect_labels"])),
                    dtype=torch.long
                )
                encoding["opinion_labels"] = torch.tensor(
                    item["opinion_labels"][:self.max_length] + [0] * (self.max_length - len(item["opinion_labels"])),
                    dtype=torch.long
                )
        
        elif self.task == "DimASQP":
            # For quadruplet prediction
            if "category" in item:
                encoding["category"] = torch.tensor(item["category"], dtype=torch.long)
            
            # Include aspect indices if available
            if "aspect_indices" in item:
                encoding["aspect_indices"] = torch.tensor(item["aspect_indices"], dtype=torch.long)
            
            # Include aspect and opinion labels if available
            if "aspect_labels" in item and "opinion_labels" in item:
                encoding["aspect_labels"] = torch.tensor(
                    item["aspect_labels"][:self.max_length] + [0] * (self.max_length - len(item["aspect_labels"])),
                    dtype=torch.long
                )
                encoding["opinion_labels"] = torch.tensor(
                    item["opinion_labels"][:self.max_length] + [0] * (self.max_length - len(item["opinion_labels"])),
                    dtype=torch.long
                )
        
        return encoding


def create_dataloaders(
    train_data,
    val_data=None,
    test_data=None,
    tokenizer="xlm-roberta-base",
    batch_size=16,
    max_length=128,
    task="DimASR"
):
    """
    Create DataLoader objects for training, validation, and testing
    
    Args:
        train_data: Training data (DataFrame or list of dicts)
        val_data: Validation data (DataFrame or list of dicts)
        test_data: Test data (DataFrame or list of dicts)
        tokenizer: Tokenizer name or path
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        task: Task type (DimASR, DimASTE, DimASQP)
        
    Returns:
        Dictionary of DataLoader objects
    """
    dataloaders = {}
    
    # Create training dataset and dataloader
    train_dataset = DimABSADataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        task=task
    )
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create validation dataset and dataloader if provided
    if val_data is not None:
        val_dataset = DimABSADataset(
            data=val_data,
            tokenizer=tokenizer,
            max_length=max_length,
            task=task
        )
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    # Create test dataset and dataloader if provided
    if test_data is not None:
        test_dataset = DimABSADataset(
            data=test_data,
            tokenizer=tokenizer,
            max_length=max_length,
            task=task
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    return dataloaders


def prepare_multilingual_data(data_path, languages=None, domains=None):
    """
    Load and prepare multilingual data from multiple domains
    
    Args:
        data_path: Path to data directory or file
        languages: List of language codes to include (None for all)
        domains: List of domains to include (None for all)
        
    Returns:
        DataFrame with prepared data
    """
    # Load data (assuming CSV format)
    df = pd.read_csv(data_path)
    
    # Filter by language if specified
    if languages:
        df = df[df['language'].isin(languages)]
    
    # Filter by domain if specified
    if domains:
        df = df[df['domain'].isin(domains)]
    
    return df


def create_aspect_labels(text, aspects, tokenizer):
    """
    Create BIO labels for aspect extraction
    
    Args:
        text: Input text
        aspects: List of aspect terms
        tokenizer: Tokenizer object
        
    Returns:
        List of aspect labels (0: O, 1: B-A, 2: I-A)
    """
    # Tokenize text
    tokens = tokenizer.tokenize(text)
    aspect_labels = [0] * len(tokens)  # Initialize with O labels
    
    for aspect in aspects:
        # Tokenize aspect
        aspect_tokens = tokenizer.tokenize(aspect)
        
        # Find aspect in tokenized text
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                # Mark first token as B-A
                aspect_labels[i] = 1
                # Mark remaining tokens as I-A
                for j in range(1, len(aspect_tokens)):
                    aspect_labels[i+j] = 2
                break
    
    return aspect_labels