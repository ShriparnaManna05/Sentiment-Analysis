import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DimABSAModel(nn.Module):
    """
    Dimensional Aspect-Based Sentiment Analysis Model
    Predicts Valence-Arousal scores for aspects in text
    """
    def __init__(
        self, 
        model_name="xlm-roberta-base", 
        num_labels=2,  # Valence and Arousal
        dropout_prob=0.1
    ):
        super(DimABSAModel, self).__init__()
        
        # Load pre-trained transformer model
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Regression head for VA prediction
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        aspect_indices=None,
        **kwargs
    ):
        # Get transformer outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Extract aspect representations
        if aspect_indices is not None:
            # For each example in batch, get the aspect token representations
            batch_size = sequence_output.shape[0]
            aspect_output = torch.zeros(
                batch_size, 
                self.config.hidden_size, 
                device=sequence_output.device
            )
            
            for i in range(batch_size):
                # Get indices for this example's aspect tokens
                indices = aspect_indices[i]
                # Filter out padding (-1)
                valid_indices = indices[indices != -1]
                if len(valid_indices) > 0:
                    # Average the representations of aspect tokens
                    aspect_output[i] = torch.mean(
                        sequence_output[i, valid_indices], dim=0
                    )
                else:
                    # If no aspect tokens, use CLS token
                    aspect_output[i] = sequence_output[i, 0]
        else:
            # If no aspect indices provided, use CLS token
            aspect_output = sequence_output[:, 0]
        
        # Apply dropout and classification
        aspect_output = self.dropout(aspect_output)
        logits = self.classifier(aspect_output)
        
        # Output format: [batch_size, 2] where each row is [valence, arousal]
        return logits


class DimASTEModel(DimABSAModel):
    """
    Dimensional Aspect Sentiment Triplet Extraction Model
    Extracts (Aspect, Opinion, VA) triplets from text
    """
    def __init__(self, model_name="xlm-roberta-base", dropout_prob=0.1):
        super(DimASTEModel, self).__init__(model_name, num_labels=2, dropout_prob=dropout_prob)
        
        # Additional layers for aspect and opinion extraction
        self.aspect_extractor = nn.Linear(self.config.hidden_size, 2)  # B-A, I-A tagging
        self.opinion_extractor = nn.Linear(self.config.hidden_size, 2)  # B-O, I-O tagging
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        # Get transformer outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Extract aspects and opinions
        aspect_logits = self.aspect_extractor(sequence_output)  # [batch_size, seq_len, 2]
        opinion_logits = self.opinion_extractor(sequence_output)  # [batch_size, seq_len, 2]
        
        # VA prediction using CLS token
        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        va_logits = self.classifier(pooled_output)  # [batch_size, 2]
        
        return {
            "aspect_logits": aspect_logits,
            "opinion_logits": opinion_logits,
            "va_logits": va_logits
        }


class DimASQPModel(DimASTEModel):
    """
    Dimensional Aspect Sentiment Quadruplet Prediction Model
    Predicts (Aspect, Category, Opinion, VA) quadruplets
    """
    def __init__(self, model_name="xlm-roberta-base", num_categories=10, dropout_prob=0.1):
        super(DimASQPModel, self).__init__(model_name, dropout_prob=dropout_prob)
        
        # Additional layer for category prediction
        self.category_classifier = nn.Linear(self.config.hidden_size, num_categories)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        aspect_indices=None,
        **kwargs
    ):
        # Get base model outputs
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get transformer outputs for category prediction
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Extract aspect representations for category prediction
        if aspect_indices is not None:
            # For each example in batch, get the aspect token representations
            batch_size = sequence_output.shape[0]
            aspect_output = torch.zeros(
                batch_size, 
                self.config.hidden_size, 
                device=sequence_output.device
            )
            
            for i in range(batch_size):
                # Get indices for this example's aspect tokens
                indices = aspect_indices[i]
                # Filter out padding (-1)
                valid_indices = indices[indices != -1]
                if len(valid_indices) > 0:
                    # Average the representations of aspect tokens
                    aspect_output[i] = torch.mean(
                        sequence_output[i, valid_indices], dim=0
                    )
                else:
                    # If no aspect tokens, use CLS token
                    aspect_output[i] = sequence_output[i, 0]
        else:
            # If no aspect indices provided, use CLS token
            aspect_output = sequence_output[:, 0]
        
        # Apply dropout and category classification
        aspect_output = self.dropout(aspect_output)
        category_logits = self.category_classifier(aspect_output)
        
        # Add category logits to output
        base_outputs["category_logits"] = category_logits
        
        return base_outputs