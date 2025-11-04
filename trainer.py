import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import json
from typing import Dict, List, Tuple, Optional, Union

class PearsonCorrelationLoss(nn.Module):
    """
    Loss function that combines MSE with Pearson correlation
    """
    def __init__(self, alpha=0.5):
        super(PearsonCorrelationLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # MSE Loss
        mse_loss = self.mse(pred, target)
        
        # Pearson Correlation Loss
        vx = pred - torch.mean(pred, dim=0)
        vy = target - torch.mean(target, dim=0)
        
        pearson_loss = 0
        for i in range(pred.shape[1]):  # For each dimension (valence, arousal)
            cost = torch.sum(vx[:, i] * vy[:, i]) / (torch.sqrt(torch.sum(vx[:, i] ** 2)) * torch.sqrt(torch.sum(vy[:, i] ** 2)) + 1e-8)
            pearson_loss += 1 - cost  # 1 - correlation to minimize
        
        pearson_loss /= pred.shape[1]  # Average across dimensions
        
        # Combined loss
        return self.alpha * mse_loss + (1 - self.alpha) * pearson_loss


class DimABSATrainer:
    """
    Trainer class for Dimensional Aspect-Based Sentiment Analysis models
    """
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        test_dataloader=None,
        lr=2e-5,
        alpha=0.5,  # Weight for MSE in combined loss
        device=None,
        output_dir="./models"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr = lr
        self.alpha = alpha
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # Initialize loss function
        self.criterion = PearsonCorrelationLoss(alpha=self.alpha)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train(self, epochs=10, save_best=True):
        """
        Train the model for the specified number of epochs
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                
                # Calculate loss
                if isinstance(outputs, dict):
                    # For DimASTE and DimASQP models
                    loss = 0
                    if "va_logits" in outputs and "va_scores" in batch:
                        loss += self.criterion(outputs["va_logits"], batch["va_scores"])
                    
                    if "aspect_logits" in outputs and "aspect_labels" in batch:
                        aspect_loss = nn.CrossEntropyLoss()(
                            outputs["aspect_logits"].view(-1, outputs["aspect_logits"].shape[-1]),
                            batch["aspect_labels"].view(-1)
                        )
                        loss += aspect_loss
                    
                    if "opinion_logits" in outputs and "opinion_labels" in batch:
                        opinion_loss = nn.CrossEntropyLoss()(
                            outputs["opinion_logits"].view(-1, outputs["opinion_logits"].shape[-1]),
                            batch["opinion_labels"].view(-1)
                        )
                        loss += opinion_loss
                    
                    if "category_logits" in outputs and "category" in batch:
                        category_loss = nn.CrossEntropyLoss()(outputs["category_logits"], batch["category"])
                        loss += category_loss
                else:
                    # For DimASR model
                    loss = self.criterion(outputs, batch["va_scores"])
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(self.train_dataloader)
            
            # Validation
            val_loss = 0
            val_metrics = {}
            
            if self.val_dataloader:
                val_metrics = self.evaluate(self.val_dataloader)
                val_loss = val_metrics["loss"]
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Pearson r: {val_metrics['pearson_r']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(self.output_dir, "best_model.pt"))
                    print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model.pt"))
        
        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics
        }
    
    def evaluate(self, dataloader):
        """
        Evaluate the model on the given dataloader
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Get predictions and targets
                if isinstance(outputs, dict):
                    # For DimASTE and DimASQP models
                    if "va_logits" in outputs and "va_scores" in batch:
                        preds = outputs["va_logits"]
                        targets = batch["va_scores"]
                        loss = self.criterion(preds, targets)
                        total_loss += loss.item()
                        all_preds.append(preds.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())
                else:
                    # For DimASR model
                    preds = outputs
                    targets = batch["va_scores"]
                    loss = self.criterion(preds, targets)
                    total_loss += loss.item()
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        
        # Concatenate predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = {
            "loss": avg_loss,
            "pearson_r": self.calculate_pearson_correlation(all_preds, all_targets),
            "mae": mean_absolute_error(all_targets, all_preds)
        }
        
        return metrics
    
    def calculate_pearson_correlation(self, preds, targets):
        """
        Calculate Pearson correlation coefficient between predictions and targets
        """
        correlations = []
        
        for i in range(preds.shape[1]):  # For each dimension (valence, arousal)
            correlation = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
            correlations.append(correlation)
        
        return np.mean(correlations)
    
    def save_model(self, path):
        """
        Save model to the specified path
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """
        Load model from the specified path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])