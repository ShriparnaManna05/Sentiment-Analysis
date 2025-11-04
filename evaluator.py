import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Union


class DimABSAEvaluator:
    """
    Evaluator class for Dimensional Aspect-Based Sentiment Analysis models
    """
    def __init__(self, model, dataloader, device=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
    
    def evaluate(self):
        """
        Evaluate the model on the given dataloader
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.dataloader:
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
                        all_preds.append(preds.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())
                else:
                    # For DimASR model
                    preds = outputs
                    targets = batch["va_scores"]
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
        
        # Concatenate predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = {
            "pearson_r": self.calculate_pearson_correlation(all_preds, all_targets),
            "mae": mean_absolute_error(all_targets, all_preds)
        }
        
        # Calculate per-dimension metrics
        for i, dim in enumerate(["valence", "arousal"]):
            metrics[f"{dim}_pearson_r"] = np.corrcoef(all_preds[:, i], all_targets[:, i])[0, 1]
            metrics[f"{dim}_mae"] = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        
        return metrics, all_preds, all_targets
    
    def calculate_pearson_correlation(self, preds, targets):
        """
        Calculate Pearson correlation coefficient between predictions and targets
        """
        correlations = []
        
        for i in range(preds.shape[1]):  # For each dimension (valence, arousal)
            correlation = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
            correlations.append(correlation)
        
        return np.mean(correlations)
    
    def visualize_results(self, preds, targets, save_path=None):
        """
        Visualize the results using scatter plots
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Valence scatter plot
        sns.scatterplot(x=targets[:, 0], y=preds[:, 0], ax=axes[0])
        axes[0].set_title("Valence: Predicted vs. True")
        axes[0].set_xlabel("True Valence")
        axes[0].set_ylabel("Predicted Valence")
        axes[0].plot([1, 9], [1, 9], 'r--')  # Diagonal line
        
        # Arousal scatter plot
        sns.scatterplot(x=targets[:, 1], y=preds[:, 1], ax=axes[1])
        axes[1].set_title("Arousal: Predicted vs. True")
        axes[1].set_xlabel("True Arousal")
        axes[1].set_ylabel("Predicted Arousal")
        axes[1].plot([1, 9], [1, 9], 'r--')  # Diagonal line
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def evaluate_by_language(self, language_data):
        """
        Evaluate the model separately for each language
        
        Args:
            language_data: Dictionary mapping language codes to dataloaders
            
        Returns:
            Dictionary of metrics for each language
        """
        results = {}
        
        for lang, dataloader in language_data.items():
            self.dataloader = dataloader
            metrics, _, _ = self.evaluate()
            results[lang] = metrics
        
        return results
    
    def evaluate_by_domain(self, domain_data):
        """
        Evaluate the model separately for each domain
        
        Args:
            domain_data: Dictionary mapping domain names to dataloaders
            
        Returns:
            Dictionary of metrics for each domain
        """
        results = {}
        
        for domain, dataloader in domain_data.items():
            self.dataloader = dataloader
            metrics, _, _ = self.evaluate()
            results[domain] = metrics
        
        return results
    
    def create_performance_dashboard(self, language_results, domain_results, save_dir="./results"):
        """
        Create a visual performance dashboard comparing models across languages and domains
        
        Args:
            language_results: Dictionary of metrics for each language
            domain_results: Dictionary of metrics for each domain
            save_dir: Directory to save the dashboard
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create language comparison dataframe
        lang_data = []
        for lang, metrics in language_results.items():
            lang_data.append({
                "Language": lang,
                "Pearson r": metrics["pearson_r"],
                "MAE": metrics["mae"],
                "Valence r": metrics["valence_pearson_r"],
                "Arousal r": metrics["arousal_pearson_r"]
            })
        
        lang_df = pd.DataFrame(lang_data)
        
        # Create domain comparison dataframe
        domain_data = []
        for domain, metrics in domain_results.items():
            domain_data.append({
                "Domain": domain,
                "Pearson r": metrics["pearson_r"],
                "MAE": metrics["mae"],
                "Valence r": metrics["valence_pearson_r"],
                "Arousal r": metrics["arousal_pearson_r"]
            })
        
        domain_df = pd.DataFrame(domain_data)
        
        # Create language comparison plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Language", y="Pearson r", data=lang_df)
        plt.title("Pearson Correlation by Language")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "language_pearson.png"))
        plt.close()
        
        # Create domain comparison plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Domain", y="Pearson r", data=domain_df)
        plt.title("Pearson Correlation by Domain")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "domain_pearson.png"))
        plt.close()
        
        # Create MAE comparison plot
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        sns.barplot(x="Language", y="MAE", data=lang_df)
        plt.title("Mean Absolute Error by Language")
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        sns.barplot(x="Domain", y="MAE", data=domain_df)
        plt.title("Mean Absolute Error by Domain")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mae_comparison.png"))
        plt.close()
        
        # Save data to CSV
        lang_df.to_csv(os.path.join(save_dir, "language_results.csv"), index=False)
        domain_df.to_csv(os.path.join(save_dir, "domain_results.csv"), index=False)
        
        return {
            "language_results": lang_df,
            "domain_results": domain_df
        }