import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union


class DimABSAVisualizer:
    """
    Visualization tools for Dimensional Aspect-Based Sentiment Analysis
    """
    def __init__(self, output_dir="./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_va_distribution(self, predictions, save_path=None):
        """
        Plot the distribution of Valence-Arousal predictions
        
        Args:
            predictions: List of (aspect, valence, arousal) tuples
            save_path: Path to save the plot
        """
        # Extract valence and arousal values
        valence = [p[1] for p in predictions]
        arousal = [p[2] for p in predictions]
        aspects = [p[0] for p in predictions]
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Plot points
        scatter = plt.scatter(valence, arousal, alpha=0.7)
        
        # Add labels for some points
        for i, aspect in enumerate(aspects):
            plt.annotate(aspect, (valence[i], arousal[i]), fontsize=8)
        
        # Add quadrant labels
        plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
        
        plt.text(2, 7.5, "Negative\nHigh Arousal", ha='center')
        plt.text(8, 7.5, "Positive\nHigh Arousal", ha='center')
        plt.text(2, 2.5, "Negative\nLow Arousal", ha='center')
        plt.text(8, 2.5, "Positive\nLow Arousal", ha='center')
        
        # Set labels and title
        plt.xlabel("Valence (1-9)")
        plt.ylabel("Arousal (1-9)")
        plt.title("Valence-Arousal Distribution")
        
        # Set axis limits
        plt.xlim(1, 9)
        plt.ylim(1, 9)
        
        # Add grid
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_language_comparison(self, results, metric="pearson_r", save_path=None):
        """
        Plot comparison of results across languages
        
        Args:
            results: Dictionary mapping languages to metric values
            metric: Metric to plot
            save_path: Path to save the plot
        """
        # Create dataframe
        df = pd.DataFrame({
            "Language": list(results.keys()),
            metric: [results[lang][metric] for lang in results.keys()]
        })
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Language", y=metric, data=df)
        
        # Set labels and title
        plt.xlabel("Language")
        plt.ylabel(metric)
        plt.title(f"{metric} by Language")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_domain_comparison(self, results, metric="pearson_r", save_path=None):
        """
        Plot comparison of results across domains
        
        Args:
            results: Dictionary mapping domains to metric values
            metric: Metric to plot
            save_path: Path to save the plot
        """
        # Create dataframe
        df = pd.DataFrame({
            "Domain": list(results.keys()),
            metric: [results[domain][metric] for domain in results.keys()]
        })
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Domain", y=metric, data=df)
        
        # Set labels and title
        plt.xlabel("Domain")
        plt.ylabel(metric)
        plt.title(f"{metric} by Domain")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_circumplex_model(self, save_path=None):
        """
        Create a visualization of Russell's Circumplex Model of Affect
        
        Args:
            save_path: Path to save the plot
        """
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Create circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black')
        plt.gca().add_patch(circle)
        
        # Add axes
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add emotion labels
        emotions = {
            (0.7, 0.7): "Happy",
            (0.9, 0): "Pleased",
            (0.7, -0.7): "Relaxed",
            (0, -0.9): "Calm",
            (-0.7, -0.7): "Sad",
            (-0.9, 0): "Displeased",
            (-0.7, 0.7): "Stressed",
            (0, 0.9): "Excited"
        }
        
        for (x, y), emotion in emotions.items():
            plt.text(x, y, emotion, ha='center', va='center', fontsize=12)
        
        # Add dimension labels
        plt.text(1.1, 0, "Valence +", ha='left', va='center', fontsize=14)
        plt.text(-1.1, 0, "Valence -", ha='right', va='center', fontsize=14)
        plt.text(0, 1.1, "Arousal +", ha='center', va='bottom', fontsize=14)
        plt.text(0, -1.1, "Arousal -", ha='center', va='top', fontsize=14)
        
        # Set equal aspect ratio
        plt.axis('equal')
        
        # Set limits
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        
        # Remove ticks
        plt.xticks([])
        plt.yticks([])
        
        # Add title
        plt.title("Russell's Circumplex Model of Affect", fontsize=16)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_dashboard(self, language_results, domain_results):
        """
        Create a comprehensive performance dashboard
        
        Args:
            language_results: Dictionary of metrics for each language
            domain_results: Dictionary of metrics for each domain
        """
        # Create language comparison plots
        self.plot_language_comparison(
            language_results, 
            metric="pearson_r",
            save_path=os.path.join(self.output_dir, "language_pearson.png")
        )
        
        self.plot_language_comparison(
            language_results, 
            metric="mae",
            save_path=os.path.join(self.output_dir, "language_mae.png")
        )
        
        # Create domain comparison plots
        self.plot_domain_comparison(
            domain_results, 
            metric="pearson_r",
            save_path=os.path.join(self.output_dir, "domain_pearson.png")
        )
        
        self.plot_domain_comparison(
            domain_results, 
            metric="mae",
            save_path=os.path.join(self.output_dir, "domain_mae.png")
        )
        
        # Create circumplex model
        self.create_circumplex_model(
            save_path=os.path.join(self.output_dir, "circumplex_model.png")
        )
        
        # Create HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DimABSA Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; }}
                .row {{ display: flex; flex-wrap: wrap; }}
                .col {{ flex: 1; margin: 10px; min-width: 300px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Dimensional Aspect-Based Sentiment Analysis Dashboard</h1>
                
                <div class="section">
                    <h2>Circumplex Model of Affect</h2>
                    <div class="row">
                        <div class="col">
                            <img src="circumplex_model.png" alt="Circumplex Model">
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Performance by Language</h2>
                    <div class="row">
                        <div class="col">
                            <img src="language_pearson.png" alt="Pearson Correlation by Language">
                        </div>
                        <div class="col">
                            <img src="language_mae.png" alt="MAE by Language">
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Performance by Domain</h2>
                    <div class="row">
                        <div class="col">
                            <img src="domain_pearson.png" alt="Pearson Correlation by Domain">
                        </div>
                        <div class="col">
                            <img src="domain_mae.png" alt="MAE by Domain">
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML dashboard
        with open(os.path.join(self.output_dir, "dashboard.html"), "w") as f:
            f.write(html_content)