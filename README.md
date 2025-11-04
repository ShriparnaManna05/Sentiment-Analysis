# Dimensional Aspect-Based Sentiment Analysis (DimABSA)

## Overview
This project implements a multilingual, multi-domain sentiment analysis system that integrates Valence-Arousal (VA) dimensional scoring with Aspect-Based Sentiment Analysis (ABSA). Instead of traditional categorical sentiment labels, this system predicts continuous VA values (1-9 scale) for aspects, opinions, and aspect categories.

## System Architecture
- **Input**: Text in multiple languages across various domains
- **Processing**: Transformer-based encoder (mBERT/XLM-R/DeBERTaV3) with regression head
- **Output**: Valence-Arousal scores for identified aspects

## Modules
- **DimASR**: Predicts VA scores for predefined aspects
- **DimASTE**: Extracts (Aspect, Opinion, VA) triplets from raw text
- **DimASQP**: Predicts (Aspect, Category, Opinion, VA) quadruplets

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from dimabsa import DimABSAPredictor

predictor = DimABSAPredictor(model_path="path/to/model")
results = predictor.predict("The food was amazing but the service was slow")
# Output: [(food, 8.2#6.7), (service, 3.9#4.5)]
```

## Training
```bash
python train.py --config configs/base_config.json
```

## Evaluation
```bash
python evaluate.py --model_path models/dimabsa_model.pt --test_data data/test.csv
```

## Project Structure
```
dimabsa/
├── data/                  # Dataset files
├── models/                # Saved model checkpoints
├── src/
│   ├── preprocessing/     # Data preprocessing modules
│   ├── models/            # Model architecture definitions
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation metrics and scripts
│   └── visualization/     # Result visualization tools
├── configs/               # Configuration files
├── notebooks/             # Exploratory notebooks
├── scripts/               # Utility scripts
└── results/               # Evaluation results and visualizations
```

## Applications
- Emotion-aware chatbots
- Cross-lingual sentiment tracking
- Mental health text analysis
- Multimodal affective computing (future extension)