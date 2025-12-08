"""
Utility functions untuk load model dan preprocessing
"""

import joblib
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np


class ModelLoader:
    """Class untuk load dan manage model + vectorizer"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vectorizer = None
        self.model_info = {}
        
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load trained model dan vectorizer dari disk
        
        Returns:
            Tuple berisi (model, vectorizer)
        """
        model_path = self.model_dir / "model.pkl"
        vectorizer_path = self.model_dir / "vectorizer.pkl"
        
        # Validasi file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file tidak ditemukan: {vectorizer_path}")
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load vectorizer
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Load metrics info jika ada
        self._load_model_info()
        
        return self.model, self.vectorizer
    
    def _load_model_info(self):
        """Load informasi metrics dari metrics.json"""
        metrics_path = Path("metrics.json")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.model_info = json.load(f)
        else:
            # Default values jika file tidak ada
            self.model_info = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return informasi tentang model
        
        Returns:
            Dictionary berisi info model
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model belum di-load. Panggil load_model() terlebih dahulu")
        
        return {
            "model_type": self.model.__class__.__name__,
            "vectorizer_type": self.vectorizer.__class__.__name__,
            "max_features": getattr(self.vectorizer, 'max_features', 'N/A'),
            "test_accuracy": self.model_info.get("accuracy", 0.0),
            "test_f1_score": self.model_info.get("f1_score", 0.0),
            "test_precision": self.model_info.get("precision", 0.0),
            "test_recall": self.model_info.get("recall", 0.0),
            "trained_date": self.model_info.get("date", None),
            "model_version": "v1.0.0"
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment dari input text
        
        Args:
            text: Review text untuk diprediksi
            
        Returns:
            Dictionary berisi hasil prediksi
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model belum di-load")
        
        # Transform text ke TF-IDF features
        text_tfidf = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Prediction sudah berupa string "positive" atau "negative"
        sentiment = str(prediction)
        
        # Get confidence from probabilities based on sentiment
        if sentiment == "positive":
            confidence = float(probabilities[1])  # index 1 = positive
        else:
            confidence = float(probabilities[0])  # index 0 = negative
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": {
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            }
        }


def validate_model_files(model_dir: str = "models") -> bool:
    """
    Check apakah model files ada dan valid
    
    Args:
        model_dir: Directory tempat model disimpan
        
    Returns:
        True jika semua file ada, False otherwise
    """
    model_dir_path = Path(model_dir)
    required_files = ["model.pkl", "vectorizer.pkl"]
    
    for file in required_files:
        if not (model_dir_path / file).exists():
            return False
    
    return True
