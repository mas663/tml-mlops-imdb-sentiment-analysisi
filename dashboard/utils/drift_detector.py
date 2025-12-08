"""
Drift Detection utility menggunakan Evidently
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Warning: Evidently not installed")

from dashboard.utils.db_manager import get_db_manager


def load_reference_data(sample_size: int = 1000) -> pd.DataFrame:
    """
    Load training data sebagai reference
    
    Args:
        sample_size: Number of samples to load
    
    Returns:
        DataFrame dengan reference data
    """
    try:
        # Load from training data
        train_df = pd.read_csv('data/processed/train.csv')
        
        # Sample jika terlalu besar
        if len(train_df) > sample_size:
            train_df = train_df.sample(n=sample_size, random_state=42)
        
        # Prepare columns
        reference_df = pd.DataFrame({
            'text': train_df['review_clean'],
            'sentiment': train_df['sentiment'],
            'text_length': train_df['review_clean'].str.len()
        })
        
        return reference_df
        
    except Exception as e:
        print(f"Error loading reference data: {e}")
        return pd.DataFrame()


def load_production_data() -> pd.DataFrame:
    """
    Load production data dari database
    
    Returns:
        DataFrame dengan production data
    """
    try:
        db = get_db_manager()
        prod_df = db.get_all_predictions()
        
        if prod_df.empty:
            return pd.DataFrame()
        
        # Prepare columns untuk comparison
        production_df = pd.DataFrame({
            'text': prod_df['text'],
            'sentiment': prod_df['sentiment'],
            'text_length': prod_df['text_length']
        })
        
        return production_df
        
    except Exception as e:
        print(f"Error loading production data: {e}")
        return pd.DataFrame()


def calculate_drift_score(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> float:
    """
    Calculate simple drift score based on distributions
    
    Returns:
        Drift score (0 = no drift, 1 = maximum drift)
    """
    try:
        # Compare text length distributions
        ref_mean = reference_df['text_length'].mean()
        prod_mean = production_df['text_length'].mean()
        length_diff = abs(ref_mean - prod_mean) / ref_mean
        
        # Compare sentiment distributions
        ref_pos_ratio = (reference_df['sentiment'] == 'positive').mean()
        prod_pos_ratio = (production_df['sentiment'] == 'positive').mean()
        sentiment_diff = abs(ref_pos_ratio - prod_pos_ratio)
        
        # Combined drift score
        drift_score = (length_diff + sentiment_diff) / 2
        
        return min(drift_score, 1.0)
        
    except Exception as e:
        print(f"Error calculating drift score: {e}")
        return 0.0


def get_drift_status(drift_score: float) -> Tuple[str, str]:
    """
    Get drift status and color based on score
    
    Returns:
        Tuple of (status_text, color)
    """
    if drift_score < 0.1:
        return ("✅ NO DRIFT DETECTED", "green")
    elif drift_score < 0.3:
        return ("⚠️ LOW DRIFT", "orange")
    elif drift_score < 0.5:
        return ("⚠️ MODERATE DRIFT", "orange")
    else:
        return ("❌ HIGH DRIFT DETECTED", "red")


def generate_drift_report(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> str:
    """
    Generate Evidently drift report
    
    Returns:
        Path to generated HTML report
    """
    if not EVIDENTLY_AVAILABLE:
        return None
    
    try:
        # Create report
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        # Run report
        report.run(
            reference_data=reference_df,
            current_data=production_df
        )
        
        # Save to file
        report_path = Path("dashboard/data/drift_report.html")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(report_path))
        
        return str(report_path)
        
    except Exception as e:
        print(f"Error generating Evidently report: {e}")
        return None


def get_comparison_stats(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comparison statistics between reference and production data
    
    Returns:
        Dictionary dengan comparison metrics
    """
    try:
        stats = {
            'reference': {
                'count': len(reference_df),
                'text_length_mean': reference_df['text_length'].mean(),
                'text_length_std': reference_df['text_length'].std(),
                'positive_ratio': (reference_df['sentiment'] == 'positive').mean(),
            },
            'production': {
                'count': len(production_df),
                'text_length_mean': production_df['text_length'].mean(),
                'text_length_std': production_df['text_length'].std(),
                'positive_ratio': (production_df['sentiment'] == 'positive').mean(),
            },
            'difference': {
                'text_length_diff': abs(reference_df['text_length'].mean() - production_df['text_length'].mean()),
                'text_length_diff_pct': abs(reference_df['text_length'].mean() - production_df['text_length'].mean()) / reference_df['text_length'].mean() * 100,
                'sentiment_diff': abs((reference_df['sentiment'] == 'positive').mean() - (production_df['sentiment'] == 'positive').mean()) * 100,
            }
        }
        
        return stats
        
    except Exception as e:
        print(f"Error calculating comparison stats: {e}")
        return {}
