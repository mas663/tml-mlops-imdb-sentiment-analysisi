"""
Logger utility untuk log predictions
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path untuk import db_manager
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils.db_manager import get_db_manager


def log_prediction(
    text: str,
    sentiment: str,
    confidence: float,
    probabilities: Dict[str, float],
    response_time: float = None
) -> int:
    """
    Log prediction ke database
    
    Args:
        text: Input text
        sentiment: Predicted sentiment (positive/negative)
        confidence: Confidence score
        probabilities: Dict dengan prob_negative dan prob_positive
        response_time: Response time dalam ms (optional)
    
    Returns:
        ID dari record yang di-insert
    """
    db = get_db_manager()
    
    record_id = db.insert_prediction(
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        prob_negative=probabilities.get('negative', 0.0),
        prob_positive=probabilities.get('positive', 0.0),
        text_length=len(text),
        response_time=response_time
    )
    
    return record_id


def get_prediction_stats() -> Dict[str, Any]:
    """Get statistics dari semua predictions"""
    db = get_db_manager()
    return db.get_stats()


def get_recent_predictions(n: int = 20):
    """Get N recent predictions"""
    db = get_db_manager()
    return db.get_recent_predictions(n)


def get_all_predictions():
    """Get all predictions"""
    db = get_db_manager()
    return db.get_all_predictions()
