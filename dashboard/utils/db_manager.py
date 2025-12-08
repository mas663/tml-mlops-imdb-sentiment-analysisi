"""
Database manager untuk menyimpan prediction logs
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class DatabaseManager:
    """Manager untuk SQLite database operations"""
    
    def __init__(self, db_path: str = "dashboard/data/predictions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database dan create table jika belum ada"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                prob_negative REAL NOT NULL,
                prob_positive REAL NOT NULL,
                text_length INTEGER NOT NULL,
                response_time REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def insert_prediction(
        self, 
        text: str,
        sentiment: str,
        confidence: float,
        prob_negative: float,
        prob_positive: float,
        text_length: int,
        response_time: Optional[float] = None
    ) -> int:
        """
        Insert prediction ke database
        
        Returns:
            ID dari record yang baru di-insert
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, text, sentiment, confidence, prob_negative, prob_positive, text_length, response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, text, sentiment, confidence, prob_negative, prob_positive, text_length, response_time))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return record_id
    
    def get_all_predictions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get semua predictions dari database
        
        Args:
            limit: Maximum number of records (None untuk semua)
        
        Returns:
            DataFrame dengan semua predictions
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM predictions ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_recent_predictions(self, n: int = 20) -> pd.DataFrame:
        """Get N predictions terbaru"""
        return self.get_all_predictions(limit=n)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics dari semua predictions
        
        Returns:
            Dictionary berisi statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        
        if total == 0:
            conn.close()
            return {
                "total_predictions": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "avg_confidence": 0.0,
                "avg_response_time": 0.0,
                "avg_text_length": 0.0
            }
        
        # Sentiment counts
        cursor.execute("SELECT sentiment, COUNT(*) FROM predictions GROUP BY sentiment")
        sentiment_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        
        # Average metrics
        cursor.execute("""
            SELECT 
                AVG(confidence) as avg_conf,
                AVG(response_time) as avg_time,
                AVG(text_length) as avg_len
            FROM predictions
        """)
        avgs = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_predictions": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": positive_count / total if total > 0 else 0.0,
            "negative_ratio": negative_count / total if total > 0 else 0.0,
            "avg_confidence": avgs[0] if avgs[0] else 0.0,
            "avg_response_time": avgs[1] if avgs[1] else 0.0,
            "avg_text_length": avgs[2] if avgs[2] else 0.0
        }
    
    def get_predictions_by_date(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get predictions filtered by date
        
        Args:
            start_date: ISO format date string (YYYY-MM-DD)
        
        Returns:
            DataFrame dengan filtered predictions
        """
        conn = sqlite3.connect(self.db_path)
        
        if start_date:
            query = f"SELECT * FROM predictions WHERE DATE(timestamp) >= '{start_date}' ORDER BY timestamp DESC"
        else:
            query = "SELECT * FROM predictions ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_prediction_count(self) -> int:
        """Get total count of predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def clear_all_predictions(self):
        """Clear all predictions (use with caution!)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()


# Singleton instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get singleton instance of DatabaseManager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
