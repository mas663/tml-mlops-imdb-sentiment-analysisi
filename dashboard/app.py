"""
MLOps Sentiment Analysis - Monitoring Dashboard
Main page dengan overview project
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dashboard.utils.db_manager import get_db_manager
import json

# Page config
st.set_page_config(
    page_title="MLOps Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    .stat-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .info-box h4 {
        color: #0d47a1;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    .info-box ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    .info-box li {
        margin-bottom: 0.5rem;
        color: #2c3e50;
        line-height: 1.6;
    }
    .info-box b {
        color: #1565c0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎬 MLOps Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IMDB 50K Movie Reviews - Complete MLOps Pipeline</div>', unsafe_allow_html=True)

st.markdown("---")

# Quick Stats Section
st.subheader("📊 System Overview")

# Get database stats
db = get_db_manager()
stats = db.get_stats()

# Load model metrics
try:
    with open('metrics.json', 'r') as f:
        model_metrics = json.load(f)
        model_accuracy = model_metrics.get('accuracy', 0.0) * 100
except:
    model_accuracy = 89.97

# Stats cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="metric-card">
            <div class="stat-value">17</div>
            <div class="stat-label">📊 Experiments</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-value">{model_accuracy:.1f}%</div>
            <div class="stat-label">✅ Best Accuracy</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-value">{stats['total_predictions']:,}</div>
            <div class="stat-label">🎯 Total Predictions</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    if stats['total_predictions'] > 0:
        avg_conf = stats['avg_confidence'] * 100
    else:
        avg_conf = 0.0
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-value">{avg_conf:.1f}%</div>
            <div class="stat-label">💪 Avg Confidence</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Architecture Section
st.subheader("🏗️ System Architecture")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("""
        <div class="info-box">
            <h4>📦 Tech Stack</h4>
            <ul>
                <li><b>Data Versioning:</b> Git LFS (130MB dataset)</li>
                <li><b>Experiment Tracking:</b> MLflow (17 runs)</li>
                <li><b>Orchestration:</b> Prefect (3-task pipeline)</li>
                <li><b>Model Serving:</b> FastAPI + Uvicorn</li>
                <li><b>Monitoring:</b> Streamlit + Evidently</li>
                <li><b>Containerization:</b> Docker Compose</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
        <div class="info-box">
            <h4>🤖 Model Details</h4>
            <ul>
                <li><b>Algorithm:</b> Logistic Regression</li>
                <li><b>Vectorization:</b> TF-IDF (20,000 features)</li>
                <li><b>Training Data:</b> 40,000 reviews</li>
                <li><b>Test Data:</b> 10,000 reviews</li>
                <li><b>Accuracy:</b> 89.97%</li>
                <li><b>F1-Score:</b> 90.02%</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Pipeline Flow
st.subheader("🔄 ML Pipeline Flow")

st.markdown("""
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Raw Data  │ ───► │ Preprocessing│ ───► │   Training  │ ───► │  Evaluation  │ ───► │  Deployment     │
│  (Git LFS)  │      │   (Prefect)  │      │   (MLflow)  │      │  (Metrics)   │      │   (FastAPI)     │
└─────────────┘      └──────────────┘      └─────────────┘      └──────────────┘      └─────────────────┘
                                                                                                  │
                                                                                                  ▼
                                                                                        ┌────────────────-─┐
                                                                                        │   Monitoring     │
                                                                                        │  (This Dashboard)│
                                                                                        └────────────────-─┘
```
""")

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Guide
st.subheader("🧭 Navigation Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
        **🎯 Live Prediction**
        
        Test the model dengan input review sendiri. 
        Lihat hasil prediksi real-time dengan confidence score.
        
        Setiap prediction akan otomatis logged untuk monitoring.
    """)

with col2:
    st.info("""
        **📊 Performance Monitor**
        
        Dashboard untuk monitoring performa model production.
        Lihat statistics, charts, dan recent predictions.
        
        Track total predictions, sentiment distribution, dan confidence trends.
    """)

with col3:
    st.info("""
        **⚠️ Data Drift Detection**
        
        Monitoring data drift antara training dan production data.
        Deteksi perubahan pola data input.
        
        Alert system untuk retraining recommendations.
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><b>MLOps Sentiment Analysis System</b> | Built with FastAPI, Streamlit, MLflow, Prefect | Version 1.0.0</p>
        <p>5026221134_TML[A] | 2025</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📌 Quick Links")
    
    import os
    # Get API URL from environment or use localhost for local dev
    api_base = os.getenv('API_URL', 'http://localhost:8000')
    
    st.markdown("### 🔗 Services")
    st.markdown("- [MLflow UI](http://localhost:5000) - Experiment Tracking")
    st.markdown("- [Prefect UI](http://localhost:4200) - Pipeline Orchestration")
    st.markdown(f"- [API Docs]({api_base}/docs) - FastAPI Swagger")
    st.markdown(f"- [API Health]({api_base}/health) - Health Check")
    
    st.markdown("---")
    
    st.markdown("### 📊 Current Status")
    
    # Check if services are running
    import requests
    
    # Check API
    try:
        response = requests.get(f"{api_base}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API: Running")
        else:
            st.error("❌ API: Error")
    except:
        st.warning("⚠️ API: Offline")
    
    # Show DB status
    pred_count = db.get_prediction_count()
    if pred_count > 0:
        st.success(f"✅ Database: {pred_count} records")
    else:
        st.info("📝 Database: Empty")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
        This dashboard provides comprehensive monitoring 
        for the sentiment analysis model including:
        
        - Live predictions
        - Performance metrics
        - Data drift detection
        - System health monitoring
    """)
