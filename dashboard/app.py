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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Simple styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #CCCCCC;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">MLOps Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Monitoring & Evaluation System</div>', unsafe_allow_html=True)

st.markdown("---")

# Quick Stats Section
st.subheader("System Overview")

# Get database stats
db = get_db_manager()
stats = db.get_stats()

# Load model metrics
try:
    with open('metrics.json', 'r') as f:
        model_metrics = json.load(f)
except:
    model_metrics = {}

# Stats cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Experiments",
        value="17",
        help="Total MLflow experiment runs"
    )

with col2:
    accuracy = model_metrics.get('test_accuracy', 0.8997)
    st.metric(
        label="Best Accuracy",
        value=f"{accuracy*100:.2f}%",
        help="Best model test accuracy"
    )

with col3:
    st.metric(
        label="Total Predictions",
        value=f"{stats['total_predictions']:,}",
        help="Total API predictions logged"
    )

with col4:
    avg_conf = stats.get('avg_confidence', 0)
    if avg_conf > 0:
        st.metric(
            label="Avg Confidence",
            value=f"{avg_conf:.1f}%",
            help="Average prediction confidence"
        )
    else:
        st.metric(
            label="Avg Confidence",
            value="N/A",
            help="No predictions yet"
        )

st.markdown("<br>", unsafe_allow_html=True)

# Architecture Section
st.subheader("System Architecture")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("""
        **Tech Stack:**
        - Data Versioning: Git LFS (130MB dataset)
        - Experiment Tracking: MLflow (17 runs)
        - Orchestration: Prefect (3-task pipeline)
        - Model Serving: FastAPI + Uvicorn
        - Monitoring: Streamlit + Evidently
        - Containerization: Docker Compose
    """)

with col_right:
    st.markdown("""
        **Model Details:**
        - Algorithm: Logistic Regression
        - Vectorization: TF-IDF (20,000 features)
        - Training Data: 40,000 reviews
        - Test Data: 10,000 reviews
        - Accuracy: 89.97%
        - F1-Score: 90.02%
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Pipeline Flow
st.subheader("ML Pipeline Flow")

st.text("""
Raw Data  -->  Preprocessing  -->  Training  -->  Evaluation  -->  Deployment
(Git LFS)      (Prefect)          (MLflow)       (Metrics)        (FastAPI)
                                                                       |
                                                                       v
                                                                  Monitoring
                                                                 (Dashboard)
""")

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Guide
st.subheader("Pages")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
        **Live Prediction**
        
        Test model dengan input review sendiri. 
        Lihat hasil prediksi real-time dengan confidence score.
    """)

with col2:
    st.info("""
        **Performance Monitor**
        
        Dashboard untuk monitoring performa model.
        Lihat statistics, charts, dan recent predictions.
    """)

with col3:
    st.info("""
        **Data Drift Detection**
        
        Monitoring data drift antara training dan production data.
        Alert system untuk retraining recommendations.
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>MLOps Final Project - Institut Teknologi Sepuluh Nopember</p>
        <p>Mohammad Affan Shofi | December 2025</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Quick Links")
    
    st.markdown("### Services")
    st.markdown("- [MLflow UI](http://localhost:5000) - Experiment Tracking")
    st.markdown("- [Prefect UI](http://localhost:4200) - Pipeline Orchestration")
    st.markdown("- [API Docs](http://localhost:8000/docs) - FastAPI Swagger")
    st.markdown("- [API Health](http://localhost:8000/health) - Health Check")
    
    st.markdown("---")
    
    st.markdown("### Current Status")
    
    # Check if services are running
    import requests
    
    # Check API
    try:
        response = requests.get("http://api:8000/health", timeout=2)
        if response.status_code == 200:
            st.success("API: Running")
        else:
            st.error("API: Error")
    except:
        st.warning("API: Offline")
    
    # Check MLflow (just check if port is accessible)
    try:
        response = requests.get("http://mlflow:5000", timeout=2)
        st.success("MLflow: Running")
    except:
        st.warning("MLflow: Offline")
