# MLOps Sentiment Analysis

Sistem sentiment analysis otomatis untuk IMDB movie reviews dengan complete MLOps pipeline.

## Quick Start

```bash
# Clone & setup
git clone <repo-url>
cd mlops-sentiment
git lfs pull

# Run dengan Docker (recommended)
docker-compose up -d

# Access services:
# - MLflow: http://localhost:5000
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
```

## Dataset & Model

- **Dataset**: IMDB 50K reviews (Git LFS)
- **Model**: Logistic Regression + TF-IDF
- **Performance**: Accuracy 89.97%, F1 90.02%

## Project Structure

```
├── data/                    # Dataset (Git LFS tracked)
├── pipeline/                # ML pipeline scripts
│   ├── prepare_data.py      # Data preparation
│   ├── train.py             # Model training
│   ├── evaluate.py          # Evaluation
│   └── prefect_flow.py      # Orchestration
├── api/                     # FastAPI endpoints
├── dashboard/               # Streamlit monitoring
├── models/                  # Saved models
├── mlruns/                  # MLflow tracking
└── docker-compose.yml       # Container orchestration
```

## Features

### A. Data Versioning (Git LFS)
- Dataset tracked dengan Git LFS
- Remote storage: GitHub LFS
- Command: `git lfs pull`

### B. Experiment Tracking (MLflow)
- Auto-logging metrics & parameters
- Web UI untuk comparison
- Access: http://localhost:5000

### C. Pipeline Orchestration (Prefect)
- 3 tasks: prepare → train → evaluate
- Retry logic & error handling
- Run: `python pipeline/prefect_flow.py`

### D. Model Deployment (FastAPI)
- REST API endpoints
- JSON input/output
- Docs: http://localhost:8000/docs

**Test API:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'
```

### E. Monitoring Dashboard (Streamlit)
- Live prediction interface
- Performance charts
- Data drift detection
- Access: http://localhost:8501

## Running Pipeline

### Option 1: Docker (Recommended)
```bash
docker-compose up -d
```

### Option 2: Local
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python pipeline/prefect_flow.py

# Start MLflow UI
mlflow ui --port 5000
```

## Tech Stack

- **Orchestration**: Prefect
- **Tracking**: MLflow
- **Versioning**: Git LFS
- **API**: FastAPI
- **Monitoring**: Streamlit + Evidently
- **ML**: scikit-learn
- **Containers**: Docker

## Key Results

**Baseline Model:**
```
Test Accuracy:  89.97%
Test Precision: 89.55%
Test Recall:    90.50%
Test F1-Score:  90.02%
```

**Confusion Matrix:**
```
[[4472  528]
 [ 475 4525]]
```

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| MLflow | 5000 | Experiment tracking |
| Prefect | 4200 | Workflow UI |
| API | 8000 | Model serving |
| Dashboard | 8501 | Monitoring UI |

## Useful Commands

```bash
# View experiments
python pipeline/compare_experiments.py

# Check service status
docker-compose ps

# View logs
docker-compose logs api

# Stop all
docker-compose down

# Test prediction
./test_api.sh
```

## Assignment Compliance

✅ **A. Data Versioning**: Git LFS dengan remote storage  
✅ **B. Experiment Tracking**: MLflow dashboard  
✅ **C. Orchestration**: Prefect pipeline + Dockerfile  
✅ **D. Model Deployment**: FastAPI REST endpoint  
✅ **E. Monitoring**: Streamlit dashboard + drift detection  

---

**Author**: Mohammad Affan Shofi  
**Institution**: ITS Surabaya  
**Date**: December 2025
