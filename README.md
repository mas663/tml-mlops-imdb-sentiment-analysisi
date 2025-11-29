# 🎯 MLOps Sentiment Analysis - IMDB 50K Dataset

Proyek MLOps end-to-end untuk Sentiment Analysis menggunakan dataset IMDB 50K reviews. Implementasi complete pipeline dengan data versioning, experiment tracking, dan reproducibility.

## 📋 Overview

- **Dataset**: IMDB 50K movie reviews (25K positive, 25K negative)
- **Model**: Logistic Regression dengan TF-IDF vectorization
- **Pipeline**: Data preparation → Training → Evaluation
- **Tools**: Prefect, MLflow, Git LFS, scikit-learn

## 🎓 Timeline Pengerjaan

### ✅ Minggu ke-13: Fondasi Sistem, Versioning, Eksperimen

- [x] Setup environment dan repository
- [x] Data versioning dengan Git LFS
- [x] Baseline model training
- [x] Setup MLflow tracking
- [x] Eksperimen dengan berbagai hyperparameter

### 🚀 Minggu ke-14: Pipeline, Orchestration, Pemilihan Model

- [x] Membuat modular pipeline (prepare_data.py, train.py, evaluate.py)
- [x] Integrasi MLflow ke pipeline
- [x] Prefect workflow orchestration
- [x] Analisis eksperimen dan pilih model terbaik
- [x] Export model untuk deployment
- [x] Dokumentasi arsitektur sistem

## 📁 Struktur Project

```
mlops-sentiment/
├── data/
│   ├── raw/
│   │   └── imdb.csv          # Dataset asli (tracked by Git LFS)
│   └── processed/
│       ├── train.csv         # 80% data untuk training (tracked by Git LFS)
│       └── test.csv          # 20% data untuk testing (tracked by Git LFS)
├── pipeline/
│   ├── prepare_data.py       # Data cleaning & splitting
│   ├── train.py              # Model training + MLflow logging
│   ├── evaluate.py           # Model evaluation + metrics
│   ├── experiment.py         # Automated experiments
│   └── prefect_flow.py       # Prefect workflow orchestration
├── models/
│   ├── model.pkl             # Trained model
│   └── vectorizer.pkl        # TF-IDF vectorizer
├── mlruns/                   # MLflow tracking data
├── .gitattributes            # Git LFS configuration
├── requirements.txt          # Python dependencies
├── metrics.json              # Evaluation metrics
└── README.md                 # Documentation

```

## 🔧 Setup & Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd mlops-sentiment
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# atau
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Git LFS for Data Files

```bash
# Install Git LFS (jika belum)
brew install git-lfs  # macOS
# atau
sudo apt-get install git-lfs  # Ubuntu

# Initialize Git LFS
git lfs install

# Pull data files
git lfs pull
```

## 🚀 Menjalankan Pipeline

### Opsi 1: Jalankan Full Pipeline dengan Prefect

```bash
# Jalankan pipeline dengan Prefect (local mode)
PREFECT_API_URL="" python pipeline/prefect_flow.py

# Atau start Prefect server untuk monitoring UI (opsional)
prefect server start
# Lalu di terminal lain:
python pipeline/prefect_flow.py
```

### Opsi 2: Jalankan Manual Step-by-Step

```bash
# Step 1: Prepare data (cleaning & splitting)
python pipeline/prepare_data.py

# Step 2: Train model
python pipeline/train.py

# Step 3: Evaluate model
python pipeline/evaluate.py
```

### Opsi 3: Jalankan Eksperimen Batch

```bash
# Jalankan multiple eksperimen dengan hyperparameter berbeda
python pipeline/experiment.py
```

## 📊 MLflow Tracking

### Start MLflow UI

```bash
mlflow ui --port 5000
```

Akses dashboard di: http://127.0.0.1:5000

### Informasi yang Tracked

- **Parameters**: max_features, max_iter, solver, model_type
- **Metrics**:
  - Training accuracy
  - Test accuracy, precision, recall, F1-score
- **Artifacts**:
  - Trained model (model.pkl)
  - Vectorizer (vectorizer.pkl)
  - Metrics file (metrics.json)

## 📈 Hasil Eksperimen

### Baseline Model (20K features)

```
Model: LogisticRegression
Max Features: 20,000
Test Accuracy: 89.97%
Test Precision: 89.55%
Test Recall: 90.50%
Test F1-Score: 90.02%
```

### Eksperimen Perbandingan

| Experiment | Model Type          | Max Features | Test Accuracy | Test F1-Score |
| ---------- | ------------------- | ------------ | ------------- | ------------- |
| Baseline   | Logistic Regression | 20,000       | 89.97%        | 90.02%        |
| Exp 1      | Logistic Regression | 10,000       | 89.83%        | 89.89%        |
| Exp 2      | Logistic Regression | 30,000       | 90.07%        | 90.12%        |

**Model Terbaik**: Logistic Regression dengan 30K features (F1: 90.12%)

## 🔄 Prefect Pipeline Architecture

### Workflow Structure

```python
@flow(name="imdb-sentiment-pipeline")
def sentiment_analysis_pipeline():
    # Task 1: Data preparation
    prepare_data_task()

    # Task 2: Model training (depends on prepare)
    train_model_task()

    # Task 3: Model evaluation (depends on train)
    evaluate_model_task()
```

### Task Features

- **Retries**: Automatic retry on failure (prepare: 2x, train/evaluate: 1x)
- **Retry Delay**: Configurable delay between retries
- **Concurrent Runner**: Tasks can run in parallel when possible
- **Logging**: Full execution logs and error tracking

### Running with Prefect UI (Optional)

```bash
# Terminal 1: Start Prefect server
prefect server start

# Terminal 2: Run pipeline
python pipeline/prefect_flow.py
```

Akses Prefect UI di: http://127.0.0.1:4200

## 📊 Metrics & Evaluation

### Confusion Matrix

```
[[4472  528]
 [ 475 4525]]
```

### Classification Report

```
              precision    recall  f1-score   support

    negative       0.90      0.89      0.90      5000
    positive       0.90      0.91      0.90      5000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000
```

## 🛠️ Tech Stack

- **Data Versioning**: Git LFS (Large File Storage)
- **Workflow Orchestration**: Prefect
- **Experiment Tracking**: MLflow
- **ML Framework**: scikit-learn
- **Feature Engineering**: TF-IDF Vectorization
- **Model**: Logistic Regression

## 📝 Best Practices Implemented

✅ **Train-Test Split**: 80-20 stratified split untuk mencegah overfitting  
✅ **Reproducibility**: Random seed tetap (42) di semua eksperimen  
✅ **Data Versioning**: Dataset tracked dengan Git LFS untuk file besar  
✅ **Experiment Tracking**: Semua hyperparameter dan metrics logged ke MLflow  
✅ **Workflow Orchestration**: Prefect untuk automated pipeline dengan retry logic  
✅ **Code Quality**: Modular code dengan error handling dan logging  
✅ **Metrics Lengkap**: Accuracy, Precision, Recall, F1, Confusion Matrix

## 🚀 Next Steps (Minggu ke-14 lanjutan)

- [ ] **Model Serving**: Buat FastAPI endpoint di folder `api/`
- [ ] **Dashboard**: Streamlit dashboard untuk inference demo
- [ ] **Containerization**: Dockerfile untuk deployment
- [ ] **Monitoring**: Evidently untuk data drift detection
- [ ] **CI/CD**: GitHub Actions untuk automated testing
- [ ] **Documentation**: Diagram arsitektur MLOps

## 🎯 Perintah Penting

```bash
# Aktivasi environment
source venv/bin/activate

# Jalankan pipeline lengkap
python pipeline/prefect_flow.py

# Lihat MLflow dashboard
mlflow ui --port 5000

# Start Prefect server (opsional, untuk monitoring)
prefect server start

# Git LFS commands
git lfs pull              # Pull large files
git lfs ls-files          # List tracked files

# Lihat metrics
cat metrics.json
```

## 📚 Resources

- [Prefect Documentation](https://docs.prefect.io/)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn Documentation](https://scikit-learn.org/)

## 👨‍💻 Author

**Mohammad Affan Shofi**  
Institut Teknologi Sepuluh Nopember (ITS)

---

## 📄 License

This project is for educational purposes.

---

**Status**: ✅ Minggu ke-13 & ke-14 COMPLETED  
**Last Update**: November 29, 2025
