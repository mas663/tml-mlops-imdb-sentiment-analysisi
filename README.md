# 🎯 MLOps Sentiment Analysis - IMDB 50K Dataset

Proyek MLOps end-to-end untuk Sentiment Analysis menggunakan dataset IMDB 50K reviews. Implementasi complete pipeline dengan data versioning, experiment tracking, dan reproducibility.

## 📋 Overview

- **Dataset**: IMDB 50K movie reviews (25K positive, 25K negative)
- **Model**: Logistic Regression dengan TF-IDF vectorization
- **Pipeline**: Data preparation → Training → Evaluation
- **Tools**: DVC, MLflow, scikit-learn

## 🎓 Timeline Pengerjaan

### ✅ Minggu ke-13: Fondasi Sistem, Versioning, Eksperimen
- [x] Setup environment dan repository
- [x] Data versioning dengan DVC
- [x] Baseline model training
- [x] Setup MLflow tracking
- [x] Eksperimen dengan berbagai hyperparameter

### 🚀 Minggu ke-14: Pipeline, Orchestration, Pemilihan Model
- [x] Membuat modular pipeline (prepare_data.py, train.py, evaluate.py)
- [x] Integrasi MLflow ke pipeline
- [x] DVC Pipeline orchestration
- [ ] Analisis eksperimen dan pilih model terbaik
- [ ] Export model untuk deployment
- [ ] Dokumentasi arsitektur sistem

## 📁 Struktur Project

```
mlops-sentiment/
├── data/
│   ├── raw/
│   │   ├── imdb.csv          # Dataset asli (tracked by DVC)
│   │   └── imdb.csv.dvc      # DVC metadata
│   └── processed/
│       ├── train.csv         # 80% data untuk training
│       └── test.csv          # 20% data untuk testing
├── pipeline/
│   ├── prepare_data.py       # Data cleaning & splitting
│   ├── train.py              # Model training + MLflow logging
│   ├── evaluate.py           # Model evaluation + metrics
│   └── experiment.py         # Automated experiments
├── models/
│   ├── model.pkl             # Trained model
│   └── vectorizer.pkl        # TF-IDF vectorizer
├── mlruns/                   # MLflow tracking data
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # DVC pipeline lock file
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

### 4. Pull Data from DVC Remote (jika sudah setup)

```bash
dvc pull
```

## 🚀 Menjalankan Pipeline

### Opsi 1: Jalankan Full Pipeline dengan DVC

```bash
# Jalankan semua stages (prepare → train → evaluate)
dvc repro

# Lihat status pipeline
dvc status

# Lihat DAG pipeline
dvc dag
```

### Opsi 2: Jalankan Manual Step-by-Step

```bash
# Step 1: Prepare data (cleaning & splitting)
venv/bin/python pipeline/prepare_data.py

# Step 2: Train model
venv/bin/python pipeline/train.py

# Step 3: Evaluate model
venv/bin/python pipeline/evaluate.py
```

### Opsi 3: Jalankan Eksperimen Batch

```bash
# Jalankan multiple eksperimen dengan hyperparameter berbeda
venv/bin/python pipeline/experiment.py
```

## 📊 MLflow Tracking

### Start MLflow UI

```bash
venv/bin/mlflow ui --port 5000
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

| Experiment | Model Type | Max Features | Test Accuracy | Test F1-Score |
|------------|------------|--------------|---------------|---------------|
| Baseline   | Logistic Regression | 20,000 | 89.97% | 90.02% |
| Exp 1      | Logistic Regression | 10,000 | 89.83% | 89.89% |
| Exp 2      | Logistic Regression | 30,000 | 90.07% | 90.12% |

**Model Terbaik**: Logistic Regression dengan 30K features (F1: 90.12%)

## 🔄 DVC Pipeline Stages

### Stage 1: Prepare
```yaml
cmd: venv/bin/python pipeline/prepare_data.py
deps:
  - data/raw/imdb.csv
  - pipeline/prepare_data.py
outs:
  - data/processed/train.csv
  - data/processed/test.csv
```

### Stage 2: Train
```yaml
cmd: venv/bin/python pipeline/train.py
deps:
  - data/processed/train.csv
  - pipeline/train.py
outs:
  - models/model.pkl
  - models/vectorizer.pkl
```

### Stage 3: Evaluate
```yaml
cmd: venv/bin/python pipeline/evaluate.py
deps:
  - data/processed/test.csv
  - models/model.pkl
  - models/vectorizer.pkl
  - pipeline/evaluate.py
outs:
  - metrics.txt
  - metrics.json
```

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

- **Data Versioning**: DVC (Data Version Control)
- **Experiment Tracking**: MLflow
- **ML Framework**: scikit-learn
- **Pipeline Orchestration**: DVC stages
- **Feature Engineering**: TF-IDF Vectorization
- **Model**: Logistic Regression

## 📝 Best Practices Implemented

✅ **Train-Test Split**: 80-20 stratified split untuk mencegah overfitting  
✅ **Reproducibility**: Random seed tetap (42) di semua eksperimen  
✅ **Data Versioning**: Dataset tidak masuk Git, hanya .dvc metadata  
✅ **Experiment Tracking**: Semua hyperparameter dan metrics logged ke MLflow  
✅ **Pipeline Automation**: DVC stages untuk reproducible pipeline  
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
dvc repro

# Lihat MLflow dashboard
venv/bin/mlflow ui --port 5000

# Push data ke remote (setelah setup remote)
dvc push

# Git commit pipeline changes
git add dvc.yaml dvc.lock .gitignore
git commit -m "Update pipeline"

# Lihat metrics
cat metrics.json
```

## 📚 Resources

- [DVC Documentation](https://dvc.org/doc)
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
