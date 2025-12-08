"""
Experiment runner for hyperparameter tuning
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

def load_data():
    """Load training and test datasets"""
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df["review_clean"]
    y_train = train_df["sentiment"]
    X_test = test_df["review_clean"]
    y_test = test_df["sentiment"]
    
    return X_train, y_train, X_test, y_test

def run_experiment(experiment_name, model_config, vectorizer_config):
    """Run a single experiment with given configuration"""
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("imdb-sentiment-analysis")
    
    with mlflow.start_run(run_name=experiment_name):
        # load datasets
        print("[1/6] Loading data...")
        X_train, y_train, X_test, y_test = load_data()
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # log parameters
        mlflow.log_params(vectorizer_config)
        mlflow.log_params(model_config)
        
        # vectorize text data
        print(f"\n[2/6] Vectorizing with max_features={vectorizer_config['max_features']}...")
        vectorizer = TfidfVectorizer(max_features=vectorizer_config["max_features"])
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        print(f"Shape: Train {X_train_vec.shape}, Test {X_test_vec.shape}")
        
        # train model
        print(f"\n[3/6] Training {model_config['model_type']}...")
        model_type = model_config.pop("model_type")
        
        if model_type == "LogisticRegression":
            model = LogisticRegression(**model_config)
        elif model_type == "RandomForest":
            model = RandomForestClassifier(**model_config)
        elif model_type == "LinearSVC":
            model = LinearSVC(**model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train_vec, y_train)
        print("Training completed")
        
        # compute training metrics
        print("\n[4/6] Computing training metrics...")
        y_train_pred = model.predict(X_train_vec)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"Training Accuracy: {train_acc:.4f}")
        mlflow.log_metric("train_accuracy", train_acc)
        
        # test set evaluation
        print("\n[5/6] Evaluating on test set...")
        y_test_pred = model.predict(X_test_vec)
        
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, average='binary', pos_label='positive')
        test_rec = recall_score(y_test, y_test_pred, average='binary', pos_label='positive')
        test_f1 = f1_score(y_test, y_test_pred, average='binary', pos_label='positive')
        
        print(f"\nTest Results:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall:    {test_rec:.4f}")
        print(f"  F1-Score:  {test_f1:.4f}")
        
        # Log test metrics
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)
        mlflow.log_metric("test_f1_score", test_f1)
        
        # log to MLflow
        print("\n[6/6] Logging to MLflow...")
        # skip full model logging to speed things up
        # mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        print(f"Experiment completed!")
        
        # restore model_type
        model_config["model_type"] = model_type
        
        return {
            "run_id": run_id,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "model_type": model_type
        }

def main():
    """Run all experiments"""
    
    print("\n" + "="*60)
    print("MLOPS SENTIMENT ANALYSIS - EXPERIMENT SUITE")
    print("Running multiple experiments for model comparison")
    print("="*60)
    
    experiments = [
        {
            "name": f"exp1_logistic_10k_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "vectorizer": {"max_features": 10000},
            "model": {
                "model_type": "LogisticRegression",
                "max_iter": 500,
                "solver": "lbfgs",
                "random_state": 42
            }
        },
        {
            "name": f"exp2_logistic_30k_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "vectorizer": {"max_features": 30000},
            "model": {
                "model_type": "LogisticRegression",
                "max_iter": 500,
                "solver": "lbfgs",
                "random_state": 42
            }
        },
        {
            "name": f"exp3_random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "vectorizer": {"max_features": 15000},
            "model": {
                "model_type": "RandomForest",
                "n_estimators": 100,
                "max_depth": 50,
                "random_state": 42,
                "n_jobs": -1
            }
        }
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'#'*60}")
        print(f"RUNNING EXPERIMENT {i}/{len(experiments)}")
        print(f"{'#'*60}")
        
        result = run_experiment(
            exp["name"],
            exp["model"].copy(),
            exp["vectorizer"]
        )
        results.append(result)
    
    # print summary
    print("\n\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\n{'Model Type':<25} {'Accuracy':<12} {'F1-Score':<12} {'Run ID'}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model_type']:<25} {result['test_accuracy']:<12.4f} {result['test_f1']:<12.4f} {result['run_id']}")
    
    # find best performing model
    best_result = max(results, key=lambda x: x['test_f1'])
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_result['model_type']}")
    print(f"  F1-Score: {best_result['test_f1']:.4f}")
    print(f"  Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"  Run ID: {best_result['run_id']}")
    print("="*60)
    
    print(f"\nAll experiments completed!")
    print(f"View results in MLflow UI: http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
