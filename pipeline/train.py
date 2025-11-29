import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os
from datetime import datetime

def main():
    print("=" * 60)
    print("MODEL TRAINING - IMDB Sentiment Analysis")
    print("=" * 60)
    
    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("imdb-sentiment-analysis")
    
    # Hyperparameters
    params = {
        "max_features": 20000,
        "max_iter": 500,
        "solver": "lbfgs",
        "random_state": 42,
        "model_type": "LogisticRegression"
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_params(params)
        
        # Load training data
        print("\n[1/6] Loading training data...")
        train_path = "data/processed/train.csv"
        try:
            df = pd.read_csv(train_path)
            print(f"✓ Training data loaded: {len(df)} rows")
        except FileNotFoundError:
            print(f"✗ Error: {train_path} not found!")
            return
        
        X_train = df["review_clean"]
        y_train = df["sentiment"]
        
        # Create and fit vectorizer
        print(f"\n[2/6] Creating TF-IDF vectorizer (max_features={params['max_features']})...")
        vectorizer = TfidfVectorizer(max_features=params["max_features"])
        X_train_vec = vectorizer.fit_transform(X_train)
        print(f"✓ Vectorizer fitted. Shape: {X_train_vec.shape}")
        
        # Train model
        print(f"\n[3/6] Training {params['model_type']}...")
        model = LogisticRegression(
            max_iter=params["max_iter"],
            solver=params["solver"],
            random_state=params["random_state"]
        )
        model.fit(X_train_vec, y_train)
        print(f"✓ Model trained successfully")
        
        # Calculate training accuracy
        print("\n[4/6] Calculating training metrics...")
        y_train_pred = model.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"✓ Training Accuracy: {train_accuracy:.4f}")
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        
        # Save models
        print("\n[5/6] Saving models...")
        os.makedirs("models", exist_ok=True)
        
        vectorizer_path = "models/vectorizer.pkl"
        model_path = "models/model.pkl"
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(model, model_path)
        
        print(f"✓ Vectorizer saved to: {vectorizer_path}")
        print(f"✓ Model saved to: {model_path}")
        
        # Log artifacts to MLflow
        print("\n[6/6] Logging artifacts to MLflow...")
        mlflow.log_artifact(vectorizer_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✓ Artifacts logged to MLflow")
        print(f"✓ Run ID: {mlflow.active_run().info.run_id}")
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

if __name__ == "__main__":
    main()
