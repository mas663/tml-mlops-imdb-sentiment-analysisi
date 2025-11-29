import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean and normalize text data"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("=" * 60)
    print("DATA PREPARATION - IMDB Sentiment Analysis")
    print("=" * 60)
    
    # Paths
    input_path = "data/raw/imdb.csv"
    output_dir = "data/processed"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/5] Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Dataset loaded: {len(df)} rows")
    except FileNotFoundError:
        print(f"✗ Error: File {input_path} not found!")
        return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Validate columns
    print("\n[2/5] Validating dataset columns...")
    required_cols = ["review", "sentiment"]
    if not all(col in df.columns for col in required_cols):
        print(f"✗ Error: Missing required columns. Found: {df.columns.tolist()}")
        return
    print(f"✓ Required columns present: {required_cols}")
    
    # Clean text
    print("\n[3/5] Cleaning text data...")
    df["review_clean"] = df["review"].apply(clean_text)
    print(f"✓ Text cleaning completed")
    
    # Train-test split
    print("\n[4/5] Splitting data (80% train, 20% test)...")
    train_df, test_df = train_test_split(
        df[["review_clean", "sentiment"]], 
        test_size=0.2, 
        random_state=42,
        stratify=df["sentiment"]
    )
    print(f"✓ Train set: {len(train_df)} rows")
    print(f"✓ Test set: {len(test_df)} rows")
    
    # Save splits
    print("\n[5/5] Saving processed data...")
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Train data saved to: {train_path}")
    print(f"✓ Test data saved to: {test_path}")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()