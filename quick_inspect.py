#!/usr/bin/env python3
"""
Quick Dataset Inspector
A simple script to quickly view the heads of original and engineered datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def quick_inspect(dataset_name, fold=1, n_rows=3):
    """Quickly inspect a dataset and show its head."""
    
    # Set up paths
    current_dir = Path(__file__).parent
    original_path = current_dir / "data" / dataset_name / str(fold)
    engineered_path = current_dir / "data_engineered" / dataset_name / str(fold)
    
    # Check if paths exist
    if not original_path.exists():
        print(f"❌ Original dataset not found: {original_path}")
        return
    
    if not engineered_path.exists():
        print(f"❌ Engineered dataset not found: {engineered_path}")
        return
    
    # Load data
    try:
        X_train_orig = pd.read_parquet(original_path / "X_train.parquet")
        X_train_eng = pd.read_parquet(engineered_path / "X_train.parquet")
        y_train_orig = pd.read_parquet(original_path / "y_train.parquet")
        
        print(f"🔍 Dataset: {dataset_name.upper()} - Fold {fold}")
        print(f"📊 Shape: {X_train_orig.shape} → {X_train_eng.shape} ({X_train_eng.shape[1]/X_train_orig.shape[1]:.1f}x expansion)")
        print(f"🎯 Target: {y_train_orig.shape}")
        
        print(f"\n📄 Original Data (first {n_rows} rows):")
        print(X_train_orig.head(n_rows))
        
        print(f"\n🔧 Engineered Data (first {n_rows} rows, first 10 columns):")
        print(X_train_eng.head(n_rows).iloc[:, :10])
        
        if X_train_eng.shape[1] > 10:
            print(f"\n🔧 Engineered Data (first {n_rows} rows, last 5 columns):")
            print(X_train_eng.head(n_rows).iloc[:, -5:])
        
        print(f"\n📈 Feature Categories:")
        original_features = len(X_train_orig.columns)
        engineered_features = len(X_train_eng.columns)
        statistical_features = len([col for col in X_train_eng.columns if 'row_' in col])
        interaction_features = len([col for col in X_train_eng.columns if any(op in col for op in ['_plus_', '_mult_', '_div_', '_minus_'])])
        transform_features = len([col for col in X_train_eng.columns if any(trans in col for trans in ['_sqrt', '_log', '_squared', '_abs'])])
        categorical_features = len([col for col in X_train_eng.columns if any(cat in col for cat in ['_target_mean', '_freq'])])
        
        print(f"  Original: {original_features}")
        print(f"  Statistical: {statistical_features}")
        print(f"  Interactions: {interaction_features}")
        print(f"  Transformations: {transform_features}")
        print(f"  Categorical: {categorical_features}")
        print(f"  Total: {engineered_features}")
        
        # Data quality
        missing_vals = X_train_eng.isnull().sum().sum()
        inf_vals = np.isinf(X_train_eng.select_dtypes(include=[np.number])).sum().sum()
        print(f"\n✅ Data Quality: Missing={missing_vals}, Infinite={inf_vals}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")

def main():
    """Main function with simple argument parsing."""
    
    datasets = ["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1"]
    
    if len(sys.argv) < 2:
        print("Usage: python quick_inspect.py <dataset_name> [fold] [n_rows]")
        print(f"Available datasets: {', '.join(datasets)}")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    if dataset_name not in datasets:
        print(f"❌ Invalid dataset. Choose from: {', '.join(datasets)}")
        sys.exit(1)
    
    quick_inspect(dataset_name, fold, n_rows)

if __name__ == "__main__":
    main()
