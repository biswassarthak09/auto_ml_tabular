"""
Verification script to check engineered datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dataset(task: str, fold: int, datadir: Path, engineered_dir: Path):
    """Check and compare original vs engineered dataset."""
    
    # Original data
    original_fold_dir = datadir / task / str(fold)
    if not original_fold_dir.exists():
        logger.warning(f"Original fold {fold} for {task} not found")
        return None
    
    # Engineered data
    engineered_fold_dir = engineered_dir / task / str(fold)
    if not engineered_fold_dir.exists():
        logger.warning(f"Engineered fold {fold} for {task} not found")
        return None
    
    try:
        # Load original data
        X_train_orig = pd.read_parquet(original_fold_dir / "X_train.parquet")
        X_test_orig = pd.read_parquet(original_fold_dir / "X_test.parquet")
        
        # Load engineered data
        X_train_eng = pd.read_parquet(engineered_fold_dir / "X_train.parquet")
        X_test_eng = pd.read_parquet(engineered_fold_dir / "X_test.parquet")
        
        # Check shapes
        logger.info(f"{task} fold {fold}:")
        logger.info(f"  Original train: {X_train_orig.shape}, test: {X_test_orig.shape}")
        logger.info(f"  Engineered train: {X_train_eng.shape}, test: {X_test_eng.shape}")
        logger.info(f"  Feature expansion: {X_train_orig.shape[1]} → {X_train_eng.shape[1]} ({X_train_eng.shape[1]/X_train_orig.shape[1]:.1f}x)")
        
        # Check for missing values
        train_missing = X_train_eng.isnull().sum().sum()
        test_missing = X_test_eng.isnull().sum().sum()
        
        if train_missing > 0 or test_missing > 0:
            logger.warning(f"  Missing values found: train={train_missing}, test={test_missing}")
        else:
            logger.info(f"  ✅ No missing values")
        
        # Check for infinite values
        train_inf = np.isinf(X_train_eng.select_dtypes(include=[np.number])).sum().sum()
        test_inf = np.isinf(X_test_eng.select_dtypes(include=[np.number])).sum().sum()
        
        if train_inf > 0 or test_inf > 0:
            logger.warning(f"  Infinite values found: train={train_inf}, test={test_inf}")
        else:
            logger.info(f"  ✅ No infinite values")
        
        return {
            'task': task,
            'fold': fold,
            'original_features': X_train_orig.shape[1],
            'engineered_features': X_train_eng.shape[1],
            'expansion_ratio': X_train_eng.shape[1] / X_train_orig.shape[1],
            'train_samples': X_train_eng.shape[0],
            'test_samples': X_test_eng.shape[0],
            'missing_values': train_missing + test_missing,
            'infinite_values': train_inf + test_inf
        }
        
    except Exception as e:
        logger.error(f"Error checking {task} fold {fold}: {e}")
        return None

def main():
    """Check all engineered datasets."""
    current_dir = Path(__file__).parent
    datadir = current_dir / "data"
    engineered_dir = current_dir / "data_engineered"
    
    datasets = ["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1"]
    
    results = []
    
    for dataset in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Checking dataset: {dataset}")
        logger.info(f"{'='*50}")
        
        for fold in range(1, 11):
            result = check_dataset(dataset, fold, datadir, engineered_dir)
            if result:
                results.append(result)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    
    if results:
        df = pd.DataFrame(results)
        
        # Group by task
        summary = df.groupby('task').agg({
            'original_features': 'mean',
            'engineered_features': 'mean',
            'expansion_ratio': 'mean',
            'train_samples': 'mean',
            'test_samples': 'mean',
            'fold': 'count'
        }).round(2)
        
        summary.columns = ['Orig_Features', 'Eng_Features', 'Expansion_Ratio', 'Train_Samples', 'Test_Samples', 'Folds_Processed']
        
        print("\nDataset Summary:")
        print(summary)
        
        # Check for any issues
        issues = df[(df['missing_values'] > 0) | (df['infinite_values'] > 0)]
        if not issues.empty:
            print(f"\n⚠️  {len(issues)} folds have data quality issues:")
            print(issues[['task', 'fold', 'missing_values', 'infinite_values']])
        else:
            print("\n✅ All datasets are clean (no missing or infinite values)")
        
        print(f"\n📊 Total folds processed: {len(results)}")
        print(f"📁 Engineered datasets saved in: {engineered_dir}")
        
    else:
        logger.warning("No results found. Check if feature engineering was run successfully.")

if __name__ == "__main__":
    main()
