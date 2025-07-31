#!/usr/bin/env python3
"""
Dataset Expansion Script for Meta-Learning
===========================================

This script downloads regression datasets from OpenML and structures them
exactly like the existing datasets with 10 folds, each containing:
- X_train.parquet, X_test.parquet, y_train.parquet, y_test.parquet

Usage:
    python expand_datasets.py --n_datasets 20 --output_dir data_expanded
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_regression
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

# OpenML import (install with: pip install openml)
try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    print("⚠️ OpenML not available. Install with: pip install openml")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetExpander:
    """Class to download and structure datasets for meta-learning"""
    
    def __init__(self, output_dir: str = "data_expanded"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenML regression dataset IDs (curated list of good datasets)
        self.openml_regression_ids = [
            # Housing and real estate
            531, 537, 4,       # Boston housing variants
            41021,             # Ames housing
            
            # Energy and environment  
            244, 245,          # Energy efficiency
            564, 567,          # Concrete strength
            294,               # Protein structure
            
            # Economics and finance
            612, 616,          # CPU performance
            4552,              # Prices
            41540,             # California housing
            
            # Automotive
            1030,              # Auto mpg
            
            # Chemistry and materials
            4534, 4535,        # Chemical data
            
            # Social and survey data
            423, 531,          # Social data
            
            # Synthetic but realistic
            505, 506, 507,     # Synthetic datasets
            
            # Time series and sequential
            41146, 41147,      # Time-based data
        ]
        
    def clean_dataset_name(self, name: str) -> str:
        """Clean dataset name to be filesystem-safe"""
        # Remove special characters and spaces
        clean_name = "".join(c for c in name if c.isalnum() or c in ('-', '_')).lower()
        # Remove consecutive underscores/hyphens
        while '__' in clean_name or '--' in clean_name:
            clean_name = clean_name.replace('__', '_').replace('--', '-')
        return clean_name.strip('_-')
    
    def download_openml_datasets(self, n_datasets: int = 15) -> list:
        """Download regression datasets from OpenML"""
        if not OPENML_AVAILABLE:
            logger.error("OpenML not available. Skipping OpenML downloads.")
            return []
        
        logger.info(f"🌐 Downloading {n_datasets} datasets from OpenML...")
        datasets = []
        successful_downloads = 0
        
        for dataset_id in self.openml_regression_ids[:n_datasets * 2]:  # Try more than needed
            if successful_downloads >= n_datasets:
                break
                
            try:
                logger.info(f"Downloading dataset {dataset_id}...")
                
                # Download dataset
                dataset = openml.datasets.get_dataset(dataset_id)
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    dataset_format="dataframe",
                    target=dataset.default_target_attribute
                )
                
                # Validate dataset
                if X is None or y is None:
                    logger.warning(f"Dataset {dataset_id}: No data available")
                    continue
                    
                if len(X) < 100:
                    logger.warning(f"Dataset {dataset_id}: Too small ({len(X)} samples)")
                    continue
                    
                if len(X) > 50000:
                    logger.warning(f"Dataset {dataset_id}: Too large ({len(X)} samples), sampling...")
                    # Sample to manageable size
                    sample_idx = np.random.choice(len(X), 10000, replace=False)
                    X = X.iloc[sample_idx]
                    y = y.iloc[sample_idx]
                
                # Clean dataset name
                dataset_name = self.clean_dataset_name(dataset.name)
                if not dataset_name:
                    dataset_name = f"openml_{dataset_id}"
                
                # Ensure regression task
                if not pd.api.types.is_numeric_dtype(y):
                    logger.warning(f"Dataset {dataset_id}: Target is not numeric (classification?)")
                    continue
                
                datasets.append({
                    'name': dataset_name,
                    'X': X,
                    'y': y,
                    'source': f'openml_{dataset_id}',
                    'description': dataset.description[:100] if dataset.description else "OpenML dataset"
                })
                
                successful_downloads += 1
                logger.info(f"✅ Successfully downloaded: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
                
            except Exception as e:
                logger.warning(f"Failed to download dataset {dataset_id}: {str(e)}")
                continue
        
        logger.info(f"✅ Downloaded {len(datasets)} datasets from OpenML")
        return datasets
    
    def generate_synthetic_datasets(self, n_datasets: int = 10) -> list:
        """Generate synthetic regression datasets with varying characteristics"""
        logger.info(f"🔬 Generating {n_datasets} synthetic datasets...")
        datasets = []
        
        for i in range(n_datasets):
            # Vary dataset characteristics
            n_samples = np.random.choice([500, 1000, 2000, 5000, 8000])
            n_features = np.random.choice([5, 10, 15, 20, 30, 50])
            n_informative = max(3, int(n_features * np.random.uniform(0.5, 0.9)))
            noise = np.random.uniform(0.1, 0.3)
            
            # Generate dataset
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                noise=noise,
                random_state=42 + i,
                bias=np.random.uniform(-10, 10)
            )
            
            # Convert to DataFrame
            feature_names = [f'feature_{j}' for j in range(n_features)]
            X_df = pd.DataFrame(X, columns=feature_names)
            y_series = pd.Series(y, name='target')
            
            dataset_name = f"synthetic_reg_{i+1:02d}"
            
            datasets.append({
                'name': dataset_name,
                'X': X_df,
                'y': y_series,
                'source': 'synthetic',
                'description': f"Synthetic regression dataset with {n_samples} samples, {n_features} features"
            })
            
            logger.info(f"✅ Generated: {dataset_name} ({n_samples} samples, {n_features} features)")
        
        logger.info(f"✅ Generated {len(datasets)} synthetic datasets")
        return datasets
    
    def preprocess_dataset(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Preprocess dataset: handle missing values, encode categoricals, etc."""
        
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Handle missing values in features
        if X_processed.isnull().any().any():
            logger.info("Handling missing values in features...")
            # Fill numeric columns with median
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 'unknown')
        
        # Handle missing values in target
        if y_processed.isnull().any():
            logger.info("Handling missing values in target...")
            y_processed = y_processed.fillna(y_processed.median())
        
        # Encode categorical features
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding categorical features: {list(categorical_cols)}")
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Ensure all features are numeric
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        # Remove any remaining infinite or very large values
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(X_processed.median())
        
        y_processed = y_processed.replace([np.inf, -np.inf], np.nan)
        y_processed = y_processed.fillna(y_processed.median())
        
        return X_processed, y_processed
    
    def create_fold_structure(self, dataset_name: str, X: pd.DataFrame, y: pd.Series):
        """Create 10-fold structure exactly like existing datasets"""
        
        logger.info(f"📁 Creating fold structure for {dataset_name}...")
        
        # Create dataset directory
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess dataset
        X_clean, y_clean = self.preprocess_dataset(X, y)
        
        # Create 10-fold split
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_num = 1
        for train_idx, test_idx in kf.split(X_clean):
            fold_dir = dataset_dir / str(fold_num)
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            # Split data
            X_train = X_clean.iloc[train_idx]
            X_test = X_clean.iloc[test_idx]
            y_train = y_clean.iloc[train_idx]
            y_test = y_clean.iloc[test_idx]
            
            # Save as parquet files (same format as existing datasets)
            X_train.to_parquet(fold_dir / 'X_train.parquet', index=False)
            X_test.to_parquet(fold_dir / 'X_test.parquet', index=False)
            
            # Save target as DataFrame with 'target' column (matching existing format)
            y_train_df = pd.DataFrame({'target': y_train})
            y_test_df = pd.DataFrame({'target': y_test})
            y_train_df.to_parquet(fold_dir / 'y_train.parquet', index=False)
            y_test_df.to_parquet(fold_dir / 'y_test.parquet', index=False)
            
            logger.info(f"  Fold {fold_num}: train({X_train.shape[0]}) test({X_test.shape[0]})")
            fold_num += 1
        
        # Save dataset metadata
        metadata = {
            'name': dataset_name,
            'n_samples': len(X_clean),
            'n_features': len(X_clean.columns),
            'target_name': y_clean.name or 'target',
            'feature_names': list(X_clean.columns),
            'n_folds': 10
        }
        
        metadata_file = dataset_dir / 'metadata.json'
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Created fold structure for {dataset_name}")
        return dataset_dir
    
    def expand_datasets(self, n_openml: int = 15, n_synthetic: int = 10):
        """Main function to expand datasets"""
        
        logger.info("🚀 Starting dataset expansion...")
        logger.info(f"Target: {n_openml} OpenML + {n_synthetic} synthetic = {n_openml + n_synthetic} total datasets")
        
        all_datasets = []
        
        # Download OpenML datasets
        if n_openml > 0:
            openml_datasets = self.download_openml_datasets(n_openml)
            all_datasets.extend(openml_datasets)
        
        # Generate synthetic datasets
        if n_synthetic > 0:
            synthetic_datasets = self.generate_synthetic_datasets(n_synthetic)
            all_datasets.extend(synthetic_datasets)
        
        # Create fold structures
        logger.info(f"📂 Creating fold structures for {len(all_datasets)} datasets...")
        created_datasets = []
        
        for dataset_info in all_datasets:
            try:
                dataset_dir = self.create_fold_structure(
                    dataset_info['name'],
                    dataset_info['X'],
                    dataset_info['y']
                )
                created_datasets.append(dataset_dir)
                
            except Exception as e:
                logger.error(f"Failed to create fold structure for {dataset_info['name']}: {str(e)}")
                continue
        
        # Summary
        logger.info("=" * 60)
        logger.info("🎉 DATASET EXPANSION COMPLETED!")
        logger.info(f"✅ Successfully created {len(created_datasets)} datasets")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📊 Total folds created: {len(created_datasets) * 10}")
        logger.info("=" * 60)
        
        # List created datasets
        logger.info("📋 Created datasets:")
        for i, dataset_dir in enumerate(created_datasets, 1):
            logger.info(f"  {i:2d}. {dataset_dir.name}")
        
        return created_datasets

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Expand datasets for meta-learning')
    parser.add_argument('--n_openml', type=int, default=15, 
                       help='Number of OpenML datasets to download (default: 15)')
    parser.add_argument('--n_synthetic', type=int, default=10,
                       help='Number of synthetic datasets to generate (default: 10)')
    parser.add_argument('--output_dir', type=str, default='data_expanded',
                       help='Output directory for expanded datasets (default: data_expanded)')
    
    args = parser.parse_args()
    
    # Create expander
    expander = DatasetExpander(args.output_dir)
    
    # Expand datasets
    created_datasets = expander.expand_datasets(
        n_openml=args.n_openml,
        n_synthetic=args.n_synthetic
    )
    
    print(f"\n🎉 Success! Created {len(created_datasets)} datasets in '{args.output_dir}'")
    print(f"💡 You can now run NAS-HPO on these datasets to get more meta-learning training data!")

if __name__ == "__main__":
    main()
