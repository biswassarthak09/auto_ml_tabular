"""
Simple Feature Engineering Script for Tabular Data
Uses manual feature engineering techniques with sklearn.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Optional
import joblib
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SimpleManualFeatureEngineer:
    """
    Simple manual feature engineering that avoids problematic techniques.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.fitted_transformers = {}
        self.feature_names = []
        self.original_feature_names = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.is_fitted = False
        
    def identify_column_types(self, X: pd.DataFrame) -> None:
        """Identify numeric and categorical columns."""
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(self.numeric_cols)} numeric and {len(self.categorical_cols)} categorical columns")
    
    def create_simple_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = True) -> pd.DataFrame:
        """Create simple features that are safe and robust."""
        X_engineered = X.copy()
        
        # 1. Handle missing values
        numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            X_engineered[numeric_cols] = X_engineered[numeric_cols].fillna(X_engineered[numeric_cols].median())
        if categorical_cols:
            # Convert categorical columns to string first, then fill
            for col in categorical_cols:
                X_engineered[col] = X_engineered[col].astype(str).fillna('unknown')
        
        # 2. Create simple interaction features for numeric columns
        if len(self.numeric_cols) >= 2:
            top_cols = self.numeric_cols[:min(5, len(self.numeric_cols))]
            
            # Statistical features
            numeric_data = X_engineered[self.numeric_cols]
            X_engineered['row_mean'] = numeric_data.mean(axis=1)
            X_engineered['row_std'] = numeric_data.std(axis=1).fillna(0)
            X_engineered['row_max'] = numeric_data.max(axis=1)
            X_engineered['row_min'] = numeric_data.min(axis=1)
            X_engineered['row_range'] = X_engineered['row_max'] - X_engineered['row_min']
            
            # Simple interactions
            for i, col1 in enumerate(top_cols):
                for col2 in top_cols[i+1:]:
                    try:
                        X_engineered[f'{col1}_plus_{col2}'] = X_engineered[col1] + X_engineered[col2]
                        X_engineered[f'{col1}_mult_{col2}'] = X_engineered[col1] * X_engineered[col2]
                        X_engineered[f'{col1}_div_{col2}'] = X_engineered[col1] / (X_engineered[col2] + 1e-8)
                    except:
                        continue
        
        # 3. Simple transformations for numeric columns
        for col in self.numeric_cols:
            try:
                if X_engineered[col].min() >= 0:
                    X_engineered[f'{col}_sqrt'] = np.sqrt(X_engineered[col])
                    X_engineered[f'{col}_log'] = np.log1p(X_engineered[col])
                X_engineered[f'{col}_squared'] = X_engineered[col] ** 2
                X_engineered[f'{col}_abs'] = np.abs(X_engineered[col])
            except:
                continue
        
        # 4. Handle categorical columns
        if fit and y is not None:
            for col in self.categorical_cols:
                try:
                    X_engineered[col] = X_engineered[col].astype(str)
                    
                    # Target encoding
                    target_means = y.groupby(X_engineered[col]).mean()
                    X_engineered[f'{col}_target_mean'] = X_engineered[col].map(target_means).fillna(y.mean())
                    self.fitted_transformers[f'target_mean_{col}'] = target_means
                    self.fitted_transformers[f'target_mean_fallback_{col}'] = y.mean()
                    
                    # Frequency encoding
                    freq_map = X_engineered[col].value_counts(normalize=True)
                    X_engineered[f'{col}_freq'] = X_engineered[col].map(freq_map)
                    self.fitted_transformers[f'freq_{col}'] = freq_map
                    
                    # Convert to numeric
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X_engineered[col] = le.fit_transform(X_engineered[col])
                    self.fitted_transformers[f'le_{col}'] = le
                    
                except:
                    continue
        elif not fit:
            # Apply fitted transformers
            for col in self.categorical_cols:
                try:
                    X_engineered[col] = X_engineered[col].astype(str)
                    
                    # Target encoding
                    if f'target_mean_{col}' in self.fitted_transformers:
                        target_means = self.fitted_transformers[f'target_mean_{col}']
                        fallback = self.fitted_transformers[f'target_mean_fallback_{col}']
                        X_engineered[f'{col}_target_mean'] = X_engineered[col].map(target_means).fillna(fallback)
                    
                    # Frequency encoding
                    if f'freq_{col}' in self.fitted_transformers:
                        freq_map = self.fitted_transformers[f'freq_{col}']
                        X_engineered[f'{col}_freq'] = X_engineered[col].map(freq_map).fillna(0)
                    
                    # Label encoding
                    if f'le_{col}' in self.fitted_transformers:
                        le = self.fitted_transformers[f'le_{col}']
                        known_categories = set(le.classes_)
                        X_engineered[col] = X_engineered[col].apply(lambda x: x if x in known_categories else le.classes_[0])
                        X_engineered[col] = le.transform(X_engineered[col])
                except:
                    continue
        
        # 5. Clean up infinite and NaN values
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(0)
        
        # 6. Scale features
        if fit:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
            X_engineered[numeric_cols] = scaler.fit_transform(X_engineered[numeric_cols])
            self.fitted_transformers['scaler'] = scaler
            self.fitted_transformers['numeric_cols'] = numeric_cols
        else:
            if 'scaler' in self.fitted_transformers:
                scaler = self.fitted_transformers['scaler']
                numeric_cols = self.fitted_transformers['numeric_cols']
                available_cols = [col for col in numeric_cols if col in X_engineered.columns]
                if available_cols:
                    X_engineered[available_cols] = scaler.transform(X_engineered[available_cols])
        
        return X_engineered
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform the data."""
        logger.info("Starting simple manual feature engineering")
        
        # Store original feature names
        self.original_feature_names = X.columns.tolist()
        
        # Identify column types
        self.identify_column_types(X)
        
        # Create features
        X_engineered = self.create_simple_features(X, y, fit=True)
        
        # Store feature names and mark as fitted
        self.feature_names = X_engineered.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Simple feature engineering complete. Original: {len(self.original_feature_names)} → Engineered: {len(self.feature_names)}")
        
        return X_engineered
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
            
        logger.info("Transforming new data")
        
        # Create features
        X_engineered = self.create_simple_features(X, fit=False)
        
        # Ensure we have the same columns as training
        for col in self.feature_names:
            if col not in X_engineered.columns:
                X_engineered[col] = 0
        
        # Select only the columns from training (in the same order)
        X_engineered = X_engineered[self.feature_names]
        
        logger.info(f"Transform complete. Output shape: {X_engineered.shape}")
        
        return X_engineered


def process_single_dataset(task: str, fold: int, datadir: Path, engineered_dir: Path, random_state: int = 42):
    """Process a single dataset and fold."""
    logger.info(f"Processing dataset: {task}, fold: {fold}")
    
    task_dir = datadir / task
    fold_dir = task_dir / str(fold)
    
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory {fold_dir} does not exist")
    
    engineered_task_dir = engineered_dir / task
    engineered_fold_dir = engineered_task_dir / str(fold)
    
    # Create output directory
    engineered_fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train = pd.read_parquet(fold_dir / "X_train.parquet")
    y_train = pd.read_parquet(fold_dir / "y_train.parquet").iloc[:, 0]
    X_test = pd.read_parquet(fold_dir / "X_test.parquet")
    
    # Check if y_test exists
    y_test_path = fold_dir / "y_test.parquet"
    y_test = None
    if y_test_path.exists():
        y_test = pd.read_parquet(y_test_path).iloc[:, 0]
    
    logger.info(f"Loaded data: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # Use simple manual feature engineering
    logger.info("Using simple manual feature engineering")
    engineer = SimpleManualFeatureEngineer(random_state=random_state)
    
    # Fit on training data and transform
    X_train_engineered = engineer.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_engineered = engineer.transform(X_test)
    
    # Save engineered data
    X_train_engineered.to_parquet(engineered_fold_dir / "X_train.parquet")
    X_test_engineered.to_parquet(engineered_fold_dir / "X_test.parquet")
    
    # Copy target variables
    y_train.to_frame().to_parquet(engineered_fold_dir / "y_train.parquet")
    if y_test is not None:
        y_test.to_frame().to_parquet(engineered_fold_dir / "y_test.parquet")
    
    # Save the feature engineer for this fold
    joblib.dump(engineer, engineered_fold_dir / "feature_engineer.pkl")
    
    logger.info(f"Successfully processed {task} fold {fold}. Features: {X_train.shape[1]} → {X_train_engineered.shape[1]}")
    logger.info(f"Output saved to: {engineered_fold_dir}")
    
    return X_train_engineered, X_test_engineered, engineer


def main():
    """Main function to process a single dataset."""
    parser = argparse.ArgumentParser(description="Process a single dataset with feature engineering")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1"],
                       help="Dataset to process")
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 11),
                       help="Fold to process (1-10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--method", type=str, default="simple", choices=["simple"],
                       help="Feature engineering method")
    
    args = parser.parse_args()
    
    # Set up paths
    current_dir = Path(__file__).parent
    datadir = current_dir / "data"
    engineered_dir = current_dir / "data_engineered"
    
    try:
        X_train_eng, X_test_eng, engineer = process_single_dataset(
            args.task, args.fold, datadir, engineered_dir, random_state=args.seed
        )
        
        print(f"\n✅ Successfully processed {args.task} fold {args.fold}")
        print(f"📊 Training set shape: {X_train_eng.shape}")
        print(f"📊 Test set shape: {X_test_eng.shape}")
        print(f"💾 Files saved to: data_engineered/{args.task}/{args.fold}/")
        
    except Exception as e:
        print(f"❌ Error processing {args.task} fold {args.fold}: {e}")
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
