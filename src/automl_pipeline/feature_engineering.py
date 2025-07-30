import os
import pandas as pd
import numpy as np
from autofeat import AutoFeatRegressor
import joblib
from pathlib import Path
import logging
from typing import Dict, Tuple, Any, List, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering pipeline using AutoFeat library
    """
    
    def __init__(self, data_dir: str, output_dir: str, task_type: str = 'regression'):
        """
        Initialize Feature Engineer
        
        Args:
            data_dir: Directory containing datasets
            output_dir: Directory to save processed features
            task_type: 'classification' or 'regression'
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.task_type = task_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store preprocessing components
        self.label_encoders = {}
        self.scaler = None
        self.autofeat_model = None  # Will be created per dataset
    
    def load_data(self, dataset_name: str, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load parquet files for specific dataset and fold"""
        fold_path = self.data_dir / dataset_name / str(fold)
        
        # Load X and y separately
        X_train_path = fold_path / 'X_train.parquet'
        X_test_path = fold_path / 'X_test.parquet'
        y_train_path = fold_path / 'y_train.parquet'
        y_test_path = fold_path / 'y_test.parquet'
        
        # Check if all files exist
        for path in [X_train_path, X_test_path, y_train_path, y_test_path]:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
        
        # Load the data
        X_train = pd.read_parquet(X_train_path)
        X_test = pd.read_parquet(X_test_path)
        y_train = pd.read_parquet(y_train_path).iloc[:, 0]  # Convert to Series
        y_test = pd.read_parquet(y_test_path).iloc[:, 0]    # Convert to Series
        
        return X_train, X_test, y_train, y_test
    
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame using the column mean for numerical data.
        """
        feature_df = df.fillna(df.mean(numeric_only=True))
        return feature_df
    
    def remove_zero_variance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Remove zero variance columns
        """
        # Check for zero variance columns in X_train
        zero_variance_columns = X_train.var(numeric_only=True)[X_train.var(numeric_only=True) == 0]
        
        # Drop zero variance columns
        X_train = X_train.drop(columns=zero_variance_columns.index)
        
        return X_train
    
    def check_for_exponential_values(self, df: pd.DataFrame) -> List[str]:
        """
        Check for exponential values in the DataFrame
        """
        exp_columns = []
        for column in df.columns:
            if df[column].dtype in ['float64', 'float32']:
                has_exponential = df[column].apply(lambda x: '{:e}'.format(x)).str.contains('e').any()
                if has_exponential:
                    exp_columns.append(column)
        return exp_columns
    
    def normalize_data(self, df: pd.DataFrame, is_first_fold: bool = False) -> pd.DataFrame:
        """
        Normalize numerical data using StandardScaler
        """
        # Normalize X and y
        if is_first_fold:
            self.scaler = StandardScaler()
            df_scaled = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns, index=df.index)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Process first fold first.")
            df_scaled = pd.DataFrame(self.scaler.transform(df), columns=df.columns, index=df.index)
        
        # Function to check for exponential values in columns
        exp_columns = self.check_for_exponential_values(df_scaled)
        
        if exp_columns:
            for col in exp_columns:
                # Log-transform the column to reduce the effect of exponential values
                df_scaled[col] = np.log1p(df_scaled[col].abs())  # log1p handles zero and negative values safely
                # Normalize again after transformation
                col_scaler = StandardScaler()
                df_scaled[col] = col_scaler.fit_transform(df_scaled[[col]]).flatten()
                # Round values to 3 decimal points
                df_scaled[col] = df_scaled[col].round(3)
        
        return df_scaled
    
    def check_for_nans_and_infs(self, df: pd.DataFrame, name: str = "data"):
        """
        Check if there are any NaN or infinite values in the DataFrame and clean them.
        """
        # Check for NaNs
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"{name} contains {nan_count} NaN values. Filling with median/mode.")
            # Fill numeric NaNs with median, categorical with mode
            for col in df.columns:
                if df[col].dtype in ['object', 'category']:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna('missing')
                else:
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(median_val)
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number]).values).any():
            logger.warning(f"{name} contains infinite values. Clipping to finite range.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # Replace positive infinity with 99th percentile
                col_values = df[col][np.isfinite(df[col])]
                if len(col_values) > 0:
                    p99 = np.percentile(col_values, 99)
                    p1 = np.percentile(col_values, 1)
                    df[col] = df[col].replace([np.inf], p99)
                    df[col] = df[col].replace([-np.inf], p1)
                else:
                    # If all values are infinite, replace with 0
                    df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        # Final check
        final_nan_count = df.isnull().sum().sum()
        final_inf_count = np.isinf(df.select_dtypes(include=[np.number]).values).sum()
        
        if final_nan_count == 0 and final_inf_count == 0:
            logger.info(f"No NaNs or infinities detected in {name}")
        else:
            logger.warning(f"{name} still has {final_nan_count} NaNs and {final_inf_count} infinities after cleaning")
        
        return df
    
    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       is_first_fold: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocessing pipeline
        """
        logger.info("Starting preprocessing...")
        
        # Handle categorical variables first
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        for col in categorical_cols:
            if is_first_fold:
                le = LabelEncoder()
                # Fit on combined train+test to handle unseen categories
                combined_values = pd.concat([X_train_processed[col], X_test_processed[col]]).astype(str)
                le.fit(combined_values)
                self.label_encoders[col] = le
            
            # Handle unseen categories in test data
            X_train_col = X_train_processed[col].astype(str)
            X_test_col = X_test_processed[col].astype(str)
            
            # Check for unseen categories in test data
            train_categories = set(self.label_encoders[col].classes_)
            test_categories = set(X_test_col.unique())
            unseen_categories = test_categories - train_categories
            
            if unseen_categories:
                # Replace unseen categories with most frequent category
                most_frequent = X_train_col.mode()[0] if not X_train_col.empty else 'unknown'
                X_test_col = X_test_col.replace(list(unseen_categories), most_frequent)
            
            X_train_processed[col] = self.label_encoders[col].transform(X_train_col)
            X_test_processed[col] = self.label_encoders[col].transform(X_test_col)
        
        # 1. Impute missing values
        X_train_processed = self.impute_missing_values(X_train_processed)
        X_test_processed = self.impute_missing_values(X_test_processed)
        
        # 2. Remove zero variance - only on training data
        if is_first_fold:
            X_train_processed = self.remove_zero_variance(X_train_processed)
            # Store the columns for this specific dataset
            self.selected_columns = X_train_processed.columns
        else:
            # For subsequent folds, remove zero variance on their own columns
            X_train_processed = self.remove_zero_variance(X_train_processed)
            self.selected_columns = X_train_processed.columns
        
        # Apply same column selection to test data (using current dataset's columns)
        X_test_processed = X_test_processed[self.selected_columns]
        
        # Ensure DataFrames (not Series)
        if isinstance(X_test_processed, pd.Series):
            X_test_processed = X_test_processed.to_frame()
        if isinstance(X_train_processed, pd.Series):
            X_train_processed = X_train_processed.to_frame()
        
        # 3. Normalize data
        X_train_processed = self.normalize_data(X_train_processed, is_first_fold)
        X_test_processed = self.normalize_data(X_test_processed, is_first_fold=False)
        
        # 4. Check for NaNs and infinities - and clean them
        X_train_processed = self.check_for_nans_and_infs(X_train_processed, "X_train")
        X_test_processed = self.check_for_nans_and_infs(X_test_processed, "X_test")
        
        logger.info(f"Preprocessing completed. Shape: {X_train_processed.shape}")
        return X_train_processed, X_test_processed
    
    def engineer_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, is_first_fold: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply CONSERVATIVE feature engineering using AutoFeat to avoid overfitting
        """
        logger.info("Starting conservative AutoFeat feature engineering...")
        
        # Create or reuse AutoFeat model for each dataset
        if is_first_fold or self.autofeat_model is None:
            logger.info("Creating ENHANCED AutoFeat model with modest feature engineering...")
            self.autofeat_model = AutoFeatRegressor(
                feateng_steps=1,                    # Allow 1 step of feature engineering
                featsel_runs=5,                     # More thorough feature selection
                max_gb=2,                           # Slightly more memory for better features
                n_jobs=1,                           # Single job to avoid memory issues
                transformations=["log", "abs", "sqrt"],  # Removed "1/" to prevent division by zero
                verbose=1
            )
            logger.info("Fitting AutoFeat on first fold of dataset...")
            X_train_engineered = self.autofeat_model.fit_transform(X_train, np.array(y_train))
            logger.info("AutoFeat fitted successfully")
        else:
            # Transform using fitted model for subsequent folds of same dataset
            X_train_engineered = self.autofeat_model.transform(X_train)
        
        # Transform test data
        X_test_engineered = self.autofeat_model.transform(X_test)
        
        # CRITICAL: Clean infinite and NaN values created by AutoFeat transformations
        # AutoFeat can create "1/x" features which lead to infinity when x approaches 0
        if isinstance(X_train_engineered, np.ndarray):
            X_train_engineered = pd.DataFrame(X_train_engineered, index=X_train.index)
        if isinstance(X_test_engineered, np.ndarray):
            X_test_engineered = pd.DataFrame(X_test_engineered, index=X_test.index)
        
        # Clean infinite and NaN values from AutoFeat transformations
        X_train_engineered = self.check_for_nans_and_infs(X_train_engineered, "AutoFeat X_train")
        X_test_engineered = self.check_for_nans_and_infs(X_test_engineered, "AutoFeat X_test")
        
        # Convert to DataFrames and ensure proper types
        if not isinstance(X_train_engineered, pd.DataFrame):
            X_train_engineered = pd.DataFrame(X_train_engineered, index=X_train.index)
        if not isinstance(X_test_engineered, pd.DataFrame):
            X_test_engineered = pd.DataFrame(X_test_engineered, index=X_test.index)
        
        # Ensure we return DataFrames (force conversion if needed)
        X_train_final = pd.DataFrame(X_train_engineered)
        X_test_final = pd.DataFrame(X_test_engineered)
        
        # ENHANCED feature limiting - allow more features for better performance
        max_features = min(50, X_train_final.shape[1])  # Increased from 25 to 50
        if X_train_final.shape[1] > max_features:
            # Keep only the first max_features (AutoFeat orders by importance)
            X_train_final = pd.DataFrame(X_train_final.iloc[:, :max_features])
            X_test_final = pd.DataFrame(X_test_final.iloc[:, :max_features])
            logger.info(f"LIMITED features to {max_features} for optimal performance")
        
        logger.info(f"Original features: {X_train.shape[1]}")
        logger.info(f"Engineered features: {X_train_final.shape[1]}")
        
        return X_train_final, X_test_final
    
    def save_engineered_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           dataset_name: str, fold: int):
        """Save engineered features and targets"""
        fold_dir = self.output_dir / dataset_name / str(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save engineered features
        X_train.to_parquet(fold_dir / 'X_train_engineered.parquet')
        X_test.to_parquet(fold_dir / 'X_test_engineered.parquet')
        
        # Save targets
        y_train.to_frame('target').to_parquet(fold_dir / 'y_train.parquet')
        y_test.to_frame('target').to_parquet(fold_dir / 'y_test.parquet')
        
        # Save feature names
        feature_names = {
            'original_features': list(X_train.columns),
            'n_features': X_train.shape[1]
        }
        joblib.dump(feature_names, fold_dir / 'feature_info.pkl')
        
        logger.info(f"Saved engineered data for {dataset_name} fold {fold}")
    
    def process_all_datasets(self):
        """Process all datasets, folds, and splits"""
        # Get all dataset directories
        dataset_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        autofeat_models = {}  # Store models per dataset
        
        for i, dataset_dir in enumerate(dataset_dirs):
            dataset_name = dataset_dir.name
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Reset preprocessing state for each new dataset
            self.label_encoders = {}
            self.scaler = None
            self.selected_columns = None
            self.autofeat_model = None  # Reset AutoFeat model for each dataset
            
            # Process each fold
            for fold in range(1, 11):  # Assuming folds 1-10
                try:
                    logger.info(f"Processing fold {fold}")
                    
                    # Load training and test data
                    X_train, X_test, y_train, y_test = self.load_data(dataset_name, fold)
                    
                    # First fold of each dataset fits the preprocessors
                    is_first_fold = (fold == 1)
                    
                    # Preprocess data
                    X_train_processed, X_test_processed = self.preprocess_data(
                        X_train, X_test, is_first_fold
                    )
                    
                    # Engineer features (AutoFeat fit on first fold of each dataset)
                    X_train_eng, X_test_eng = self.engineer_features(
                        X_train_processed, y_train, X_test_processed, is_first_fold
                    )
                    
                    # Save engineered data
                    self.save_engineered_data(
                        X_train_eng, X_test_eng,
                        y_train, y_test,
                        dataset_name, fold
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing {dataset_name} fold {fold}: {str(e)}")
                    continue
            
            # Save the AutoFeat model for this dataset
            if self.autofeat_model is not None:
                autofeat_models[dataset_name] = self.autofeat_model
        
        # Save all AutoFeat models
        joblib.dump(autofeat_models, self.output_dir / 'autofeat_models.pkl')
        logger.info("Feature engineering completed for all datasets!")

def main():
    """Main execution function"""
    # Configuration
    DATA_DIR = "data"  # Directory containing your 5 datasets
    OUTPUT_DIR = "data_engineered_autofeat"  # Match the existing directory structure
    TASK_TYPE = "regression"

    # Initialize and run feature engineering
    fe = FeatureEngineer(DATA_DIR, OUTPUT_DIR, TASK_TYPE)
    fe.process_all_datasets()

if __name__ == "__main__":
    main()
