"""
AutoML Test Pipeline
Complete testing pipeline for new/unknown datasets

Components:
1. Feature Engineering (using trained models from training phase)
2. Final Model Prediction (using trained models and meta-learning)

Usage:
    python test_pipeline.py --dataset_path path/to/new/dataset
    
    Dataset should contain:
    - X_train.parquet, X_test.parquet, y_train.parquet, y_test.parquet
    OR
    - X.parquet, y.parquet (will be split automatically)
"""

import os
import sys
import argparse
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoMLTestPipeline:
    """AutoML pipeline for testing on new datasets"""
    
    def __init__(self):
        """Initialize the test pipeline"""
        self.config = {
            'trained_fe_models_dir': 'data_engineered_autofeat',
            'meta_learning_model_dir': 'meta_learning_model', 
            'final_models_dir': 'result/final_models',
            'nas_hpo_results': 'nas_hpo_results/all_optimization_results.json'
        }
        
        # Load trained components
        self._load_trained_components()
    
    def _load_trained_components(self):
        """Load all trained components from training phase"""
        
        logger.info("Loading trained components...")
        
        try:
            # Load AutoFeat models
            import joblib
            autofeat_models_path = Path(self.config['trained_fe_models_dir']) / 'autofeat_models.pkl'
            if autofeat_models_path.exists():
                self.autofeat_models = joblib.load(autofeat_models_path)
                logger.info("✅ AutoFeat models loaded")
            else:
                self.autofeat_models = None
                logger.warning("❌ AutoFeat models not found")
            
            # Load meta-learning model
            from meta_learning import AdvancedMetaLearningAutoML
            self.meta_learner = AdvancedMetaLearningAutoML(self.config['meta_learning_model_dir'])
            
            meta_model_path = Path(self.config['meta_learning_model_dir'])
            if meta_model_path.exists() and any(meta_model_path.iterdir()):
                success = self.meta_learner.load_model(self.config['meta_learning_model_dir'])
                if success:
                    logger.info("✅ Meta-learning model loaded")
                else:
                    self.meta_learner = None
                    logger.warning("❌ Failed to load meta-learning model")
            else:
                self.meta_learner = None
                logger.warning("❌ Meta-learning model not found")
            
            # Load NAS-HPO results as fallback
            nas_hpo_path = Path(self.config['nas_hpo_results'])
            if nas_hpo_path.exists():
                import json
                with open(nas_hpo_path, 'r') as f:
                    self.nas_hpo_results = json.load(f)
                logger.info("✅ NAS-HPO results loaded as fallback")
            else:
                self.nas_hpo_results = None
                logger.warning("❌ NAS-HPO results not found")
                
        except Exception as e:
            logger.error(f"Error loading trained components: {str(e)}")
            raise
    
    def prepare_new_dataset(self, dataset_path: str, dataset_name: str = "unknown_dataset") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare a new dataset for testing
        
        Args:
            dataset_path: Path to dataset directory or files
            dataset_name: Name for the dataset
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        
        dataset_path_obj = Path(dataset_path)
        
        logger.info(f"Preparing dataset: {dataset_name}")
        
        try:
            # Case 1: Pre-split data (preferred)
            if (dataset_path_obj / 'X_train.parquet').exists():
                X_train = pd.read_parquet(dataset_path_obj / 'X_train.parquet')
                X_test = pd.read_parquet(dataset_path_obj / 'X_test.parquet')
                y_train = pd.read_parquet(dataset_path_obj / 'y_train.parquet')
                y_test = pd.read_parquet(dataset_path_obj / 'y_test.parquet')
                
                # Convert y to series if needed
                if isinstance(y_train, pd.DataFrame):
                    y_train = y_train.iloc[:, 0]
                if isinstance(y_test, pd.DataFrame):
                    y_test = y_test.iloc[:, 0]
                
                logger.info(f"✅ Loaded pre-split data: Train {X_train.shape}, Test {X_test.shape}")
                
            # Case 2: Single dataset files - need to split
            elif (dataset_path_obj / 'X.parquet').exists():
                X = pd.read_parquet(dataset_path_obj / 'X.parquet')
                y = pd.read_parquet(dataset_path_obj / 'y.parquet')
                
                if isinstance(y, pd.DataFrame):
                    y = y.iloc[:, 0]
                
                # Split the data
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=None
                )
                
                logger.info(f"✅ Split dataset: Train {X_train.shape}, Test {X_test.shape}")
                
            # Case 3: CSV files
            elif (dataset_path_obj / 'data.csv').exists() or dataset_path_obj.suffix == '.csv':
                if dataset_path_obj.suffix == '.csv':
                    df = pd.read_csv(dataset_path_obj)
                else:
                    df = pd.read_csv(dataset_path_obj / 'data.csv')
                
                # Assume last column is target
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                
                # Split the data
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                logger.info(f"✅ Loaded CSV and split: Train {X_train.shape}, Test {X_test.shape}")
                
            else:
                raise FileNotFoundError(f"No recognized data files found in {dataset_path_obj}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise
    
    def engineer_features_for_new_dataset(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                         y_train: pd.Series, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply feature engineering to new dataset using trained models
        """
        
        logger.info(f"Engineering features for {dataset_name}...")
        
        try:
            from feature_engineering import FeatureEngineer
            
            # Create a temporary feature engineer
            fe = FeatureEngineer(
                data_dir="temp",  # Not used for new datasets
                output_dir="temp_output",  # Not used
                task_type='regression'
            )
            
            # Apply preprocessing (this doesn't require trained models)
            X_train_processed, X_test_processed = fe.preprocess_data(
                X_train, X_test, is_first_fold=True
            )
            
            logger.info(f"✅ Feature engineering completed: {X_train_processed.shape}")
            
            return X_train_processed, X_test_processed
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            # Fallback to basic preprocessing
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            X_train_fallback = X_train.copy()
            X_test_fallback = X_test.copy()
            
            # Handle categorical variables
            categorical_cols = X_train_fallback.select_dtypes(include=['object', 'category']).columns
            label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                combined_values = pd.concat([X_train_fallback[col], X_test_fallback[col]]).astype(str)
                le.fit(combined_values)
                X_train_fallback[col] = le.transform(X_train_fallback[col].astype(str))
                X_test_fallback[col] = le.transform(X_test_fallback[col].astype(str))
                label_encoders[col] = le
            
            # Scale features
            scaler = StandardScaler()
            X_train_fallback = pd.DataFrame(
                scaler.fit_transform(X_train_fallback),
                columns=X_train_fallback.columns,
                index=X_train_fallback.index
            )
            X_test_fallback = pd.DataFrame(
                scaler.transform(X_test_fallback),
                columns=X_test_fallback.columns,
                index=X_test_fallback.index
            )
            
            logger.info(f"✅ Fallback preprocessing completed: {X_train_fallback.shape}")
            return X_train_fallback, X_test_fallback
    
    def predict_best_algorithm(self, dataset_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """
        Predict the best algorithm and hyperparameters for the new dataset
        """
        
        logger.info(f"Predicting best algorithm for {dataset_name}...")
        
        # Try meta-learning first
        if self.meta_learner and hasattr(self.meta_learner, 'is_trained') and self.meta_learner.is_trained:
            try:
                prediction = self.meta_learner.predict_for_new_dataset(
                    dataset_name,
                    data_dir="temp",  # Temporary
                    original_data_dir="temp"
                )
                
                if prediction:
                    logger.info(f"✅ Meta-learning prediction: {prediction['algorithm']}")
                    return prediction['algorithm'], prediction['hyperparams']
                    
            except Exception as e:
                logger.warning(f"Meta-learning prediction failed: {str(e)}")
        
        # Fallback to simple algorithm selection based on dataset characteristics
        n_samples, n_features = X_train.shape
        
        if n_samples < 1000:
            algorithm = 'random_forest'
            hyperparams = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        elif n_features > 50:
            algorithm = 'ridge'
            hyperparams = {'alpha': 1.0, 'random_state': 42}
        else:
            algorithm = 'gradient_boosting'
            hyperparams = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}
        
        logger.info(f"✅ Fallback algorithm selected: {algorithm}")
        return algorithm, hyperparams
    
    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series,
                          algorithm: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the final model and evaluate performance
        """
        
        logger.info(f"Training {algorithm} model...")
        
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import Ridge
            from sklearn.neural_network import MLPRegressor
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            # Model mapping
            model_map = {
                'random_forest': RandomForestRegressor,
                'gradient_boosting': GradientBoostingRegressor,
                'ridge': Ridge,
                'mlp': MLPRegressor
            }
            
            if algorithm not in model_map:
                logger.warning(f"Unknown algorithm {algorithm}, using RandomForest")
                algorithm = 'random_forest'
                hyperparams = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            
            # Create and train model
            model_class = model_map[algorithm]
            
            # Filter hyperparameters to valid ones
            import inspect
            valid_params = inspect.signature(model_class.__init__).parameters.keys()
            filtered_params = {k: v for k, v in hyperparams.items() if k in valid_params}
            
            model = model_class(**filtered_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            results = {
                'algorithm': algorithm,
                'hyperparams': filtered_params,
                'train_r2': float(r2_score(y_train, y_pred_train)),
                'test_r2': float(r2_score(y_test, y_pred_test)),
                'train_mse': float(mean_squared_error(y_train, y_pred_train)),
                'test_mse': float(mean_squared_error(y_test, y_pred_test)),
                'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
                'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
                'overfitting_gap': float(r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test)),
                'model_trained': True
            }
            
            logger.info(f"✅ Model trained successfully:")
            logger.info(f"   Train R²: {results['train_r2']:.4f}")
            logger.info(f"   Test R²: {results['test_r2']:.4f}")
            logger.info(f"   Overfitting gap: {results['overfitting_gap']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {
                'algorithm': algorithm,
                'error': str(e),
                'model_trained': False
            }
    
    def test_new_dataset(self, dataset_path: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete test pipeline for a new dataset
        """
        
        if dataset_name is None:
            dataset_name = Path(dataset_path).name
        
        start_time = time.time()
        
        print(f"🧪 TESTING NEW DATASET: {dataset_name}")
        print("=" * 60)
        print(f"Dataset path: {dataset_path}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # Step 1: Prepare dataset
            print("📊 STEP 1: DATASET PREPARATION")
            print("-" * 40)
            X_train, X_test, y_train, y_test = self.prepare_new_dataset(dataset_path, dataset_name)
            
            # Step 2: Feature engineering
            print("\\n🔧 STEP 2: FEATURE ENGINEERING")
            print("-" * 40)
            X_train_eng, X_test_eng = self.engineer_features_for_new_dataset(
                X_train, X_test, y_train, dataset_name
            )
            
            # Step 3: Algorithm prediction
            print("\\n🧠 STEP 3: ALGORITHM SELECTION")
            print("-" * 40)
            algorithm, hyperparams = self.predict_best_algorithm(dataset_name, X_train_eng, y_train)
            
            # Step 4: Training and evaluation
            print("\\n🎯 STEP 4: MODEL TRAINING & EVALUATION")
            print("-" * 40)
            results = self.train_and_evaluate(
                X_train_eng, X_test_eng, y_train, y_test, algorithm, hyperparams
            )
            
            # Final results
            end_time = time.time()
            duration = end_time - start_time
            
            print("\\n" + "=" * 60)
            print("🎯 TEST RESULTS SUMMARY")
            print("=" * 60)
            
            if results.get('model_trained', False):
                print(f"✅ Dataset: {dataset_name}")
                print(f"📊 Data shape: Train {X_train.shape}, Test {X_test.shape}")
                print(f"🤖 Algorithm: {results['algorithm']}")
                print(f"📈 Test R²: {results['test_r2']:.4f}")
                print(f"📉 Test MSE: {results['test_mse']:.4f}")
                print(f"🎯 Overfitting gap: {results['overfitting_gap']:.4f}")
                
                # Performance assessment
                if results['test_r2'] > 0.8:
                    print("🎉 EXCELLENT: Very strong predictive performance!")
                elif results['test_r2'] > 0.6:
                    print("✅ GOOD: Strong predictive performance!")
                elif results['test_r2'] > 0.3:
                    print("🟡 MODERATE: Decent predictive performance")
                elif results['test_r2'] > 0:
                    print("🔧 WEAK: Low predictive performance - consider data quality")
                else:
                    print("❌ POOR: Negative R² - model performs worse than mean predictor")
                
                if results['overfitting_gap'] > 0.3:
                    print("⚠️  WARNING: High overfitting detected")
                elif results['overfitting_gap'] > 0.1:
                    print("🟡 NOTICE: Moderate overfitting detected")
                else:
                    print("✅ GOOD: Low overfitting")
                    
            else:
                print(f"❌ FAILED: {results.get('error', 'Unknown error')}")
            
            print(f"\\n⏱️  Duration: {duration:.1f} seconds")
            print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Save results
            results.update({
                'dataset_name': dataset_name,
                'dataset_path': str(dataset_path),
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save to file
            import json
            results_file = f"test_results_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\\n📄 Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Test pipeline failed: {str(e)}")
            print(f"❌ TEST FAILED: {str(e)}")
            return {
                'dataset_name': dataset_name,
                'error': str(e),
                'model_trained': False
            }

def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(description='AutoML Test Pipeline for New Datasets')
    parser.add_argument('--dataset_path', required=True, help='Path to dataset directory or file')
    parser.add_argument('--dataset_name', help='Name for the dataset (optional)')
    
    args = parser.parse_args()
    
    # Initialize and run test pipeline
    test_pipeline = AutoMLTestPipeline()
    results = test_pipeline.test_new_dataset(args.dataset_path, args.dataset_name)
    
    return results

if __name__ == "__main__":
    main()
