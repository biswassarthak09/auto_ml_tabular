import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated imports - Initialize as None for optional dependencies
xgb = None
lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    # Import PyTorch MLP from nas_hpo_optuna
    from nas_hpo_optuna import PyTorchMLP, PYTORCH_AVAILABLE
except ImportError:
    PyTorchMLP = None
    PYTORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalModelTrainer:
    """
    Train and evaluate final models using Meta-Learning predictions.
    This class trains models on x_train/y_train and predicts y on x_test.
    """
    
    def __init__(self, data_dir: str, output_dir: str, meta_learning_model_dir: str):
        """
        Initialize Final Model Trainer
        
        Args:
            data_dir: Directory containing engineered features
            output_dir: Directory to save final model and results
            meta_learning_model_dir: Directory containing trained meta-learning model (REQUIRED)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_learning_model_dir = meta_learning_model_dir
        
        # Simple CUDA check
        self.cuda_available = torch.cuda.is_available() if torch else False
        
        if self.cuda_available:
            logger.info("🚀 CUDA detected! GPU acceleration will be used where available")
        else:
            logger.info("💻 Using CPU-only mode")
        
        # Algorithm mapping for regression - Updated with GPU support
        self.algorithm_map = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'ada_boost': AdaBoostRegressor,
            'decision_tree': DecisionTreeRegressor,
            'ridge': Ridge,
            'lasso': Lasso
        }
        
        # Add GPU-accelerated algorithms if available
        if XGBOOST_AVAILABLE and xgb is not None:
            self.algorithm_map['xgboost'] = xgb.XGBRegressor
            logger.info("✅ XGBoost available for final model training")
            
        if LIGHTGBM_AVAILABLE and lgb is not None:
            self.algorithm_map['lightgbm'] = lgb.LGBMRegressor
            logger.info("✅ LightGBM available for final model training")
            
        if PYTORCH_AVAILABLE and PyTorchMLP:
            self.algorithm_map['mlp'] = PyTorchMLP
            logger.info("✅ PyTorch MLP available for final model training (GPU/CPU auto-detection)")
        
        logger.info(f"📋 Total algorithms available for final training: {len(self.algorithm_map)}")
        
        # Load meta-learning model (MANDATORY - no fallbacks)
        self.meta_model = None
        if not meta_learning_model_dir:
            raise ValueError("meta_learning_model_dir is required! Final model trainer requires meta-learning predictions.")
        
        # Store the directory path (now guaranteed to be not None)
        self.meta_learning_model_dir = meta_learning_model_dir
        
        self.load_meta_learning_model()
        
        if not hasattr(self, 'meta_model') or not self.meta_model:
            raise RuntimeError("Failed to load meta-learning model! Cannot proceed without meta-learning predictions.")
    
    def load_meta_learning_model(self):
        """Load the trained meta-learning model - MANDATORY"""
        logger.info(f"Loading meta-learning model from {self.meta_learning_model_dir}")
        
        # Import meta-learning model
        from meta_learning import AdvancedMetaLearningAutoML
        self.meta_model = AdvancedMetaLearningAutoML(self.meta_learning_model_dir)
        
        # Load the trained model
        if not self.meta_model.load_model(self.meta_learning_model_dir):
            raise RuntimeError(f"Failed to load meta-learning model from {self.meta_learning_model_dir}")
        
        logger.info("✅ Meta-learning model loaded successfully")
    
    def get_algorithm_prediction(self, dataset_name: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get algorithm and hyperparameter prediction for a dataset using meta-learning model.
        
        This method ONLY uses the meta-learning model - no fallbacks.
        """
        
        # Use meta-learning model for prediction
        if not hasattr(self, 'meta_model') or not self.meta_model:
            raise RuntimeError("Meta-learning model not loaded! Cannot proceed without meta-learning predictions.")
        
        logger.info(f"Using meta-learning model for prediction on {dataset_name}")
        
        # Use absolute path for original data directory
        original_data_dir = r"C:\Users\ahker\Desktop\University\auto_ml\auto_ml_tabular\data"
        
        prediction = self.meta_model.predict_for_new_dataset(
            dataset_name, 
            data_dir=str(self.data_dir),
            original_data_dir=original_data_dir
        )
        
        if not prediction:
            raise RuntimeError(f"Meta-learning prediction failed for {dataset_name}. No fallback available.")
        
        algorithm = prediction['algorithm']
        hyperparams = prediction['hyperparams']
        confidence = prediction['confidence']
        
        # Apply hyperparameter constraints to prevent extreme values
        hyperparams = self.constrain_hyperparameters(algorithm, hyperparams)
        
        logger.info(f"Meta-learning prediction for {dataset_name}: {algorithm} (confidence: {confidence:.3f})")
        logger.info(f"Constrained hyperparameters: {hyperparams}")
        
        return algorithm, hyperparams
    
    def constrain_hyperparameters(self, algorithm: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply reasonable constraints to hyperparameters to prevent extreme values
        that lead to excessive training time or poor performance.
        """
        constrained = hyperparams.copy()
        
        if algorithm == 'random_forest':
            # Limit n_estimators to reasonable range
            if 'n_estimators' in constrained:
                original_n_est = constrained['n_estimators']
                constrained['n_estimators'] = max(10, min(500, constrained['n_estimators']))
                if original_n_est != constrained['n_estimators']:
                    logger.info(f"Constrained n_estimators from {original_n_est} to {constrained['n_estimators']}")
            
            # Limit max_depth to prevent overfitting
            if 'max_depth' in constrained:
                constrained['max_depth'] = max(3, min(20, constrained['max_depth']))
        
        elif algorithm == 'gradient_boosting':
            # Limit n_estimators
            if 'n_estimators' in constrained:
                original_n_est = constrained['n_estimators']
                constrained['n_estimators'] = max(50, min(1000, constrained['n_estimators']))
                if original_n_est != constrained['n_estimators']:
                    logger.info(f"Constrained GB n_estimators from {original_n_est} to {constrained['n_estimators']}")
            
            # Ensure reasonable learning rate
            if 'learning_rate' in constrained:
                constrained['learning_rate'] = max(0.01, min(0.3, constrained['learning_rate']))
        
        elif algorithm == 'xgboost':
            # Limit n_estimators
            if 'n_estimators' in constrained:
                constrained['n_estimators'] = max(50, min(1000, constrained['n_estimators']))
            
            # Ensure reasonable learning rate
            if 'learning_rate' in constrained:
                constrained['learning_rate'] = max(0.01, min(0.3, constrained['learning_rate']))
        
        elif algorithm == 'lightgbm':
            # Limit n_estimators
            if 'n_estimators' in constrained:
                constrained['n_estimators'] = max(50, min(1000, constrained['n_estimators']))
            
            # Ensure reasonable learning rate
            if 'learning_rate' in constrained:
                constrained['learning_rate'] = max(0.01, min(0.3, constrained['learning_rate']))
        
        elif algorithm in ['mlp', 'neural_network']:
            # Limit hidden layer sizes
            if 'hidden_layer_sizes' in constrained:
                if isinstance(constrained['hidden_layer_sizes'], (list, tuple)):
                    # Limit each layer size
                    constrained['hidden_layer_sizes'] = tuple(
                        max(10, min(500, size)) for size in constrained['hidden_layer_sizes']
                    )
                else:
                    # Single value
                    constrained['hidden_layer_sizes'] = max(10, min(500, constrained['hidden_layer_sizes']))
        
        return constrained
    
    def create_model(self, algorithm: str, hyperparams: Dict[str, Any]):
        """Create model instance with specified algorithm and hyperparameters - NO FALLBACKS"""
        
        if algorithm not in self.algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm}. Meta-learning model predicted unsupported algorithm.")
        
        model_class = self.algorithm_map[algorithm]
        
        # Filter hyperparameters to only include valid ones for the model
        import inspect
        valid_params = inspect.signature(model_class.__init__).parameters.keys()
        filtered_params = {k: v for k, v in hyperparams.items() if k in valid_params}
        
        # Add random_state if not present and supported
        if 'random_state' in valid_params and 'random_state' not in filtered_params:
            filtered_params['random_state'] = 42
            
        # Add GPU parameters for supported algorithms
        if algorithm == 'xgboost' and self.cuda_available:
            filtered_params.update({
                'device': 'cuda',
                'tree_method': 'gpu_hist'
            })
            logger.info("Using GPU acceleration for XGBoost")
            
        elif algorithm == 'lightgbm' and self.cuda_available:
            filtered_params.update({'device': 'gpu'})
            logger.info("Using GPU acceleration for LightGBM")
            
        elif algorithm == 'mlp' and PYTORCH_AVAILABLE:
            # PyTorch MLP handles device automatically
            filtered_params['device'] = 'auto'
            logger.info("Using PyTorch MLP with auto GPU/CPU detection")
            
        model = model_class(**filtered_params)
        logger.info(f"Created {algorithm} model with params: {filtered_params}")
        return model
    
    def load_dataset_data(self, dataset_name: str, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load training and test data for a specific dataset and fold.
        Returns X_train, X_test, y_train, y_test for training and prediction.
        """
        
        fold_dir = self.data_dir / dataset_name / str(fold)
        
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
        
        # Load engineered features
        X_train = pd.read_parquet(fold_dir / 'X_train_engineered.parquet')
        X_test = pd.read_parquet(fold_dir / 'X_test_engineered.parquet')
        
        # Load target variables and ensure they are Series
        y_train_df = pd.read_parquet(fold_dir / 'y_train.parquet')
        y_test_df = pd.read_parquet(fold_dir / 'y_test.parquet')
        
        # Convert to Series if needed
        if isinstance(y_train_df, pd.DataFrame):
            y_train = y_train_df.iloc[:, 0]  # Get first column as Series
        else:
            y_train = y_train_df
            
        if isinstance(y_test_df, pd.DataFrame):
            y_test = y_test_df.iloc[:, 0]  # Get first column as Series
        else:
            y_test = y_test_df
        
        logger.info(f"Loaded data for {dataset_name} fold {fold}: "
                   f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model performance using regression metrics"""
        
        # Predict y on X_test
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return metrics, y_pred
    
    def train_and_evaluate_single_fold(self, dataset_name: str, fold: int) -> Dict[str, Any]:
        """
        Train and evaluate model on a single fold.
        
        Process:
        1. Get algorithm prediction from NAS-HPO results
        2. Create model with predicted algorithm/hyperparameters
        3. Load X_train, X_test, y_train, y_test
        4. Train model on X_train, y_train
        5. Predict y on X_test and evaluate performance
        """
        
        logger.info(f"Training on {dataset_name} fold {fold}")
        
        # Get algorithm prediction from NAS-HPO results
        algorithm, hyperparams = self.get_algorithm_prediction(dataset_name)
        
        # Create model
        model = self.create_model(algorithm, hyperparams)
        
        # Load data (X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = self.load_dataset_data(dataset_name, fold)
        
        # Train model on X_train, y_train
        logger.info(f"Training {algorithm} model on training data")
        model.fit(X_train, y_train)
        
        # Predict y on X_test and evaluate
        logger.info(f"Predicting y on test data")
        metrics, y_pred = self.evaluate_model(model, X_test, y_test)
        
        # Save trained model
        model_path = self.output_dir / f"{dataset_name}_fold_{fold}_model.joblib"
        joblib.dump(model, model_path)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred
        })
        pred_df.to_csv(self.output_dir / f"{dataset_name}_fold_{fold}_predictions.csv", index=False)
        
        result = {
            'dataset': dataset_name,
            'fold': fold,
            'algorithm': algorithm,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'model_path': str(model_path),
            'train_shape': X_train.shape,
            'test_shape': X_test.shape
        }
        
        logger.info(f"Fold {fold} results: MSE={metrics['mse']:.4f}, R²={metrics['r2_score']:.4f}")
        
        return result
    
    def train_and_evaluate_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Train and evaluate models on all folds of a dataset"""
        
        logger.info(f"Starting training for dataset: {dataset_name}")
        
        results = []
        dataset_dir = self.data_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return results
        
        # Process each fold
        for fold in range(1, 11):  # Assuming 10 folds
            fold_dir = dataset_dir / str(fold)
            
            if not fold_dir.exists():
                logger.warning(f"Fold directory not found: {fold_dir}")
                continue
            
            try:
                fold_result = self.train_and_evaluate_single_fold(dataset_name, fold)
                results.append(fold_result)
            except Exception as e:
                logger.error(f"Error processing {dataset_name} fold {fold}: {str(e)}")
                continue
        
        # Calculate average metrics across folds
        if results:
            avg_metrics = {}
            for metric in results[0]['metrics']:
                avg_metrics[f'avg_{metric}'] = np.mean([r['metrics'][metric] for r in results])
                avg_metrics[f'std_{metric}'] = np.std([r['metrics'][metric] for r in results])
            
            logger.info(f"Dataset {dataset_name} average results:")
            logger.info(f"  Average MSE: {avg_metrics['avg_mse']:.4f} ± {avg_metrics['std_mse']:.4f}")
            logger.info(f"  Average R²: {avg_metrics['avg_r2_score']:.4f} ± {avg_metrics['std_r2_score']:.4f}")
            
            # Save summary
            summary = {
                'dataset': dataset_name,
                'total_folds': len(results),
                'average_metrics': avg_metrics,
                'all_results': results
            }
            
            summary_path = self.output_dir / f"{dataset_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        return results
    
    def train_and_evaluate_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Train and evaluate models on all datasets"""
        
        logger.info("Starting training for all datasets")
        logger.info("=" * 60)
        logger.info("FINAL MODEL TRAINING")
        logger.info("Training on x_train/y_train and predicting y on x_test")
        logger.info("Using meta-learning predictions when available, NAS-HPO as fallback")
        logger.info("=" * 60)
        
        # Get all dataset directories
        datasets = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        logger.info(f"Found datasets: {datasets}")
        
        all_results = {}
        
        for dataset_name in datasets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'='*50}")
            
            dataset_results = self.train_and_evaluate_dataset(dataset_name)
            all_results[dataset_name] = dataset_results
        
        # Save overall summary
        overall_summary_path = self.output_dir / "overall_summary.json"
        with open(overall_summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"\n🎉 Training completed! Results saved to {self.output_dir}")
        
        return all_results
    
    def create_visualization(self, results: Dict[str, List[Dict[str, Any]]]):
        """Create visualizations of the results"""
        
        # Prepare data for visualization
        viz_data = []
        for dataset_name, dataset_results in results.items():
            for result in dataset_results:
                viz_data.append({
                    'dataset': dataset_name,
                    'fold': result['fold'],
                    'algorithm': result['algorithm'],
                    'mse': result['metrics']['mse'],
                    'r2_score': result['metrics']['r2_score'],
                    'mae': result['metrics']['mae'],
                    'rmse': result['metrics']['rmse']
                })
        
        if not viz_data:
            logger.warning("No data available for visualization")
            return
            
        viz_df = pd.DataFrame(viz_data)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² Score by Dataset
        sns.boxplot(data=viz_df, x='dataset', y='r2_score', ax=axes[0, 0])
        axes[0, 0].set_title('R² Score by Dataset')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MSE by Dataset
        sns.boxplot(data=viz_df, x='dataset', y='mse', ax=axes[0, 1])
        axes[0, 1].set_title('MSE by Dataset')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Algorithm Distribution
        algorithm_counts = viz_df['algorithm'].value_counts()
        axes[1, 0].pie(algorithm_counts.values, labels=algorithm_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Algorithm Distribution')
        
        # Performance Scatter Plot
        sns.scatterplot(data=viz_df, x='mse', y='r2_score', hue='dataset', ax=axes[1, 1])
        axes[1, 1].set_title('MSE vs R² Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_results_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualization saved to training_results_visualization.png")

def main():
    """
    Main function to run the final model training.
    
    This function:
    1. Uses NAS-HPO predictions to select algorithms/hyperparameters
    2. Trains models on x_train/y_train 
    3. Predicts y on x_test
    4. Evaluates performance and saves results
    """
    
    # Configuration
    import os
    base_dir = r"C:\Users\ahker\Desktop\University\auto_ml\auto_ml_tabular"
    data_dir = os.path.join(base_dir, "data_engineered_autofeat")
    output_dir = os.path.join(base_dir, "result/final_models")
    meta_learning_model_dir = os.path.join(base_dir, "meta_learning_model")  # Absolute path to trained meta-learning model
    
    logger.info("Starting Final Model Training Pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize trainer
    trainer = FinalModelTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        meta_learning_model_dir=meta_learning_model_dir
    )
    
    # Train and evaluate all datasets
    results = trainer.train_and_evaluate_all_datasets()
    
    # Create visualizations
    trainer.create_visualization(results)
    
    logger.info("✅ Final model training completed successfully!")
    logger.info("✅ Models trained on x_train/y_train and predicted y on x_test")
    logger.info("✅ Results saved with performance metrics")

if __name__ == "__main__":
    main()
