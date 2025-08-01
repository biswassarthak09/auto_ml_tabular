import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import joblib
import logging
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated imports - Initialize as None for optional dependencies
xgb = None
lgb = None  
torch = None
nn = None
optim = None
DataLoader = None
TensorDataset = None
StandardScaler = None

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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PyTorchMLP:
    """GPU-accelerated MLP using PyTorch"""
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, 
                 max_iter=200, batch_size=32, device='auto', early_stopping=True, 
                 validation_fraction=0.1, n_iter_no_change=10, alpha=0.0001):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use PyTorchMLP.")
            
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.alpha = alpha
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.scaler = StandardScaler()
        
    def _create_model(self, input_size):
        """Create PyTorch model"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
            
        layers = []
        prev_size = input_size
        
        for size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(self.alpha))
            prev_size = size
            
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        return nn.Sequential(*layers).to(self.device)
    
    def fit(self, X, y):
        """Fit the model"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
            
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        # Create model
        self.model = self._create_model(X_scaled.shape[1])
        
        # Split for validation if early stopping
        if self.early_stopping:
            val_size = int(len(X_tensor) * self.validation_fraction)
            train_size = len(X_tensor) - val_size
            
            X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
            y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
        else:
            X_train, y_train = X_tensor, y_tensor
            X_val, y_val = None, None
            
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if self.early_stopping and X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.n_iter_no_change:
                    break
                    
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
            
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        if hasattr(X, 'values'):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy().flatten()

class NASHPOOptimizer:
    """
    Neural Architecture Search and Hyperparameter Optimization using Optuna
    Enhanced with GPU acceleration
    """
    
    def __init__(self, data_dir: str, results_dir: str, task_type: str = 'regression'):
        """
        Initialize NAS-HPO Optimizer
        
        Args:
            data_dir: Directory containing engineered features
            results_dir: Directory to save optimization results
            task_type: 'regression'
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.task_type = task_type
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple CUDA check
        if PYTORCH_AVAILABLE:
            self.cuda_available = torch.cuda.is_available()
        else:
            self.cuda_available = False
            
        if self.cuda_available:
            logger.info("🚀 CUDA detected! GPU acceleration will be used where available")
        else:
            logger.info("💻 Using CPU-only mode")
        
        # Setup algorithms
        self._setup_algorithms()
        
        self.optimization_results = []
    
    def _setup_algorithms(self):
        """Setup algorithm space with optional GPU acceleration"""
        
        # Core algorithms (always available)
        self.algorithm_space = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'ada_boost': AdaBoostRegressor,
            'decision_tree': DecisionTreeRegressor,
            'knn': KNeighborsRegressor,
            'ridge': Ridge,
            'lasso': Lasso,
        }
        
        # Add GPU-accelerated algorithms if available
        if XGBOOST_AVAILABLE:
            self.algorithm_space['xgboost'] = xgb.XGBRegressor
            logger.info("✅ XGBoost available")
            
        if LIGHTGBM_AVAILABLE:
            self.algorithm_space['lightgbm'] = lgb.LGBMRegressor
            logger.info("✅ LightGBM available")
            
        if PYTORCH_AVAILABLE:
            self.algorithm_space['mlp'] = PyTorchMLP  # GPU/CPU auto-detection
            logger.info("✅ PyTorch MLP available (GPU/CPU auto-detection)")
        
        logger.info(f"📋 Total algorithms available: {len(self.algorithm_space)}")
        
        # Keep track of core algorithms for high-performance testing
        # Focus on diverse, high-performance algorithms: XGBoost, Random Forest, MLP, KNN
        self.core_algorithms = {
            'random_forest': RandomForestRegressor,  # Reliable ensemble method
            'knn': KNeighborsRegressor,              # Instance-based learning
        }
        
        # Add XGBoost - usually the top performer
        if XGBOOST_AVAILABLE:
            self.core_algorithms['xgboost'] = xgb.XGBRegressor  # Gradient boosting champion
        
        # Add MLP - neural network for complex patterns
        if PYTORCH_AVAILABLE:
            self.core_algorithms['mlp'] = PyTorchMLP  # Deep learning option
    
    def load_fold_data(self, dataset_name: str, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load engineered data for specific dataset and fold"""
        fold_dir = self.data_dir / dataset_name / str(fold)
        
        X_train = pd.read_parquet(fold_dir / 'X_train_engineered.parquet')
        X_test = pd.read_parquet(fold_dir / 'X_test_engineered.parquet')
        y_train = pd.read_parquet(fold_dir / 'y_train.parquet')['target']
        y_test = pd.read_parquet(fold_dir / 'y_test.parquet')['target']
        
        return X_train, X_test, y_train, y_test
    
    def suggest_hyperparameters(self, trial: optuna.Trial, algorithm: str) -> Dict[str, Any]:
        """Suggest hyperparameters based on algorithm type"""
        params = {}
        
        if algorithm == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42
            }
        
        elif algorithm == 'gradient_boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }
        
        elif algorithm == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbosity': 0
            }
            # Use GPU if available
            if self.cuda_available:
                params.update({
                    'device': 'cuda',
                    'tree_method': 'gpu_hist'
                })
        
        elif algorithm == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
                'random_state': 42,
                'verbosity': -1
            }
            # Use GPU if available
            if self.cuda_available:
                params.update({'device': 'gpu'})
        
        elif algorithm == 'mlp':
            # PyTorch MLP with GPU/CPU auto-detection
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_layer_sizes = []
            
            for i in range(n_layers):
                layer_size = trial.suggest_int(f'layer_{i}_size', 50, 512)
                hidden_layer_sizes.append(layer_size)
            
            params = {
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'max_iter': trial.suggest_int('max_iter', 100, 500),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                'device': 'auto',  # Auto GPU/CPU detection
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 15
            }
        
        elif algorithm == 'svm':
            params = {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid', 'linear']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            }
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
        
        elif algorithm == 'ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
                'random_state': 42
            }
        
        elif algorithm == 'lasso':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
                'random_state': 42
            }
        
        elif algorithm == 'ada_boost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0),
                'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
                'random_state': 42
            }
        
        elif algorithm == 'decision_tree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error']),
                'random_state': 42
            }
        
        elif algorithm == 'knn':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'p': trial.suggest_int('p', 1, 2)  # 1 for manhattan, 2 for euclidean
            }
        
        return params
    
    def objective(self, trial: optuna.Trial, dataset_name: str, algorithm: str) -> float:
        """Objective function for optimization"""
        try:
            # Get hyperparameters
            params = self.suggest_hyperparameters(trial, algorithm)
            
            # Initialize model
            model_class = self.algorithm_space[algorithm]
            model = model_class(**params)
            
            # Use only subset of folds for faster evaluation
            # Use 3 folds instead of 10 for initial screening
            test_folds = [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]  # Representative subset
            fold_scores = []
            
            for fold in test_folds:
                try:
                    X_train, X_test, y_train, y_test = self.load_fold_data(dataset_name, fold)
                    
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate multiple metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Use R² score directly for optimization (0 to 1, higher is better)
                    score = r2
                    
                    # Store individual metrics for this fold
                    fold_metrics = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'composite_score': score
                    }
                    fold_scores.append(fold_metrics)
                    
                except Exception as e:
                    logger.warning(f"Error in fold {fold}: {str(e)}")
                    continue
            
            if not fold_scores:
                return 0.0
            
            # Calculate mean metrics across folds
            mean_mse = np.mean([fold['mse'] for fold in fold_scores])
            mean_mae = np.mean([fold['mae'] for fold in fold_scores])
            mean_r2 = np.mean([fold['r2'] for fold in fold_scores])
            mean_composite = np.mean([fold['composite_score'] for fold in fold_scores])
            
            # Store detailed metrics in trial user attributes for later retrieval
            trial.set_user_attr('mean_mse', mean_mse)
            trial.set_user_attr('mean_mae', mean_mae)
            trial.set_user_attr('mean_r2', mean_r2)
            trial.set_user_attr('fold_details', fold_scores)
            
            # Return composite score for optimization
            return float(mean_composite)
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return 0.0
    
    def optimize_algorithm(self, dataset_name: str, algorithm: str, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific algorithm"""
        logger.info(f"Optimizing {algorithm} for {dataset_name}")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self.objective(trial, dataset_name, algorithm),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Get best results
        best_params = study.best_params
        best_score = study.best_value
        best_trial = study.best_trial
        
        # Extract detailed metrics from best trial
        best_metrics = {
            'composite_score': best_score,
            'mse': best_trial.user_attrs.get('mean_mse', None),
            'mae': best_trial.user_attrs.get('mean_mae', None),
            'r2': best_trial.user_attrs.get('mean_r2', None),
            'fold_details': best_trial.user_attrs.get('fold_details', [])
        }
        
        result = {
            'dataset': dataset_name,
            'algorithm': algorithm,
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'n_trials': n_trials
        }
        
        logger.info(f"Best {algorithm} score for {dataset_name}: {best_score:.4f}")
        
        return result
    
    def test_single_dataset(self, dataset_name: str, n_trials: int = 5, use_core_only: bool = True):
        """Quick test on a single dataset with selected algorithms"""
        logger.info(f"Quick test on dataset: {dataset_name}")
        
        if use_core_only:
            # Use only core high-performance algorithms
            test_algorithms = list(self.core_algorithms.keys())
            logger.info("Using core algorithms: gradient_boosting, random_forest, ada_boost, mlp, decision_tree, knn")
        else:
            # Use fast algorithms for quick testing
            test_algorithms = ['ridge', 'lasso', 'decision_tree', 'knn']
            logger.info("Using fast algorithms for quick baseline testing")
        
        results = []
        for algorithm in test_algorithms:
            try:
                logger.info(f"Testing {algorithm}...")
                result = self.optimize_algorithm(dataset_name, algorithm, n_trials)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error testing {algorithm}: {str(e)}")
                continue
        
        # Show quick results
        if results:
            best = max(results, key=lambda x: x['best_score'])
            logger.info(f"Quick test complete. Best: {best['algorithm']} with score {best['best_score']:.4f}")
            
            # Save the quick test results
            self.save_dataset_results(f"{dataset_name}_quick_test", results)
            
            # Also save a simple summary
            summary_file = self.results_dir / f'{dataset_name}_quick_test_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(f"Quick Test Results for {dataset_name}\n")
                f.write("=" * 40 + "\n\n")
                f.write("Algorithm Performance Ranking:\n")
                f.write("-" * 30 + "\n")
                for i, result in enumerate(sorted(results, key=lambda x: x['best_score'], reverse=True)):
                    metrics = result.get('best_metrics', {})
                    f.write(f"{i+1}. {result['algorithm']}: {result['best_score']:.4f}\n")
                    if metrics.get('r2') is not None:
                        f.write(f"   R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}\n")
                
                f.write(f"\nBest Algorithm: {best['algorithm']}\n")
                f.write(f"Best Composite Score: {best['best_score']:.4f}\n")
                
                # Add detailed metrics for best algorithm
                best_metrics = best.get('best_metrics', {})
                if best_metrics.get('r2') is not None:
                    f.write(f"Best R²: {best_metrics['r2']:.4f}\n")
                    f.write(f"Best MSE: {best_metrics['mse']:.4f}\n")
                    f.write(f"Best MAE: {best_metrics['mae']:.4f}\n")
                
                f.write(f"Best Params: {json.dumps(best['best_params'], indent=2)}\n")
        
        return results
    
    def run_nas_hpo(self, n_trials_per_algorithm: int = 100, use_core_only: bool = True):
        """Run NAS and HPO for all datasets and algorithms"""
        # Get all dataset directories (exclude any non-dataset directories)
        dataset_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
        
        # Choose algorithm set
        algorithms_to_test = list(self.core_algorithms.keys()) if use_core_only else list(self.algorithm_space.keys())
        
        logger.info(f"Running NAS-HPO with {'core' if use_core_only else 'all'} algorithms: {algorithms_to_test}")
        logger.info(f"Processing {len(dataset_dirs)} datasets with {len(algorithms_to_test)} algorithms each")
        
        all_results = []
        total_combinations = len(dataset_dirs) * len(algorithms_to_test)
        current_combination = 0
        
        for dataset_dir in sorted(dataset_dirs, key=lambda x: x.name):  # Sort for consistent order
            dataset_name = dataset_dir.name
            current_combination += 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset {current_combination}/{len(dataset_dirs)}: {dataset_name}")
            logger.info(f"{'='*60}")
            
            dataset_results = []
            
            # Optimize each algorithm for this dataset
            for algo_idx, algorithm in enumerate(algorithms_to_test, 1):
                try:
                    logger.info(f"\n--- Algorithm {algo_idx}/{len(algorithms_to_test)}: {algorithm} ---")
                    progress = ((current_combination - 1) * len(algorithms_to_test) + algo_idx) / total_combinations * 100
                    logger.info(f"Overall progress: {progress:.1f}% ({algo_idx + (current_combination-1)*len(algorithms_to_test)}/{total_combinations})")
                    
                    result = self.optimize_algorithm(dataset_name, algorithm, n_trials_per_algorithm)
                    dataset_results.append(result)
                    all_results.append(result)
                    
                    # Save intermediate results after each algorithm
                    self.save_all_results(all_results)
                    logger.info(f"✓ Completed {algorithm} for {dataset_name} - Results saved")
                    
                except Exception as e:
                    logger.error(f"✗ Error optimizing {algorithm} for {dataset_name}: {str(e)}")
                    continue
            
            # Save results for this dataset
            if dataset_results:
                self.save_dataset_results(dataset_name, dataset_results)
                logger.info(f"✓ Completed all algorithms for {dataset_name}")
            
            # Log progress summary
            logger.info(f"\nDataset {dataset_name} completed ({current_combination}/{len(dataset_dirs)})")
            remaining_datasets = len(dataset_dirs) - current_combination
            if remaining_datasets > 0:
                logger.info(f"Remaining datasets: {remaining_datasets}")
        
        # Final save of all results
        self.save_all_results(all_results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"NAS-HPO COMPLETED! Total results: {len(all_results)}")
        logger.info(f"{'='*60}")
        
        return all_results
    
    def save_dataset_results(self, dataset_name: str, results: List[Dict[str, Any]]):
        """Save optimization results for a single dataset"""
        output_file = self.results_dir / f'{dataset_name}_optimization_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved results for {dataset_name}")
    
    def save_all_results(self, results: List[Dict[str, Any]]):
        """Save all optimization results"""
        output_file = self.results_dir / 'all_optimization_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save a summary
        summary = self.create_summary(results)
        summary_file = self.results_dir / 'optimization_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info("Saved all optimization results")
    
    def create_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create a summary of optimization results"""
        summary_lines = ["=== NAS-HPO Optimization Summary ===\n"]
        
        # Group by dataset
        datasets = {}
        for result in results:
            dataset = result['dataset']
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(result)
        
        for dataset, dataset_results in datasets.items():
            summary_lines.append(f"\nDataset: {dataset}")
            summary_lines.append("-" * 50)
            
            # Sort by score
            sorted_results = sorted(dataset_results, key=lambda x: x['best_score'], reverse=True)
            
            for i, result in enumerate(sorted_results):
                line = f"{i+1}. {result['algorithm']}: {result['best_score']:.4f}"
                
                # Add individual metrics if available
                metrics = result.get('best_metrics', {})
                if metrics.get('r2') is not None:
                    line += f" (R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f})"
                
                summary_lines.append(line)
            
            # Best algorithm for this dataset
            best_result = sorted_results[0]
            summary_lines.append(f"\nBest Algorithm: {best_result['algorithm']}")
            summary_lines.append(f"Best Composite Score: {best_result['best_score']:.4f}")
            
            # Add detailed metrics for best algorithm
            best_metrics = best_result.get('best_metrics', {})
            if best_metrics.get('r2') is not None:
                summary_lines.append(f"Best R²: {best_metrics['r2']:.4f}")
                summary_lines.append(f"Best MSE: {best_metrics['mse']:.4f}")
                summary_lines.append(f"Best MAE: {best_metrics['mae']:.4f}")
            
            summary_lines.append(f"Best Params: {json.dumps(best_result['best_params'], indent=2)}")
        
        return "\n".join(summary_lines)

def main():
    """Main execution function"""
    # Configuration - Optimized for high performance with focused algorithms
    DATA_DIR = "data_engineered_autofeat"  # Use the engineered data from feature_engineering.py
    RESULTS_DIR = "nas_hpo_results"
    TASK_TYPE = "regression"
    N_TRIALS_PER_ALGORITHM = 50  # Increased trials for better optimization
    
    # Initialize and run optimization
    optimizer = NASHPOOptimizer(DATA_DIR, RESULTS_DIR, TASK_TYPE)
    
    # Log configuration
    logger.info("=== NAS-HPO RUN CONFIGURATION ===")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Trials per algorithm: {N_TRIALS_PER_ALGORITHM}")
    logger.info(f"Core algorithms: {list(optimizer.core_algorithms.keys())}")
    
    # Check available datasets
    dataset_dirs = [d for d in optimizer.data_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
    logger.info(f"Found {len(dataset_dirs)} datasets: {[d.name for d in dataset_dirs]}")
    
    # Estimate runtime
    total_optimizations = len(dataset_dirs) * len(optimizer.core_algorithms)
    estimated_minutes = total_optimizations * N_TRIALS_PER_ALGORITHM * 0.5  # ~30 seconds per trial
    logger.info(f"Estimated runtime: {estimated_minutes/60:.1f} hours ({total_optimizations} total optimizations)")
    
    # Run full optimization with core algorithms
    optimizer.run_nas_hpo(N_TRIALS_PER_ALGORITHM, use_core_only=True)
    
    logger.info("=== NAS-HPO RUN COMPLETED! ===")
    print(f"Results saved in {RESULTS_DIR}")
    print("Ready for algorithm selection phase!")

if __name__ == "__main__":
    main()