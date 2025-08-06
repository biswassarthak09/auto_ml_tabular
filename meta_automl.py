import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings

import os
from pathlib import Path

# TensorFlow imports with fallback handling
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from scikeras.wrappers import KerasClassifier, KerasRegressor
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    Sequential = None
    Dense = None
    Dropout = None
    BatchNormalization = None
    Adam = None
    KerasClassifier = None
    KerasRegressor = None
    EarlyStopping = None
    print("Warning: TensorFlow not available. Neural network options will be disabled.")

warnings.filterwarnings('ignore')

def load_datasets_from_directory(data_dir="data_engineered"):
    """Load all datasets from the directory structure"""
    datasets = {}
    data_path = Path(data_dir)
    
    # Find all dataset folders
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            
            # Check if this is a dataset with numbered folders (like 1, 2, 3, etc.)
            fold_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            
            if fold_dirs:
                # Use the first fold for training data
                fold_1_path = dataset_dir / "1"
                if fold_1_path.exists():
                    try:
                        # Load the separate X and y files
                        X_train_path = fold_1_path / "X_train.parquet"
                        y_train_path = fold_1_path / "y_train.parquet"
                        X_test_path = fold_1_path / "X_test.parquet"
                        y_test_path = fold_1_path / "y_test.parquet"
                        
                        if all(path.exists() for path in [X_train_path, y_train_path, X_test_path, y_test_path]):
                            X_train = pd.read_parquet(X_train_path)
                            y_train = pd.read_parquet(y_train_path).iloc[:, 0]  # Assume single column
                            X_test = pd.read_parquet(X_test_path)
                            y_test = pd.read_parquet(y_test_path).iloc[:, 0]  # Assume single column
                            
                            datasets[dataset_name] = {
                                'train': (X_train, y_train),
                                'test': (X_test, y_test)
                            }
                            print(f"Loaded dataset: {dataset_name}")
                        else:
                            print(f"Missing files in {dataset_name}/1/")
                    except Exception as e:
                        print(f"Error loading {dataset_name}: {str(e)}")
            else:
                # Fallback to original logic for different structure
                train_path = dataset_dir / "train.parquet"
                test_path = dataset_dir / "test.parquet"
                
                if train_path.exists() and test_path.exists():
                    try:
                        train_df = pd.read_parquet(train_path)
                        test_df = pd.read_parquet(test_path)
                        
                        # Assume last column is target
                        X_train = train_df.iloc[:, :-1]
                        y_train = train_df.iloc[:, -1]
                        X_test = test_df.iloc[:, :-1]
                        y_test = test_df.iloc[:, -1]
                        
                        datasets[dataset_name] = {
                            'train': (X_train, y_train),
                            'test': (X_test, y_test)
                        }
                        print(f"Loaded dataset: {dataset_name}")
                    except Exception as e:
                        print(f"Error loading {dataset_name}: {str(e)}")
    
    return datasets

class MetaAutoMLOptuna:
    def __init__(self, task_type='classification', n_trials=100, cv_folds=5, 
                 random_state=42, enable_nas=True):
        """
        Initialize Meta-AutoML with Optuna for cross-dataset optimization
        
        Parameters:
        - task_type: 'classification' or 'regression'
        - n_trials: Number of optimization trials
        - cv_folds: Number of cross-validation folds
        - random_state: Random state for reproducibility
        - enable_nas: Whether to enable Neural Architecture Search (if TensorFlow available)
        """
        self.task_type = task_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.enable_nas = enable_nas if TF_AVAILABLE else False
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.study = None
        self.datasets = []
        self.scalers = []
        
        # Define models for each task type
        if task_type == 'classification':
            self.models = {
                'random_forest': RandomForestClassifier,
                'logistic_regression': LogisticRegression,
                'svm': SVC,
                'knn': KNeighborsClassifier,
                'decision_tree': DecisionTreeClassifier,
                'gradient_boosting': GradientBoostingClassifier
            }
            self.metric = 'accuracy'
        else:
            self.models = {
                'random_forest': RandomForestRegressor,
                'linear_regression': LinearRegression,
                'svr': SVR,
                'knn': KNeighborsRegressor,
                'decision_tree': DecisionTreeRegressor,
                'gradient_boosting': GradientBoostingRegressor
            }
            self.metric = 'r2'
        
        # Add MLP if NAS is enabled and TensorFlow is available
        if self.enable_nas:
            self.models['mlp'] = self.create_mlp_model
    
    def create_mlp_model(self, **params):
        """Create MLP model for NAS"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available. Cannot create MLP model.")
            
        model = Sequential()  # type: ignore
        
        # Input layer
        model.add(Dense(params['input_units'], activation=params['activation'],   # type: ignore
                       input_dim=params['input_dim']))
        
        if params.get('batch_norm', False):
            model.add(BatchNormalization())  # type: ignore
        
        # Hidden layers
        for units in params['hidden_units']:
            model.add(Dense(units, activation=params['activation']))  # type: ignore
            if params.get('dropout_rate', 0) > 0:
                model.add(Dropout(params['dropout_rate']))  # type: ignore
            if params.get('batch_norm', False):
                model.add(BatchNormalization())  # type: ignore
        
        # Output layer
        if self.task_type == 'classification':
            model.add(Dense(params['n_classes'], activation='softmax'))  # type: ignore
        else:
            model.add(Dense(1))  # type: ignore
        
        optimizer = Adam(learning_rate=params.get('learning_rate', params.get('mlp_learning_rate', 0.001)))  # type: ignore
        
        if self.task_type == 'classification':
            model.compile(optimizer=optimizer,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            return model
        else:
            model.compile(optimizer=optimizer,
                         loss='mean_squared_error',
                         metrics=['mse'])
            return model
    
    def get_hyperparameter_space(self, trial, model_name):
        """Define hyperparameter space for each model including NAS"""
    
        if model_name == 'mlp' and self.enable_nas:
            # Verify required attributes are set
            if 'input_dim' not in trial.user_attrs:
                raise ValueError("input_dim not set in trial user attributes")
            
            input_dim = trial.user_attrs['input_dim']
            n_classes = trial.user_attrs.get('n_classes', 1)
            
            return {
                'input_dim': input_dim,
                'input_units': trial.suggest_int('input_units', 64, 256),
                'hidden_units': [
                    trial.suggest_int(f'hidden_units_{i}', 32, 256) 
                    for i in range(trial.suggest_int('n_layers', 1, 3))
                ],
                'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'tanh']),
                'learning_rate': trial.suggest_float('mlp_learning_rate', 1e-4, 1e-2, log=True),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
                'n_classes': n_classes,
                'epochs': trial.suggest_int('epochs', 20, 100),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
            }
        
        if model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2']),
                'random_state': self.random_state
            }
        
        elif model_name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': self.random_state
            }
        
        elif model_name == 'linear_regression':
            return {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
        
        elif model_name in ['svm', 'svr']:
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 5) if trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']) == 'poly' else 3
            }
        
        elif model_name == 'knn':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'p': trial.suggest_int('p', 1, 3)
            }
        
        elif model_name == 'decision_tree':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('dt_max_features', ['sqrt', 'log2']),
                'random_state': self.random_state
            }
        
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': self.random_state
            }
    
    def preprocess_data(self, X, y):
        """Preprocess the data"""
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Fill missing values
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        if len(categorical_cols) > 0:
            X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encode target variable if classification
        if self.task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X_scaled, y, scaler
    
    def add_dataset(self, X, y, dataset_name=None):
        """Add a dataset to the meta-learning collection"""
        print(f"Adding dataset: {dataset_name or f'Dataset_{len(self.datasets)+1}'}")
        print(f"Shape: {X.shape}")
        
        # Preprocess data
        X_processed, y_processed, scaler = self.preprocess_data(X.copy(), y.copy())
        
        # Store preprocessed data and scaler
        self.datasets.append((X_processed, y_processed))
        self.scalers.append(scaler)
        
        print(f"Added successfully. Total datasets: {len(self.datasets)}")
        return self
    
    def objective(self, trial):
        """Objective function for meta-learning across datasets"""
    
        # Suggest model (now includes 'mlp' for NAS)
        model_name = trial.suggest_categorical('model', list(self.models.keys()))
        
        # For NAS, set input_dim before getting parameters (use max dimensions to handle varying sizes)
        if model_name == 'mlp' and self.enable_nas:
            if not self.datasets:
                raise ValueError("No datasets available for NAS")
            # Use the maximum feature dimension across all datasets
            max_features = max(X.shape[1] for X, y in self.datasets)
            trial.set_user_attr('input_dim', max_features)
            if self.task_type == 'classification':
                trial.set_user_attr('n_classes', len(np.unique(self.datasets[0][1])))
        
        # Get hyperparameters for the selected model
        params = self.get_hyperparameter_space(trial, model_name)
        
        # Ensure params is not None
        if params is None:
            raise ValueError(f"No hyperparameters defined for model {model_name}")
        
        # For NAS, we need to pass input dimensions (use max dimensions to handle varying sizes)
        if model_name == 'mlp' and self.enable_nas and len(self.datasets) > 0:
            max_features = max(X.shape[1] for X, y in self.datasets)
            trial.set_user_attr('input_dim', max_features)
            if self.task_type == 'classification':
                trial.set_user_attr('n_classes', len(np.unique(self.datasets[0][1])))
        
        # Evaluate on all datasets
        dataset_scores = []
        
        for i, (X_train, y_train) in enumerate(self.datasets):
            try:
                # Create model instance
                if model_name == 'mlp' and self.enable_nas:
                    # Special handling for neural networks
                    if not TF_AVAILABLE:
                        raise ValueError("TensorFlow not available for MLP model")
                    
                    if not all(k in params for k in ['input_dim', 'hidden_units']):
                        raise ValueError("Missing required MLP parameters")
                    
                    # Handle different input dimensions by padding/truncating features
                    expected_dim = params['input_dim']
                    current_dim = X_train.shape[1]
                    
                    if current_dim < expected_dim:
                        # Pad with zeros
                        padding = np.zeros((X_train.shape[0], expected_dim - current_dim))
                        X_train_padded = np.concatenate([X_train, padding], axis=1)
                    elif current_dim > expected_dim:
                        # Truncate features
                        X_train_padded = X_train[:, :expected_dim]
                    else:
                        X_train_padded = X_train
                    
                    # Map parameter names for MLP
                    mlp_params = params.copy()
                    if 'mlp_learning_rate' in mlp_params:
                        mlp_params['learning_rate'] = mlp_params.pop('mlp_learning_rate')
                        
                    model = self.models[model_name](**mlp_params)
                    
                    # Wrap for sklearn compatibility
                    if self.task_type == 'classification':
                        wrapped_model = KerasClassifier(build_fn=lambda: model,  # type: ignore
                                                      epochs=params['epochs'],
                                                      batch_size=params['batch_size'],
                                                      verbose=0)
                    else:
                        wrapped_model = KerasRegressor(build_fn=lambda: model,  # type: ignore
                                                     epochs=params['epochs'],
                                                     batch_size=params['batch_size'],
                                                     verbose=0)
                    
                    # Use smaller validation set for faster NAS
                    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                        X_train_padded, y_train, test_size=0.2, random_state=self.random_state)
                    
                    wrapped_model.fit(X_train_fit, y_train_fit,
                                    validation_data=(X_val, y_val),
                                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])  # type: ignore
                    
                    # Evaluate
                    if self.task_type == 'classification':
                        score = wrapped_model.score(X_val, y_val)
                    else:
                        
                        score = wrapped_model.score(X_val, y_val)
                else:
                    # Traditional ML models - handle parameter mapping
                    model_params = params.copy()
                    if model_name == 'gradient_boosting' and 'gb_learning_rate' in model_params:
                        model_params['learning_rate'] = model_params.pop('gb_learning_rate')
                    elif model_name == 'random_forest' and 'rf_max_features' in model_params:
                        model_params['max_features'] = model_params.pop('rf_max_features')
                    elif model_name == 'decision_tree' and 'dt_max_features' in model_params:
                        model_params['max_features'] = model_params.pop('dt_max_features')
                    
                    model = self.models[model_name](**model_params)
                    
                    # Perform cross-validation
                    scores = cross_val_score(model, X_train, y_train, 
                                           cv=self.cv_folds, scoring=self.metric)
                    score = scores.mean()
                
                dataset_scores.append(score)
                
            except Exception as e:
                print(f"Error with model {model_name} on dataset {i}: {str(e)}")
                dataset_scores.append(0.0 if self.task_type == 'classification' else -1.0)
        
        # Return the mean performance across all datasets
        mean_score = np.mean(dataset_scores)
        variance_penalty = np.std(dataset_scores) * 0.1
        
        return mean_score - variance_penalty
    
    def fit(self):
        """Fit the meta-AutoML model across all datasets"""
        if len(self.datasets) == 0:
            raise ValueError("No datasets added. Use add_dataset() to add datasets first.")
        
        print(f"\nStarting meta-learning optimization for {self.task_type} task...")
        print(f"Number of datasets: {len(self.datasets)}")
        print(f"Optimizing for performance across ALL datasets...")
        print(f"Models being searched: {list(self.models.keys())}")
        
        # Create study
        direction = 'maximize' if self.metric in ['accuracy', 'r2'] else 'minimize'
        self.study = optuna.create_study(direction=direction)
        
        # Optimize across all datasets
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Get best parameters and model
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if self.best_params is None:
            raise ValueError("Optimization failed to produce any results")
        
        # Train best model on all datasets combined
        model_name = self.best_params.get('model')
        if model_name is None:
            raise ValueError("No model type found in best parameters")
            
        model_params = {k: v for k, v in self.best_params.items() if k != 'model'}
        
        # Handle parameter name mapping for different models
        if model_name == 'mlp' and 'mlp_learning_rate' in model_params:
            model_params['learning_rate'] = model_params.pop('mlp_learning_rate')
        elif model_name == 'gradient_boosting' and 'gb_learning_rate' in model_params:
            model_params['learning_rate'] = model_params.pop('gb_learning_rate')
        elif model_name == 'random_forest' and 'rf_max_features' in model_params:
            model_params['max_features'] = model_params.pop('rf_max_features')
        elif model_name == 'decision_tree' and 'dt_max_features' in model_params:
            model_params['max_features'] = model_params.pop('dt_max_features')
        
        if model_name == 'mlp' and self.enable_nas:
            # For neural networks, we need to create the final architecture
            if not TF_AVAILABLE:
                raise ValueError("TensorFlow not available for MLP model")
            if len(self.datasets) == 0:
                raise ValueError("No datasets available for final training")

            # Use max dimensions for the final model
            max_features = max(X.shape[1] for X, y in self.datasets)
            model_params['input_dim'] = max_features
            if self.task_type == 'classification':
                model_params['n_classes'] = len(np.unique(np.concatenate([y for X, y in self.datasets])))

        self.best_model = self.models[model_name](**model_params)
            
        if self.best_params.get('model')== 'mlp':
            self.best_model.save_weights(f"best_mlp_model.h5")
        else:
            import joblib
            joblib.dump(self.best_model, f"best_{model_name}_model.pkl")
        
        print(f"\nMeta-learning optimization completed!")
        print(f"Best model: {model_name}")
        print(f"Best average score across datasets: {self.best_score:.4f}")
        print(f"Best parameters: {model_params}")
        
        return self
    
    def get_model_for_new_dataset(self, X_new, y_new=None):
        """Get a trained model for a new dataset using the best found configuration"""
        if self.best_model is None or self.best_params is None:
            raise ValueError("Meta-model not fitted yet or failed to fit. Call fit() first.")
        
        # Preprocess new data
        X_processed, y_processed, scaler = self.preprocess_data(
            X_new.copy(), 
            y_new.copy() if y_new is not None else None
        )
        
        # Create new model with best parameters
        model_name = self.best_params.get('model')
        if model_name is None:
            raise ValueError("No model type found in best parameters")
            
        model_params = {k: v for k, v in self.best_params.items() if k != 'model'}
        
        # Handle parameter name mapping for different models
        if model_name == 'mlp' and 'mlp_learning_rate' in model_params:
            model_params['learning_rate'] = model_params.pop('mlp_learning_rate')
        elif model_name == 'gradient_boosting' and 'gb_learning_rate' in model_params:
            model_params['learning_rate'] = model_params.pop('gb_learning_rate')
        elif model_name == 'random_forest' and 'rf_max_features' in model_params:
            model_params['max_features'] = model_params.pop('rf_max_features')
        elif model_name == 'decision_tree' and 'dt_max_features' in model_params:
            model_params['max_features'] = model_params.pop('dt_max_features')
        
        if model_name == 'mlp' and self.enable_nas:
            # Use max dimensions from training datasets
            max_features = max(X.shape[1] for X, y in self.datasets) if self.datasets else X_processed.shape[1]
            model_params['input_dim'] = max_features
            if self.task_type == 'classification' and y_processed is not None:
                model_params['n_classes'] = len(np.unique(y_processed))
            
            # Pad/truncate features to match expected dimensions
            current_dim = X_processed.shape[1]
            if current_dim < max_features:
                padding = np.zeros((X_processed.shape[0], max_features - current_dim))
                X_processed = np.concatenate([X_processed, padding], axis=1)
            elif current_dim > max_features:
                X_processed = X_processed[:, :max_features]
        
        new_model = self.models[model_name](**model_params)
        
        if y_new is not None:
            new_model.fit(X_processed, y_processed)
        
        return new_model, scaler
    
    def evaluate_on_datasets(self, test_datasets):
        """Evaluate the best model on test datasets"""
        if self.best_model is None or self.best_params is None:
            raise ValueError("Meta-model not fitted yet. Call fit() first.")
        
        results = {}
        
        for dataset_name, (X_test, y_test) in test_datasets.items():
            try:
                # Get model for this dataset
                model, scaler = self.get_model_for_new_dataset(X_test, y_test)
                
                # Preprocess test data the same way as training data
                X_test_processed, _, _ = self.preprocess_data(X_test.copy(), y_test.copy())
                
                # Make predictions
                predictions = model.predict(X_test_processed)
                
                # Calculate metrics
                if self.task_type == 'classification':
                    score = accuracy_score(y_test, predictions)
                    results[dataset_name] = {'accuracy': score}
                else:
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    results[dataset_name] = {'mse': mse, 'r2': r2}
            except Exception as e:
                print(f"Error evaluating on dataset {dataset_name}: {str(e)}")
                results[dataset_name] = {'error': str(e)}
        
        return results
    
    def get_trial_results(self):
        """Get detailed results from all trials"""
        if self.study is None:
            raise ValueError("No optimization performed yet.")
        
        trials_df = self.study.trials_dataframe()
        return trials_df.sort_values('value', ascending=False)
    
    def add_datasets_from_directory(self, data_dir="data"):
        """Add all datasets from directory to meta-learner"""
        datasets = load_datasets_from_directory(data_dir)
        
        for name, data in datasets.items():
            X_train, y_train = data['train']
            self.add_dataset(X_train, y_train, name)
        
        # Store test sets for final evaluation
        self.test_datasets = {
            name: data['test'] 
            for name, data in datasets.items()
        }
        return self
def train_and_test_meta_model_on_exam_dataset(meta_automl, data_dir):
    """
    Train a meta-automl model on the exam dataset.
    Args:
        meta_automl: Fitted MetaAutoMLOptuna instance
        data_dir: Directory containing exam_dataset files
    """
    X_train_path = data_dir / "X_train.parquet"
    y_train_path = data_dir / "y_train.parquet"
    X_test_path = data_dir / "X_test.parquet"

    if not (X_train_path.exists() and y_train_path.exists() and X_test_path.exists()):
        print("Missing exam_dataset files.")
        exit(1)

    X_train = pd.read_parquet(X_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0]
    X_test = pd.read_parquet(X_test_path)

    # Preprocessing: fit on train, apply to test
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # Fill missing values
    if len(numeric_cols) > 0:
        X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
        X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].mean())
    if len(categorical_cols) > 0:
        X_train[categorical_cols] = X_train[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
        X_test[categorical_cols] = X_test[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])

    # Encode categorical variables (fit on train, transform both)
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # Scale features (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    meta_automl.best_model.fit(X_train_scaled, y_train)
    print("Model trained on exam_dataset train set.")
    
    # R2 on train
    r2_train = meta_automl.best_model.score(X_train_scaled, y_train)
    print(f"R2 score on exam_dataset train: {r2_train:.4f}")

    # Predict on test (no test R2)
    y_pred = meta_automl.best_model.predict(X_test_scaled)
    import json
    preds_out = {"predictions": y_pred.tolist()}
    with open("predictions_exam_dataset.json", "w") as f:
        json.dump(preds_out, f)
    print("Predictions saved to predictions_exam_dataset.json")
    return y_pred, r2_train

def meta_learning_with_nas_example():
    """Example of using meta-learning with NAS"""
    from sklearn.datasets import make_classification
    
    # Initialize meta-AutoML with NAS enabled if available
    meta_automl = MetaAutoMLOptuna(
        task_type='regression',
        n_trials=100,
        cv_folds=10,
        random_state=42,
        enable_nas=TF_AVAILABLE
    )
    
    # Load and add all datasets
    meta_automl.add_datasets_from_directory("data_engineered")
    
    # Perform meta-learning
    meta_automl.fit()
    
    # Evaluate on test sets
    if hasattr(meta_automl, 'test_datasets'):
        test_results = meta_automl.evaluate_on_datasets(meta_automl.test_datasets)
        print("\nTest Results:")
        for name, metrics in test_results.items():
            print(f"\n{name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    # Show best model
    print("\nBest model architecture:")
    if (meta_automl.best_params and 
        meta_automl.best_params.get('model') == 'mlp' and 
        meta_automl.best_model is not None and
        hasattr(meta_automl.best_model, 'summary')):
        try:
            meta_automl.best_model.summary()
        except Exception as e:
            print(f"Could not display model summary: {str(e)}")
    else:
        print(meta_automl.best_model)

    # Test on exam-dataset if available
    exam_data_dir = Path(__file__).parent / "data_engineered" / "exam_dataset"/ "1"
    X_test_path = exam_data_dir / "X_test.parquet"
    if X_test_path.exists():
        preds, score = train_and_test_meta_model_on_exam_dataset(meta_automl, exam_data_dir)
        print("exam-dataset results:")
        print(f"Predictions: {preds[:10]}")
        print(f"Score on training: {score}")
    else:
        print("exam-dataset not found or missing files.")

    return meta_automl

if __name__ == "__main__":
    meta_learning_with_nas_example()