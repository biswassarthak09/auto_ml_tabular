"""
Advanced MLP-Based Meta-Learning AutoML Model
Uses neural networks for better meta-feature learning and algorithm/hyperparameter prediction
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)

# Real meta-features extractor
class MetaFeaturesExtractor:
    def extract_all_datasets_meta_features(self, datasets, original_dir, fe_dir):
        """Extract real meta-features from datasets"""
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        rows = []
        for dataset in datasets:
            try:
                # Read first fold of original data to get meta-features
                dataset_path = Path(original_dir) / dataset / "1"
                X_train = pd.read_parquet(dataset_path / "X_train.parquet")
                y_train = pd.read_parquet(dataset_path / "y_train.parquet")
                
                # Read engineered data for comparison
                fe_dataset_path = Path(fe_dir) / dataset / "1"
                if (fe_dataset_path / "X_train_engineered.parquet").exists():
                    X_fe = pd.read_parquet(fe_dataset_path / "X_train_engineered.parquet")
                else:
                    X_fe = X_train  # Fallback to original if engineered doesn't exist
                
                # Extract basic meta-features
                n_samples, n_features = X_train.shape
                n_numerical = X_train.select_dtypes(include=[np.number]).shape[1]
                n_categorical = n_features - n_numerical
                missing_ratio = X_train.isnull().sum().sum() / (n_samples * n_features)
                
                # Statistical features  
                target_values = y_train.iloc[:, 0]
                target_variance = 1.0  # Simple default for now
                
                # Correlation features
                numeric_data = X_train.select_dtypes(include=[np.number])
                if numeric_data.shape[1] > 1:
                    corr_matrix = numeric_data.corr()
                    correlation_mean = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
                    correlation_std = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].std())
                else:
                    correlation_mean = 0.0
                    correlation_std = 0.0
                
                # Distribution features
                skewness_mean = float(numeric_data.skew().mean()) if numeric_data.shape[1] > 0 else 0.0
                kurtosis_mean = float(numeric_data.kurtosis().mean()) if numeric_data.shape[1] > 0 else 0.0
                
                # Feature engineering impact
                fe_impact_score = (X_fe.shape[1] - X_train.shape[1]) / X_train.shape[1]
                
                # Information theoretic features
                feature_entropy = 0.0
                for col in numeric_data.columns:
                    # Simple entropy approximation using histogram
                    hist, _ = np.histogram(numeric_data[col].dropna(), bins=10)
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        prob = hist / hist.sum()
                        feature_entropy += -np.sum(prob * np.log2(prob))
                feature_entropy = feature_entropy / max(1, numeric_data.shape[1])
                
                row = {
                    'dataset': dataset,
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'n_categorical': n_categorical,
                    'n_numerical': n_numerical,
                    'missing_ratio': missing_ratio,
                    'feature_entropy': feature_entropy,
                    'target_variance': target_variance,
                    'correlation_mean': correlation_mean,
                    'correlation_std': correlation_std,
                    'skewness_mean': skewness_mean,
                    'kurtosis_mean': kurtosis_mean,
                    'fe_impact_score': fe_impact_score
                }
                rows.append(row)
                
            except Exception as e:
                logger.warning(f"Error extracting meta-features for {dataset}: {str(e)}")
                # Fallback to dummy features
                row = {
                    'dataset': dataset,
                    'n_samples': 1000,
                    'n_features': 10,
                    'n_categorical': 2,
                    'n_numerical': 8,
                    'missing_ratio': 0.05,
                    'feature_entropy': 2.0,
                    'target_variance': 1.0,
                    'correlation_mean': 0.2,
                    'correlation_std': 0.15,
                    'skewness_mean': 0.0,
                    'kurtosis_mean': 0.0,
                    'fe_impact_score': 0.2
                }
                rows.append(row)
        
        return pd.DataFrame(rows)

logger = logging.getLogger(__name__)

class MetaLearningDataset(Dataset):
    """PyTorch Dataset for meta-learning"""
    
    def __init__(self, meta_features: np.ndarray, algorithm_targets: np.ndarray, 
                 hp_targets: np.ndarray, hp_mask: np.ndarray):
        self.meta_features = torch.FloatTensor(meta_features)
        self.algorithm_targets = torch.LongTensor(algorithm_targets)
        self.hp_targets = torch.FloatTensor(hp_targets)
        self.hp_mask = torch.BoolTensor(hp_mask)  # Mask for missing hyperparameters
    
    def __len__(self):
        return len(self.meta_features)
    
    def __getitem__(self, idx):
        return (
            self.meta_features[idx],
            self.algorithm_targets[idx],
            self.hp_targets[idx],
            self.hp_mask[idx]
        )

class MetaLearningMLP(nn.Module):
    """
    Multi-output MLP for meta-learning AutoML
    Predicts both algorithm choice and hyperparameters simultaneously
    """
    
    def __init__(self, n_meta_features: int, n_algorithms: int, n_hyperparams: int,
                 hidden_dims: List[int] = [256, 128, 64]):
        super(MetaLearningMLP, self).__init__()
        
        self.n_meta_features = n_meta_features
        self.n_algorithms = n_algorithms
        self.n_hyperparams = n_hyperparams
        
        # Shared feature encoder
        layers = []
        prev_dim = n_meta_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_encoder = nn.Sequential(*layers)
        
        # Algorithm classifier head
        self.algorithm_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_algorithms)
        )
        
        # Hyperparameter regression head
        self.hyperparameter_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_hyperparams)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, meta_features):
        """Forward pass"""
        # Shared feature encoding
        encoded_features = self.feature_encoder(meta_features)
        
        # Algorithm prediction
        algorithm_logits = self.algorithm_head(encoded_features)
        
        # Hyperparameter prediction
        hyperparameter_values = self.hyperparameter_head(encoded_features)
        
        return algorithm_logits, hyperparameter_values

class AdvancedMetaLearningAutoML:
    """
    Advanced Meta-Learning AutoML using MLP with multi-task learning
    """
    
    def __init__(self, output_dir: str, device: str = 'auto'):
        """Initialize Advanced Meta-Learning AutoML"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Components
        self.meta_extractor = MetaFeaturesExtractor()
        self.feature_scaler = StandardScaler()
        self.algorithm_encoder = LabelEncoder()
        self.hp_scalers = {}  # Individual scalers for each hyperparameter
        
        # Model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training data storage
        self.meta_features_df = None
        self.algorithm_performance_df = None
        self.hyperparameter_mappings = {}
        
        # Model state
        self.is_trained = False
        self.algorithms_list = []
        self.hyperparameter_names = []
        
        # Training configuration
        self.training_config = {
            'batch_size': 16,
            'learning_rate': 0.001,
            'n_epochs': 200,
            'early_stopping_patience': 20,
            'algorithm_loss_weight': 1.0,
            'hyperparameter_loss_weight': 1.0
        }
    
    def prepare_training_data(self, nas_hpo_results_dir: str, 
                            original_data_dir: str, fe_data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for MLP training using fold-level data"""
        logger.info("Preparing fold-level training data for advanced meta-learning...")
        
        # Load NAS-HPO results
        nas_results = self._load_nas_hpo_results(nas_hpo_results_dir)
        
        # Get unique datasets
        datasets = list(set(result['dataset'] for result in nas_results))
        logger.info(f"Found datasets: {datasets}")
        
        # Extract meta-features for all datasets (one per dataset)
        self.meta_features_df = self.meta_extractor.extract_all_datasets_meta_features(
            datasets, original_data_dir, fe_data_dir
        )
        
        # Prepare fold-level targets
        algorithm_targets_df = self._prepare_algorithm_targets(nas_results)
        hyperparameter_targets_df = self._prepare_hyperparameter_targets(nas_results)
        
        # Create fold-level meta-features by replicating dataset meta-features for each fold
        fold_meta_features = []
        for _, algo_row in algorithm_targets_df.iterrows():
            dataset = algo_row['dataset']
            fold = algo_row['fold']
            
            # Find corresponding meta-features for this dataset
            dataset_meta = self.meta_features_df[self.meta_features_df['dataset'] == dataset]
            if not dataset_meta.empty:
                meta_row = dataset_meta.iloc[0].copy()
                meta_row['fold'] = fold  # Add fold column
                meta_row['dataset_fold'] = f"{dataset}_fold_{fold}"
                fold_meta_features.append(meta_row)
        
        fold_meta_features_df = pd.DataFrame(fold_meta_features)
        
        # Merge with fold-level targets
        training_data = fold_meta_features_df.merge(algorithm_targets_df, on=['dataset', 'fold'], how='inner')
        training_data = training_data.merge(hyperparameter_targets_df, on=['dataset', 'fold'], how='inner')
        
        # Prepare arrays for training
        feature_cols = [col for col in self.meta_features_df.columns if col != 'dataset']
        X = training_data[feature_cols].fillna(0).values
        
        # Algorithm targets
        y_algorithm = training_data['best_algorithm'].values
        self.algorithms_list = list(set(y_algorithm))
        y_algorithm_encoded = self.algorithm_encoder.fit_transform(y_algorithm.astype(str))
        
        # Hyperparameter targets
        hp_cols = [col for col in training_data.columns if col.startswith('hp_')]
        self.hyperparameter_names = hp_cols
        
        y_hyperparams = training_data[hp_cols].values
        hp_mask = ~np.isnan(y_hyperparams)  # Mask for missing values
        
        # Scale features and hyperparameters
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Scale hyperparameters individually
        y_hyperparams_scaled = y_hyperparams.copy()
        for i, hp_name in enumerate(hp_cols):
            valid_mask = ~np.isnan(y_hyperparams[:, i])
            if valid_mask.sum() > 1:
                scaler = StandardScaler()
                y_hyperparams_scaled[valid_mask, i] = scaler.fit_transform(
                    y_hyperparams[valid_mask, i].reshape(-1, 1)
                ).flatten()
                self.hp_scalers[hp_name] = scaler
        
        # Fill NaN with 0 after scaling
        y_hyperparams_scaled = np.nan_to_num(y_hyperparams_scaled, nan=0)
        
        logger.info(f"Fold-level training data prepared:")
        logger.info(f"  Meta-features shape: {X_scaled.shape}")
        logger.info(f"  Algorithm targets shape: {np.array(y_algorithm_encoded).shape}")
        logger.info(f"  Hyperparameter targets shape: {y_hyperparams_scaled.shape}")
        logger.info(f"  Algorithms: {self.algorithms_list}")
        logger.info(f"  Hyperparameters: {len(hp_cols)}")
        logger.info(f"  Total training examples: {len(X_scaled)} (fold-level)")
        
        return X_scaled, np.array(y_algorithm_encoded), y_hyperparams_scaled, hp_mask
    
    def _load_nas_hpo_results(self, results_dir: str) -> List[Dict[str, Any]]:
        """Load all NAS-HPO results from files"""
        results_path = Path(results_dir)
        results = []
        
        # Load main results
        main_file = results_path / 'all_optimization_results.json'
        if main_file.exists():
            with open(main_file, 'r') as f:
                results.extend(json.load(f))
        
        # Load quick test results
        for quick_file in results_path.glob('*_quick_test_optimization_results.json'):
            try:
                with open(quick_file, 'r') as f:
                    results.extend(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading {quick_file}: {str(e)}")
        
        logger.info(f"Loaded {len(results)} NAS-HPO results")
        return results
    
    def _prepare_algorithm_targets(self, nas_results: List[Dict]) -> pd.DataFrame:
        """Prepare algorithm selection targets using fold-level data"""
        rows = []
        
        for result in nas_results:
            dataset = result['dataset']
            algorithm = result['algorithm']
            best_params = result['best_params']
            fold_details = result.get('best_metrics', {}).get('fold_details', [])
            
            # Extract fold-level performance
            if fold_details:
                for fold_idx, fold_metrics in enumerate(fold_details, 1):
                    composite_score = fold_metrics.get('composite_score', fold_metrics.get('r2', 0))
                    
                    rows.append({
                        'dataset': dataset,
                        'fold': fold_idx,
                        'algorithm': algorithm,
                        'score': composite_score,
                        'params': best_params
                    })
            else:
                # Fallback to aggregate score if fold details not available
                rows.append({
                    'dataset': dataset,
                    'fold': 1,  # Default fold
                    'algorithm': algorithm,
                    'score': result['best_score'],
                    'params': best_params
                })
        
        # For each dataset-fold combination, find the best algorithm
        fold_level_df = pd.DataFrame(rows)
        
        if fold_level_df.empty:
            return pd.DataFrame(columns=['dataset', 'fold', 'best_algorithm', 'best_score'])
        
        # Group by dataset and fold, find best algorithm for each
        best_per_fold = fold_level_df.loc[fold_level_df.groupby(['dataset', 'fold'])['score'].idxmax()]
        
        # Create result dataframe manually to avoid pandas version issues
        result_data = []
        for _, row in best_per_fold.iterrows():
            result_data.append({
                'dataset': row['dataset'],
                'fold': row['fold'],
                'best_algorithm': row['algorithm'],
                'best_score': row['score']
            })
        
        result_df = pd.DataFrame(result_data)
        
        # Create unique identifier for dataset-fold combinations
        result_df['dataset_fold'] = result_df['dataset'] + '_fold_' + result_df['fold'].astype(str)
        
        return result_df
    
    def _prepare_hyperparameter_targets(self, nas_results: List[Dict]) -> pd.DataFrame:
        """Prepare hyperparameter targets using fold-level data"""
        rows = []
        
        for result in nas_results:
            dataset = result['dataset']
            algorithm = result['algorithm']
            best_params = result['best_params']
            fold_details = result.get('best_metrics', {}).get('fold_details', [])
            
            # Extract fold-level performance
            if fold_details:
                for fold_idx, fold_metrics in enumerate(fold_details, 1):
                    composite_score = fold_metrics.get('composite_score', fold_metrics.get('r2', 0))
                    
                    rows.append({
                        'dataset': dataset,
                        'fold': fold_idx,
                        'algorithm': algorithm,
                        'score': composite_score,
                        'params': best_params
                    })
            else:
                # Fallback to aggregate score if fold details not available
                rows.append({
                    'dataset': dataset,
                    'fold': 1,  # Default fold
                    'algorithm': algorithm,
                    'score': result['best_score'],
                    'params': best_params
                })
        
        # For each dataset-fold combination, find the best algorithm and its params
        fold_level_df = pd.DataFrame(rows)
        
        if fold_level_df.empty:
            return pd.DataFrame()
        
        # Group by dataset and fold, find best algorithm for each
        best_per_fold = fold_level_df.loc[fold_level_df.groupby(['dataset', 'fold'])['score'].idxmax()]
        
        # Collect all hyperparameter names
        all_param_names = set()
        for _, row in best_per_fold.iterrows():
            all_param_names.update(row['params'].keys())
        
        # Prepare hyperparameter targets
        hp_rows = []
        for _, row in best_per_fold.iterrows():
            hp_row = {
                'dataset': row['dataset'],
                'fold': row['fold'],
                'dataset_fold': f"{row['dataset']}_fold_{row['fold']}"
            }
            
            for param_name in all_param_names:
                if param_name in row['params']:
                    param_value = row['params'][param_name]
                    if isinstance(param_value, (int, float)):
                        hp_row[f'hp_{param_name}'] = float(param_value)
                    else:
                        # Handle categorical parameters
                        if param_name not in self.hyperparameter_mappings:
                            self.hyperparameter_mappings[param_name] = LabelEncoder()
                        
                        # For now, assign a simple encoding
                        hp_row[f'hp_{param_name}'] = hash(str(param_value)) % 100  # Simple hash encoding
                else:
                    hp_row[f'hp_{param_name}'] = np.nan
            
            hp_rows.append(hp_row)
        
        return pd.DataFrame(hp_rows)
    
    def train_meta_model(self, X: np.ndarray, y_algorithm: np.ndarray, 
                        y_hyperparams: np.ndarray, hp_mask: np.ndarray):
        """Train the MLP meta-learning model"""
        logger.info("Training advanced MLP meta-learning model...")
        
        # Initialize model
        n_meta_features = X.shape[1]
        n_algorithms = len(self.algorithms_list)
        n_hyperparams = y_hyperparams.shape[1]
        
        # Store dimensions for later use
        self.meta_features_dim = n_meta_features
        self.hp_dim = n_hyperparams
        self.hidden_dim = 256  # Default hidden dimension
        
        self.model = MetaLearningMLP(
            n_meta_features=n_meta_features,
            n_algorithms=n_algorithms,
            n_hyperparams=n_hyperparams
        ).to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.training_config['learning_rate'],
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Create dataset and dataloader
        dataset = MetaLearningDataset(X, y_algorithm, y_hyperparams, hp_mask)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=True
        )
        
        # Loss functions
        algorithm_criterion = nn.CrossEntropyLoss()
        hyperparameter_criterion = nn.MSELoss(reduction='none')  # No reduction for masking
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.training_config['n_epochs']):
            self.model.train()
            total_loss = 0
            total_algo_loss = 0
            total_hp_loss = 0
            
            for batch_meta, batch_algo, batch_hp, batch_mask in dataloader:
                batch_meta = batch_meta.to(self.device)
                batch_algo = batch_algo.to(self.device)
                batch_hp = batch_hp.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                algo_logits, hp_pred = self.model(batch_meta)
                
                # Algorithm loss
                algo_loss = algorithm_criterion(algo_logits, batch_algo)
                
                # Hyperparameter loss (masked)
                hp_loss = hyperparameter_criterion(hp_pred, batch_hp)
                hp_loss = (hp_loss * batch_mask.float()).sum() / batch_mask.float().sum()
                
                # Combined loss
                total_batch_loss = (
                    self.training_config['algorithm_loss_weight'] * algo_loss + 
                    self.training_config['hyperparameter_loss_weight'] * hp_loss
                )
                
                total_batch_loss.backward()
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                total_algo_loss += algo_loss.item()
                total_hp_loss += hp_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_algo_loss = total_algo_loss / len(dataloader)
            avg_hp_loss = total_hp_loss / len(dataloader)
            
            # Learning rate scheduling
            self.scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model with proper path handling
                best_model_path = str((self.output_dir / 'best_model.pth').absolute())
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Algo={avg_algo_loss:.4f}, HP={avg_hp_loss:.4f}")
            
            if patience_counter >= self.training_config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(self.output_dir / 'best_model.pth'))
        self.is_trained = True
        
        logger.info("MLP meta-learning model training completed!")
    
    def predict_for_new_dataset(self, dataset_name: str, data_dir: str = "data_engineered_autofeat", 
                               original_data_dir: str = "data") -> Optional[Dict[str, Any]]:
        """
        Predict best algorithm and hyperparameters for a new/unknown dataset
        
        Args:
            dataset_name: Name of the dataset to predict for
            data_dir: Directory containing engineered features
            original_data_dir: Directory containing original data
            
        Returns:
            Dictionary with predicted algorithm, hyperparameters, and confidence
        """
        if not self.is_trained or self.model is None:
            logger.error("Meta-learning model is not trained yet!")
            return None
            
        try:
            # Extract meta-features for the new dataset
            logger.info(f"Extracting meta-features for {dataset_name}")
            
            # Use fold 1 as representative
            dataset_path = Path(original_data_dir) / dataset_name / "1"
            if not dataset_path.exists():
                logger.error(f"Dataset path not found: {dataset_path}")
                return None
                
            X_train = pd.read_parquet(dataset_path / "X_train.parquet")
            y_train = pd.read_parquet(dataset_path / "y_train.parquet")
            
            # Get engineered features for comparison
            fe_dataset_path = Path(data_dir) / dataset_name / "1"
            if (fe_dataset_path / "X_train_engineered.parquet").exists():
                X_fe = pd.read_parquet(fe_dataset_path / "X_train_engineered.parquet")
            else:
                X_fe = X_train  # Fallback
                
            # Extract meta-features (same as in training)
            n_samples, n_features = X_train.shape
            n_numerical = X_train.select_dtypes(include=[np.number]).shape[1]
            n_categorical = n_features - n_numerical
            missing_ratio = X_train.isnull().sum().sum() / (n_samples * n_features)
            
            target_values = y_train.iloc[:, 0]
            try:
                if len(target_values) > 1:
                    numeric_array = np.array(target_values, dtype=np.float64)
                    target_variance = float(np.var(numeric_array))
                else:
                    target_variance = 1.0
            except:
                target_variance = 1.0
            
            # Correlation features
            numeric_data = X_train.select_dtypes(include=[np.number])
            if numeric_data.shape[1] > 1:
                corr_matrix = numeric_data.corr()
                correlation_mean = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
                correlation_std = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].std())
            else:
                correlation_mean = 0.0
                correlation_std = 0.0
                
            # Distribution features
            skewness_mean = float(numeric_data.skew().mean()) if numeric_data.shape[1] > 0 else 0.0
            kurtosis_mean = float(numeric_data.kurtosis().mean()) if numeric_data.shape[1] > 0 else 0.0
            
            # Feature engineering impact
            fe_impact_score = (X_fe.shape[1] - X_train.shape[1]) / X_train.shape[1]
            
            # Information theoretic features
            feature_entropy = 0.0
            for col in numeric_data.columns:
                hist, _ = np.histogram(numeric_data[col].dropna(), bins=10)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    prob = hist / hist.sum()
                    feature_entropy += -np.sum(prob * np.log2(prob))
            feature_entropy = feature_entropy / max(1, numeric_data.shape[1])
            
            # Create meta-features vector
            meta_features = np.array([
                n_samples, n_features, n_categorical, n_numerical, missing_ratio,
                feature_entropy, target_variance, correlation_mean, correlation_std,
                skewness_mean, kurtosis_mean, fe_impact_score
            ]).reshape(1, -1)
            
            # Scale meta-features
            meta_features_scaled = self.feature_scaler.transform(meta_features)
            
            # Convert to tensor
            meta_tensor = torch.FloatTensor(meta_features_scaled).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                algo_logits, hp_pred = self.model(meta_tensor)
                
                # Get algorithm prediction
                algo_probs = torch.softmax(algo_logits, dim=1)
                predicted_algo_idx = int(torch.argmax(algo_logits, dim=1).item())
                confidence = float(algo_probs[0, predicted_algo_idx].item())
                
                predicted_algorithm = self.algorithms_list[predicted_algo_idx]
                
                # Get hyperparameter predictions
                hp_pred_np = hp_pred.cpu().numpy().flatten()
                
                # Map back to hyperparameter values
                predicted_hyperparams = {}
                hp_idx = 0
                
                # Common hyperparameters based on the algorithm
                if predicted_algorithm == 'random_forest':
                    predicted_hyperparams = {
                        'n_estimators': max(10, int(hp_pred_np[hp_idx] * 500)),
                        'max_depth': max(3, int(hp_pred_np[hp_idx + 1] * 20)) if hp_idx + 1 < len(hp_pred_np) else 10,
                        'min_samples_split': max(2, int(hp_pred_np[hp_idx + 2] * 20)) if hp_idx + 2 < len(hp_pred_np) else 2,
                        'random_state': 42
                    }
                elif predicted_algorithm == 'gradient_boosting':
                    predicted_hyperparams = {
                        'n_estimators': max(10, int(hp_pred_np[hp_idx] * 500)),
                        'learning_rate': max(0.01, hp_pred_np[hp_idx + 1]) if hp_idx + 1 < len(hp_pred_np) else 0.1,
                        'max_depth': max(3, int(hp_pred_np[hp_idx + 2] * 10)) if hp_idx + 2 < len(hp_pred_np) else 6,
                        'random_state': 42
                    }
                elif predicted_algorithm == 'mlp':
                    hidden_size = max(50, int(hp_pred_np[hp_idx] * 500))
                    predicted_hyperparams = {
                        'hidden_layer_sizes': [hidden_size, hidden_size],
                        'activation': 'relu',
                        'alpha': max(0.0001, hp_pred_np[hp_idx + 1]) if hp_idx + 1 < len(hp_pred_np) else 0.001,
                        'learning_rate': 'adaptive',
                        'max_iter': 1000,
                        'random_state': 42
                    }
                else:
                    # Default hyperparameters for other algorithms
                    predicted_hyperparams = {'random_state': 42}
                
                result = {
                    'algorithm': predicted_algorithm,
                    'hyperparams': predicted_hyperparams,
                    'confidence': confidence,
                    'meta_features': meta_features.flatten().tolist()
                }
                
                logger.info(f"Meta-learning prediction for {dataset_name}:")
                logger.info(f"  Algorithm: {predicted_algorithm} (confidence: {confidence:.3f})")
                logger.info(f"  Hyperparameters: {predicted_hyperparams}")
                
                return result
                
        except Exception as e:
            logger.error(f"Error making prediction for {dataset_name}: {str(e)}")
            return None
    
    def save_model(self, save_path: str = "meta_learning_model"):
        """Save the trained meta-learning model"""
        try:
            if not self.is_trained or self.model is None:
                logger.error("No trained model to save!")
                return False
                
            # Simple cross-platform path handling - os.path.join handles everything!
            import os
            os.makedirs(save_path, exist_ok=True)
            
            # Save model
            save_file_path = os.path.join(save_path, "meta_model.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'feature_scaler': self.feature_scaler,
                'algorithms_list': self.algorithms_list,
                'model_config': {
                    'meta_features_dim': self.meta_features_dim,
                    'n_algorithms': len(self.algorithms_list),
                    'hidden_dim': self.hidden_dim,
                    'hp_dim': self.hp_dim
                }
            }, save_file_path)
            
            logger.info(f"Meta-learning model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, load_path: str = "meta_learning_model"):
        """Load a trained meta-learning model"""
        try:
            # Simple cross-platform path handling - os.path.join handles everything!
            import os
            model_file_path = os.path.join(load_path, "meta_model.pth")
            if not os.path.exists(model_file_path):
                logger.error(f"Model file not found: {model_file_path}")
                return False
                
            # Load model
            checkpoint = torch.load(model_file_path, map_location=self.device, weights_only=False)
            
            # Restore configuration
            config = checkpoint['model_config']
            self.meta_features_dim = config['meta_features_dim']
            self.hidden_dim = config['hidden_dim']
            self.hp_dim = config['hp_dim']
            self.algorithms_list = checkpoint['algorithms_list']
            
            # Recreate and load model
            self.model = MetaLearningMLP(
                n_meta_features=self.meta_features_dim,
                n_algorithms=len(self.algorithms_list),
                n_hyperparams=self.hp_dim
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.feature_scaler = checkpoint['feature_scaler']
            self.is_trained = True
            
            logger.info(f"Meta-learning model loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_trained_model(self):
        """Load a previously trained model"""
        model_path = self.output_dir / 'best_model.pth'
        if not model_path.exists():
            logger.error(f"No trained model found at {model_path}")
            return False
            
        try:
            # Load model configuration and components
            scaler_path = self.output_dir / 'feature_scaler.joblib'
            encoder_path = self.output_dir / 'algorithm_encoder.joblib'
            config_path = self.output_dir / 'model_config.json'
            
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
            if encoder_path.exists():
                self.algorithm_encoder = joblib.load(encoder_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.algorithms_list = config.get('algorithms_list', [])
                    
            # Initialize model with same architecture
            if hasattr(self, 'meta_features_df') and self.meta_features_df is not None:
                n_meta_features = len(self.meta_features_df.columns) - 1  # Exclude dataset column
            else:
                n_meta_features = 12  # Default based on our meta-features
                
            n_algorithms = len(self.algorithms_list) if self.algorithms_list else 7
            n_hyperparams = 10  # Approximate
            
            self.model = MetaLearningMLP(
                n_meta_features=n_meta_features,
                n_algorithms=n_algorithms,
                n_hyperparams=n_hyperparams
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_trained = True
            
            logger.info("Trained meta-learning model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}")
            return False

def main():
    """Test the advanced meta-learning system"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration - Use absolute paths
    import os
    base_dir = "C:\Users\ahker\Desktop\University\auto_ml\auto_ml_tabular"
    NAS_HPO_RESULTS_DIR = os.path.join(base_dir, "nas_hpo_results")
    ORIGINAL_DATA_DIR = os.path.join(base_dir, "data")
    FE_DATA_DIR = os.path.join(base_dir, "data_engineered_autofeat")
    OUTPUT_DIR = os.path.join(base_dir, "meta_learning_models")
    
    # Initialize meta-learning system
    meta_automl = AdvancedMetaLearningAutoML(OUTPUT_DIR)
    
    # Test data preparation
    try:
        X, y_algo, y_hp, hp_mask = meta_automl.prepare_training_data(
            NAS_HPO_RESULTS_DIR, ORIGINAL_DATA_DIR, FE_DATA_DIR
        )
        print("✅ Advanced meta-learning data preparation successful!")
        print(f"Features shape: {X.shape}")
        print(f"Algorithm targets shape: {y_algo.shape}")
        print(f"Hyperparameter targets shape: {y_hp.shape}")
        print(f"Algorithms found: {meta_automl.algorithms_list}")
        print(f"Hyperparameters: {len(meta_automl.hyperparameter_names)}")
        
        # Train model
        meta_automl.train_meta_model(X, y_algo, y_hp, hp_mask)
        print("✅ MLP training completed!")
        
        # Save the trained model
        save_path = os.path.join(base_dir, "meta_learning_model")
        if meta_automl.save_model(save_path):
            print(f"✅ Meta-learning model saved to {save_path}")
        else:
            print("❌ Failed to save meta-learning model")
        
        # Test predictions on all samples
        if meta_automl.model is not None:
            import torch
            meta_automl.model.eval()
            with torch.no_grad():
                test_input = torch.FloatTensor(X).to(meta_automl.device)
                algo_logits, hp_pred = meta_automl.model(test_input)
                
                predicted_algos = torch.argmax(algo_logits, dim=1).cpu().numpy()
                algo_probs = torch.softmax(algo_logits, dim=1).cpu().numpy()
                
                print("\n🎯 Prediction Results for Each Dataset:")
                datasets = ["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1"]
                for i, dataset in enumerate(datasets):
                    if i < len(predicted_algos):
                        predicted_algo = meta_automl.algorithms_list[predicted_algos[i]]
                        confidence = algo_probs[i].max()
                        print(f"  {dataset}: {predicted_algo} (confidence: {confidence:.3f})")
                
                # Overall accuracy
                accuracy = (predicted_algos == y_algo).mean()
                print(f"\n📈 Overall Algorithm Accuracy: {accuracy:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
