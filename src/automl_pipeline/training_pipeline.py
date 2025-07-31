"""
AutoML Training Pipeline
Complete training pipeline for tabular regression tasks

Components:
1. Feature Engineering (Conservative AutoFeat)
2. NAS-HPO (Neural Architecture Search + Hyperparameter Optimization)
3. Meta-Learning (Train meta-model for algorithm selection)
4. Final Model Training & Evaluation

Usage:
    python training_pipeline.py
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
def run_training_pipeline():
    """Execute the complete AutoML training pipeline"""
    
    start_time = time.time()
    
    print("🚀 AUTOML TRAINING PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    config = {
        'data_dir': str(ROOT_DIR / 'data'),
        'engineered_data_dir': str(ROOT_DIR / 'data_engineered_autofeat'),
        'nas_hpo_output_dir': 'nas_hpo_results',
        'meta_learning_output_dir': str(ROOT_DIR / 'nas_hpo_results'),
        'final_models_output_dir': str(ROOT_DIR / 'meta_learning_model'),
        'datasets': ['bike_sharing_demand', 'brazilian_houses', 'superconductivity', 'wine_quality', 'yprop_4_1']
    }
    
    results = {}
    
    try:
        # ==========================================
        # STEP 1: FEATURE ENGINEERING
        # ==========================================
        print("📊 STEP 1: FEATURE ENGINEERING")
        print("-" * 40)
        
        from feature_engineering import FeatureEngineer
        
        logger.info("Starting feature engineering with conservative AutoFeat...")
        
        fe = FeatureEngineer(
            data_dir=config['data_dir'],
            output_dir=config['engineered_data_dir'],
            task_type='regression'
        )
        
        fe.process_all_datasets()
        
        print("✅ Feature engineering completed")
        results['feature_engineering'] = 'SUCCESS'
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        results['feature_engineering'] = f'FAILED: {str(e)}'
        print("Feature engineering failed - continuing with existing data...")
    
    try:
        # ==========================================
        # STEP 2: NAS-HPO
        # ==========================================
        print("\\n STEP 2: NAS-HPO OPTIMIZATION")
        print("-" * 40)
        
        from nas_hpo_optuna import main as nas_hpo_main
        
        logger.info("Starting NAS-HPO optimization...")
        
        # Run NAS-HPO (this will use the engineered data)
        nas_hpo_main()
        
        print("✅ NAS-HPO optimization completed")
        results['nas_hpo'] = 'SUCCESS'
        
    except Exception as e:
        logger.error(f"NAS-HPO failed: {str(e)}")
        results['nas_hpo'] = f'FAILED: {str(e)}'
        print(" NAS-HPO failed - will use default algorithms...")
    
    try:
        # ==========================================
        # STEP 3: META-LEARNING
        # ==========================================
        print("\\n STEP 3: META-LEARNING TRAINING")
        print("-" * 40)
        
        from meta_learning import main as meta_learning_main
        
        logger.info("Training meta-learning model...")
        
        # Run meta-learning training (this will use engineered data and NAS-HPO results)
        meta_learning_main()
        
        print("✅ Meta-learning training completed")
        results['meta_learning'] = 'SUCCESS'
        
    except Exception as e:
        logger.error(f"Meta-learning failed: {str(e)}")
        results['meta_learning'] = f'FAILED: {str(e)}'
        print("Meta-learning failed - will use NAS-HPO results directly...")
    
    try:
        # ==========================================
        # STEP 4: FINAL MODEL TRAINING & EVALUATION
        # ==========================================
        print("\\n STEP 4: FINAL MODEL TRAINING & EVALUATION")
        print("-" * 40)
        
        from final_model_trainer import FinalModelTrainer
        
        logger.info("Training final models and evaluating performance...")
        
        # Initialize final trainer
        final_trainer = FinalModelTrainer(
            data_dir=config['engineered_data_dir'],
            output_dir=config['final_models_output_dir'],
            meta_learning_model_dir=config['meta_learning_output_dir']
        )
        
        # Train and evaluate all datasets
        evaluation_results = final_trainer.train_and_evaluate_all_datasets()
        
        print("Final model training completed")
        results['final_training'] = 'SUCCESS'
        results['evaluation_results'] = evaluation_results
        
        # ==========================================
        # PERFORMANCE SUMMARY
        # ==========================================
        print("\\n PERFORMANCE SUMMARY")
        print("=" * 60)
        
        reference_scores = {
            'bike_sharing_demand': 0.9457,
            'brazilian_houses': 0.9896,
            'superconductivity': 0.9311,
            'wine_quality': 0.4410,
            'yprop_4_1': 0.0778
        }
        
        print(f"{'Dataset':<20} {'Our R²':<10} {'Reference':<10} {'Gap':<8} {'Status'}")
        print("-" * 60)
        
        total_gap = 0
        valid_datasets = 0
        
        for dataset_name, dataset_results in evaluation_results.items():
            if isinstance(dataset_results, list) and len(dataset_results) > 0:
                # Get average R2 across all folds for this dataset
                fold_r2_scores = [fold_result.get('test_r2', 0) for fold_result in dataset_results 
                                 if 'error' not in fold_result and 'test_r2' in fold_result]
                
                if fold_r2_scores:
                    our_r2 = sum(fold_r2_scores) / len(fold_r2_scores)  # Average across folds
                    ref_r2 = reference_scores.get(dataset_name, 0)
                    gap = ref_r2 - our_r2
                    total_gap += abs(gap)
                    valid_datasets += 1
                    
                    if gap < 0.05:
                        status = "🎯 EXCELLENT"
                    elif gap < 0.1:
                        status = "✅ GOOD"
                    elif gap < 0.2:
                        status = "🟡 OK"
                    else:
                        status = "🔴 NEEDS WORK"
                    
                    print(f"{dataset_name:<20} {our_r2:<10.4f} {ref_r2:<10.4f} {gap:<8.4f} {status}")
                else:
                    print(f"{dataset_name:<20} {'ERROR':<10} {'-':<10} {'-':<8} ❌ NO VALID FOLDS")
            else:
                print(f"{dataset_name:<20} {'ERROR':<10} {'-':<10} {'-':<8} ❌ FAILED")
        
        if valid_datasets > 0:
            avg_gap = total_gap / valid_datasets
            print(f"\\nAverage Gap: {avg_gap:.4f}")
            
            if avg_gap < 0.05:
                print("🎉 OUTSTANDING: Pipeline performing at reference level!")
            elif avg_gap < 0.1:
                print("🎯 EXCELLENT: Pipeline performing very well!")
            elif avg_gap < 0.2:
                print("✅ GOOD: Pipeline performing adequately!")
            else:
                print("🔧 IMPROVEMENT NEEDED: Consider hyperparameter tuning")
        
    except Exception as e:
        logger.error(f"Final training failed: {str(e)}")
        results['final_training'] = f'FAILED: {str(e)}'
        print("Final training failed")
    
    # ==========================================
    # PIPELINE SUMMARY
    # ==========================================
    end_time = time.time()
    duration = end_time - start_time
    
    print("\\n" + "=" * 60)
    print(" TRAINING PIPELINE SUMMARY")
    print("=" * 60)
    
    for step, status in results.items():
        if step != 'evaluation_results':
            if 'SUCCESS' in str(status):
                print(f"✅ {step}: {status}")
            else:
                print(f"❌ {step}: {status}")
    
    print(f"\\n⏱️  Total Duration: {duration/60:.1f} minutes")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save pipeline results
    import json
    with open('training_pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n Results saved to: training_pipeline_results.json")
    print("Training pipeline completed!")
    
    return results

if __name__ == "__main__":
    run_training_pipeline()
