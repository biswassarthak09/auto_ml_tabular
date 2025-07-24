"""
AutoML Pipeline Runner
Simple script to run training or testing pipelines

Usage:
    # Run training pipeline
    python run_automl.py train
    
    # Run test pipeline on new dataset
    python run_automl.py test --dataset_path path/to/new/dataset
    
    # Run test pipeline with custom dataset name
    python run_automl.py test --dataset_path path/to/new/dataset --dataset_name my_dataset
"""

import argparse
import sys
from pathlib import Path

def run_training():
    """Run the complete training pipeline"""
    print("🚀 Starting AutoML Training Pipeline...")
    print("This will run: Feature Engineering → NAS-HPO → Meta-Learning → Final Training")
    print()
    
    try:
        from training_pipeline import run_training_pipeline
        results = run_training_pipeline()
        return results
    except Exception as e:
        print(f"❌ Training pipeline failed: {str(e)}")
        return None

from typing import Optional

def run_testing(dataset_path: str, dataset_name: Optional[str] = None):
    """Run the test pipeline on a new dataset"""
    print(f"🧪 Starting AutoML Test Pipeline...")
    print(f"Testing dataset: {dataset_path}")
    print()
    
    try:
        from test_pipeline import AutoMLTestPipeline
        
        test_pipeline = AutoMLTestPipeline()
        results = test_pipeline.test_new_dataset(dataset_path, dataset_name)
        return results
    except Exception as e:
        print(f"❌ Test pipeline failed: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='AutoML Pipeline Runner')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    
    # Testing command  
    test_parser = subparsers.add_parser('test', help='Run test pipeline')
    test_parser.add_argument('--dataset_path', required=True, 
                            help='Path to dataset directory or file')
    test_parser.add_argument('--dataset_name', 
                            help='Custom name for the dataset')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        results = run_training()
        if results:
            print("\\n✅ Training completed successfully!")
        else:
            print("\\n❌ Training failed!")
            sys.exit(1)
            
    elif args.command == 'test':
        if not Path(args.dataset_path).exists():
            print(f"❌ Dataset path does not exist: {args.dataset_path}")
            sys.exit(1)
            
        results = run_testing(args.dataset_path, args.dataset_name)
        if results and results.get('model_trained', False):
            print("\\n✅ Testing completed successfully!")
        else:
            print("\\n❌ Testing failed!")
            sys.exit(1)
            
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
