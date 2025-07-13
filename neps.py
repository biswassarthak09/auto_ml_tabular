from automl.data import Dataset # your loader in data.py
from automl.automl import AutoML # your AutoML class in automl.py
import argparse
import json
from pathlib import Path

import optuna
import numpy as np
import pandas as pd




def parse_args():
    parser = argparse.ArgumentParser("NEPS: Neural/Ensemble/Probabilistic Search")
    parser.add_argument("--task", type=str, required=True, help="Dataset task name")
    parser.add_argument("--mode", type=str, choices=["raw", "engineered"], default="raw",
                        help="Use raw features or engineered features")
    parser.add_argument("--fold", type=int, default=1, help="Fold number (if applicable)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, required=True, help="Path to write results JSON")
    return parser.parse_args()


class TrialCallback:
    def __init__(self, study, trial):
        self.study = study
        self.trial = trial

    def __call__(self, study, trial):
        pass  # can integrate MLflow or CSV logging here


def objective(trial: optuna.Trial, X_train, X_valid, y_train, y_valid, seed: int):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
    }
    
    model = AutoML(seed=seed, model_type='lgbm', n_estimators=params["n_estimators"], max_depth=params["max_depth"], learning_rate=params["learning_rate"])
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    
    # compute R^2
    score = np.corrcoef(y_valid, preds)[0, 1] ** 2
    return score


def run_neps(datadir: Path, task: str, mode: str, fold: int, seed: int):
    # Load data
    if mode == 'raw':
        ds = Dataset.load(datadir, task, fold)
        X_train, X_valid = ds.X_train, ds.X_test
        y_train, y_valid = ds.y_train, ds.y_test
    else:
        X_train, X_valid, y_train, y_valid = Dataset.load_and_preprocess(datadir, task, fold)

    # Create Optuna study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: objective(t, X_train, X_valid, y_train, y_valid, seed), n_trials=50)

    # Gather results
    best = study.best_trial
    results = {
        'best_trial': {
            'params': best.params,
            'value': best.value
        }
    }
    return results


def main():
    args = parse_args()
    results = run_neps(Path('data'), args.task, args.mode, args.fold, args.seed)
    
    # Ensure output folder exists
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Write results
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"NEPS completed: best R^2 = {results['best_trial']['value']:.4f}")


if __name__ == '__main__':
    main()
