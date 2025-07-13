from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging

from typing import Optional, Union

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

METRICS = {"r2": r2_score}

class AutoML:
    def __init__(
        self,
        seed: int,
        model_type: str = "lgbm",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        learning_rate: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Initialize AutoML with specified model and hyperparameters.

        Args:
            seed: Random seed for reproducibility.
            model_type: One of ['lgbm', 'xgb', 'dummy'].
            n_estimators: Number of trees (if applicable).
            max_depth: Maximum tree depth (if applicable).
            learning_rate: Learning rate for boosters (if applicable).
            **kwargs: Additional backend-specific parameters.
        """
        self.seed = seed
        self.metric = METRICS["r2"]
        self._model_type = model_type.lower()
        self._model: Optional[Union[LGBMRegressor, XGBRegressor, DummyRegressor]] = None

        if self._model_type == "lgbm":
            self._model = LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=seed,
                **kwargs,
            )
        elif self._model_type == "xgb":
            self._model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=seed,
                **kwargs,
            )
        elif self._model_type == "dummy":
            self._model = DummyRegressor()
        else:
            raise ValueError(f"Unknown model_type={model_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> AutoML:
        """
        Fit the underlying model using a train/validation split for early feedback.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            random_state=self.seed,
            test_size=0.2,
        )

        self._model.fit(X_train, y_train)  # type: ignore
        val_preds = self._model.predict(X_val)  # type: ignore
        val_score = self.metric(y_val, val_preds)
        logger.info(f"Validation score: {val_score:.4f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        """
        if self._model is None:
            raise ValueError("Model not fitted or unknown model_type")
        return self._model.predict(X)  # type: ignore