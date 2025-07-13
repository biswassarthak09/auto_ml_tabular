from __future__ import annotations

# import pandas as pd
# from pathlib import Path
from dataclasses import dataclass
# from sklearn.pipeline import Pipeline
# from sklearn.compose  import ColumnTransformer
# from sklearn.impute   import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
import pandas as pd
from sklearn.pipeline       import Pipeline
from sklearn.compose        import ColumnTransformer
from sklearn.impute         import SimpleImputer
from sklearn.preprocessing  import OneHotEncoder, StandardScaler


import logging

logger = logging.getLogger(__name__)

@dataclass
class Dataset:
    path: Path
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series | None = None

    @classmethod
    def load(cls, datadir: Path, task: str, fold: int) -> Dataset:
        if task == "exam_dataset":
            path = datadir / task / str(1)
        else:
            path = datadir / task / str(fold)
        if not path.exists():
            raise FileNotFoundError(path)

        X_train_path = path / "X_train.parquet"
        y_train_path = path / "y_train.parquet"
        X_test_path = path / "X_test.parquet"
        y_test_path = path / "y_test.parquet"

        return Dataset(
            path=path,
            X_train=pd.read_parquet(X_train_path),
            y_train=pd.read_parquet(y_train_path).iloc[:, 0],
            X_test=pd.read_parquet(X_test_path),
            y_test=pd.read_parquet(y_test_path).iloc[:, 0]
            if y_test_path.exists()
            else None,
        )
    @staticmethod
    def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        
        num_pipe = Pipeline([
            ("i0mpute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        
        return ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ], remainder="drop")
        
    @classmethod
    def load_and_preprocess(cls, datadir: Path, task: str, fold: int):
        
        """
        Load via Dataset.load(), fit the preprocessor on train,
        transform both train & test, and return arrays.
        """
        #load raw splits
        ds = Dataset.load(Path(datadir), task, fold)
        
        #build and fit the preprocessor
        pre = ds.build_preprocessor(ds.X_train)
        X_tr = pre.fit_transform(ds.X_train)
        X_te = pre.transform(ds.X_test)
        
        #return as arrays
        return X_tr, X_te, ds.y_train.values, ds.y_test.values