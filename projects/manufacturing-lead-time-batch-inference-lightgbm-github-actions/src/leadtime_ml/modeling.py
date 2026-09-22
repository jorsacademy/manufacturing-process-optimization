from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", model),
        ]
    )


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str | Path) -> Pipeline:
    return joblib.load(path)
