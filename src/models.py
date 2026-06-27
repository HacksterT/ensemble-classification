"""Model factory.

Each entry returns an imbalanced-learn Pipeline of
``preprocessor -> RandomUnderSampler -> classifier`` plus a hyperparameter grid.
Resampling sits *inside* the pipeline so it is applied only to the training rows
of each fold, never to validation or test data. Undersampling is the chosen
strategy because the baseline analysis found it beat SMOTE for recovering the
minority (denial) class at equal AUC (see docs/learnings.md, item 4).
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .preprocessing import build_preprocessor

RANDOM_STATE = 42


def _pipe(classifier, scale_numeric: bool = False) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("undersample", RandomUnderSampler(random_state=RANDOM_STATE)),
            ("clf", classifier),
        ]
    )


def build_models(random_state: int = RANDOM_STATE) -> dict:
    """Return {name: (pipeline, param_grid)} for every candidate model.

    Grids are deliberately small: the baseline analysis showed tuning depth and
    leaf size is the dominant lever and that all tree families converge to the
    same ceiling, so the search targets regularization, not breadth.
    """
    models: dict = {}

    # Linear baseline: a genuinely different family, included so the model
    # comparison can show whether trees actually beat a regularized linear model.
    models["logistic_regression"] = (
        _pipe(
            LogisticRegression(max_iter=2000, random_state=random_state),
            scale_numeric=True,
        ),
        {"clf__C": [0.1, 1.0]},
    )

    # Random forest: the baseline reference model.
    models["random_forest"] = (
        _pipe(
            RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
        ),
        {"clf__max_depth": [10, 20], "clf__min_samples_leaf": [1, 4]},
    )

    # XGBoost.
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = (
            _pipe(
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                )
            ),
            {"clf__max_depth": [4, 6]},
        )
    except ImportError:
        pass

    # LightGBM.
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = (
            _pipe(
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
            ),
            {"clf__num_leaves": [31, 63]},
        )
    except ImportError:
        pass

    # CatBoost.
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = (
            _pipe(
                CatBoostClassifier(
                    iterations=400,
                    learning_rate=0.1,
                    random_state=random_state,
                    verbose=False,
                    allow_writing_files=False,
                )
            ),
            {"clf__depth": [4, 6]},
        )
    except ImportError:
        pass

    return models
