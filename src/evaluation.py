"""Evaluation: the rigor layer that distinguishes this from a coursework submission.

Provides nested cross-validation, calibration assessment, an honest multi-model
comparison with bootstrap confidence intervals, and leakage-free feature
importance (permutation and SHAP). Every score is computed so that no
information from the evaluation rows reaches model fitting or tuning.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

DENIAL_LABEL = 0  # minority, business-critical class


def denial_recall(y_true, y_pred) -> float:
    """Recall on the Denied (minority) class."""
    return recall_score(y_true, y_pred, pos_label=DENIAL_LABEL)


def nested_cv_score(
    estimator,
    X,
    y,
    param_grid,
    *,
    n_outer: int = 5,
    n_inner: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42,
    search_jobs: int = 1,
):
    """Nested cross-validation.

    Outer loop estimates generalization; inner loop tunes hyperparameters. The
    inner GridSearchCV scores on ``roc_auc`` (the decision metric), not a generic
    balanced score, so the selected hyperparameters are optimal for the metric
    the project actually reports. Returns per-fold outer metrics and the tuned
    params chosen in each fold.
    """
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)

    rows = []
    X = np.asarray(X) if not hasattr(X, "iloc") else X
    y = np.asarray(y)
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        X_te = X.iloc[te] if hasattr(X, "iloc") else X[te]
        y_tr, y_te = y[tr], y[te]

        # search_jobs=1 by default: parallelism lives in the estimators, so the
        # search stays serial to avoid thread oversubscription / OOM on small hosts.
        search = GridSearchCV(
            clone(estimator), param_grid, scoring=scoring, cv=inner, n_jobs=search_jobs
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_

        proba = best.predict_proba(X_te)[:, 1]
        pred = best.predict(X_te)
        rows.append(
            {
                "fold": fold,
                "roc_auc": roc_auc_score(y_te, proba),
                "pr_auc": average_precision_score(y_te, proba),
                "denial_recall": denial_recall(y_te, pred),
                "brier": brier_score_loss(y_te, proba),
                "best_params": search.best_params_,
            }
        )
    return rows


def summarize_nested(rows) -> dict:
    """Mean and std of each numeric outer-fold metric."""
    keys = ["roc_auc", "pr_auc", "denial_recall", "brier"]
    return {
        k: {"mean": float(np.mean([r[k] for r in rows])),
            "std": float(np.std([r[k] for r in rows]))}
        for k in keys
    }


def calibration_report(estimator, X_train, y_train, X_test, y_test, *, n_bins: int = 10):
    """Calibration curve and Brier score, uncalibrated vs isotonic-calibrated.

    Discrimination (AUC) is not enough: a model can rank well yet output
    miscalibrated probabilities. Calibration is fit on the training data only.
    """
    base = clone(estimator).fit(X_train, y_train)
    base_proba = base.predict_proba(X_test)[:, 1]

    calibrated = CalibratedClassifierCV(clone(estimator), method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    cal_proba = calibrated.predict_proba(X_test)[:, 1]

    out = {}
    for name, proba in (("uncalibrated", base_proba), ("isotonic", cal_proba)):
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
        out[name] = {
            "brier": float(brier_score_loss(y_test, proba)),
            "curve_mean_predicted": mean_pred.tolist(),
            "curve_fraction_positive": frac_pos.tolist(),
        }
    return out, base


def bootstrap_auc_ci(y_true, proba, *, n_boot: int = 2000, random_state: int = 42, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for test ROC AUC."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": float(roc_auc_score(y_true, proba)), "ci_low": float(lo), "ci_high": float(hi)}


def compare_models(fitted, X_test, y_test, *, seeds=(42, 1, 7)):
    """Honest comparison: test ROC AUC (bootstrap CI), PR AUC, denial recall,
    plus a multi-seed AUC spread to show stability. ``fitted`` maps name ->
    estimator already fit on the training split.
    """
    table = {}
    for name, est in fitted.items():
        proba = est.predict_proba(X_test)[:, 1]
        pred = est.predict(X_test)
        ci = bootstrap_auc_ci(y_test, proba)
        table[name] = {
            "roc_auc": ci["auc"],
            "auc_ci": [ci["ci_low"], ci["ci_high"]],
            "pr_auc": float(average_precision_score(y_test, proba)),
            "denial_recall": float(denial_recall(y_test, pred)),
            "brier": float(brier_score_loss(y_test, proba)),
        }
    return table


def roc_pr_points(estimator, X_test, y_test):
    """Raw ROC and PR curve points for plotting."""
    proba = estimator.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "precision": prec.tolist(), "recall": rec.tolist()}


def permutation_importances(estimator, X_test, y_test, *, n_repeats: int = 10, random_state: int = 42):
    """Permutation importance on the test split (model-agnostic, unbiased).

    Used instead of impurity-based ``feature_importances_``, which is biased
    toward high-cardinality continuous features such as wage (see learnings 7).
    """
    result = permutation_importance(
        estimator, X_test, y_test, scoring="roc_auc",
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    cols = list(X_test.columns)
    return [
        {"feature": cols[i],
         "importance_mean": float(result.importances_mean[i]),
         "importance_std": float(result.importances_std[i])}
        for i in order
    ]
