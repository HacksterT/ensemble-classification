"""End-to-end rigor benchmark for the visa-case classification task.

Runs nested cross-validation across all candidate models, selects the winner by
mean outer ROC AUC, then evaluates it on a held-out test split with bootstrap
confidence intervals, calibration, permutation importance, and SHAP. Writes
machine-readable metrics and figures to ``results/``.

The train/test split is deterministic (fixed seed on the deterministic prepared
frame), so the benchmark is resumable: each model's nested CV can be computed in
its own process and cached, then assembled. This keeps memory bounded on small
hosts.

Usage:
    python scripts/run_benchmark.py                  # full run, in one process
    python scripts/run_benchmark.py --only random_forest --outer 5 --inner 3
    python scripts/run_benchmark.py --assemble --outer 5 --inner 3
    python scripts/run_benchmark.py --sample 6000    # fast smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
CACHE = RESULTS / "_cache"
RESULTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from src import data as D  # noqa: E402
from src import evaluation as E  # noqa: E402
from src.models import build_models  # noqa: E402

RANDOM_STATE = 42


def get_split(sample):
    frame = D.prepare(D.load_raw())
    if sample:
        frame = frame.groupby(D.TARGET, group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample // 2), random_state=RANDOM_STATE)
        )
    X, y = D.split_xy(frame)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def run_one(name, sample, n_outer, n_inner):
    X_tr, _, y_tr, _ = get_split(sample)
    pipe, grid = build_models()[name]
    print(f"[nested CV] {name} on train={X_tr.shape} ...", flush=True)
    rows = E.nested_cv_score(pipe, X_tr, y_tr, grid, n_outer=n_outer, n_inner=n_inner,
                             scoring="roc_auc", random_state=RANDOM_STATE)
    summary = E.summarize_nested(rows)
    (CACHE / f"nested_{name}.json").write_text(
        json.dumps({"summary": summary, "folds": rows}, indent=2, default=str))
    print(f"    ROC AUC {summary['roc_auc']['mean']:.4f} +/- {summary['roc_auc']['std']:.4f} | "
          f"denial recall {summary['denial_recall']['mean']:.4f}  [cached]")


def assemble(sample, n_inner):
    X_tr, X_te, y_tr, y_te = get_split(sample)
    models = build_models()

    nested = {}
    for name in models:
        f = CACHE / f"nested_{name}.json"
        if f.exists():
            nested[name] = json.loads(f.read_text())
    if not nested:
        raise SystemExit("No cached nested-CV results. Run --only <model> first.")
    winner = max(nested, key=lambda m: nested[m]["summary"]["roc_auc"]["mean"])
    print(f"WINNER (mean outer ROC AUC): {winner}")

    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_STATE)
    fitted, tuned = {}, {}
    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1)
        gs.fit(X_tr, y_tr)
        fitted[name] = gs.best_estimator_
        tuned[name] = gs.best_params_

    comparison = E.compare_models(fitted, X_te, y_te)
    win_est = fitted[winner]
    cal, _ = E.calibration_report(win_est, X_tr, y_tr, X_te, y_te)
    importance = E.permutation_importances(win_est, X_te, y_te, n_repeats=8)
    curves = E.roc_pr_points(win_est, X_te, y_te)

    (RESULTS / "nested_cv.json").write_text(json.dumps(nested, indent=2, default=str))
    (RESULTS / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    (RESULTS / "calibration.json").write_text(json.dumps(cal, indent=2))
    (RESULTS / "permutation_importance.json").write_text(json.dumps(importance, indent=2))
    (RESULTS / "tuned_params.json").write_text(json.dumps(tuned, indent=2, default=str))
    (RESULTS / "SUMMARY.md").write_text(
        build_summary_md(nested, winner, comparison, tuned, cal, importance, X_tr, X_te, y_tr.append(y_te) if hasattr(y_tr, "append") else pd.concat([y_tr, y_te])))
    make_figures(comparison, curves, cal, importance, winner)
    run_shap(win_est, X_te, winner)
    print("DONE. Results written to results/.")


def build_summary_md(nested, winner, comparison, tuned, cal, importance, X_tr, X_te, y):
    L = ["# Rigor Benchmark Results", ""]
    n_outer = len(next(iter(nested.values()))["folds"])
    L.append(f"Train rows: {len(X_tr)}, test rows: {len(X_te)}, positive (Certified) rate: {float(np.mean(y)):.3f}.")
    L.append(f"Nested CV: {n_outer} outer folds, inner tuning on ROC AUC. Held-out test split: 20%.")
    L.append(f"Winner by mean outer ROC AUC: **{winner}**.")
    L += ["", "## Nested cross-validation (outer-fold mean +/- std, training split)", "",
          "| Model | ROC AUC | PR AUC | Denial recall | Brier |", "|---|---|---|---|---|"]
    for name, d in sorted(nested.items(), key=lambda kv: -kv[1]["summary"]["roc_auc"]["mean"]):
        s = d["summary"]
        L.append(f"| {name} | {s['roc_auc']['mean']:.3f} +/- {s['roc_auc']['std']:.3f} | "
                 f"{s['pr_auc']['mean']:.3f} | {s['denial_recall']['mean']:.3f} | {s['brier']['mean']:.3f} |")
    L += ["", "## Held-out test comparison (tuned, refit on full train)", "",
          "| Model | ROC AUC | 95% CI | PR AUC | Denial recall | Brier |", "|---|---|---|---|---|---|"]
    for name, d in sorted(comparison.items(), key=lambda kv: -kv[1]["roc_auc"]):
        ci = d["auc_ci"]
        L.append(f"| {name} | {d['roc_auc']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                 f"{d['pr_auc']:.3f} | {d['denial_recall']:.3f} | {d['brier']:.3f} |")
    L += ["", "## Calibration of the winner (test split)", "",
          f"Brier uncalibrated: {cal['uncalibrated']['brier']:.4f}; isotonic-calibrated: {cal['isotonic']['brier']:.4f}.",
          "", "## Permutation importance of the winner (top 8, ROC AUC drop)", "",
          "| Feature | Importance | Std |", "|---|---|---|"]
    for r in importance[:8]:
        L.append(f"| {r['feature']} | {r['importance_mean']:.4f} | {r['importance_std']:.4f} |")
    L += ["", "_Generated by scripts/run_benchmark.py._"]
    return "\n".join(L) + "\n"


def make_figures(comparison, curves, cal, importance, winner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(curves["fpr"], curves["tpr"], label=winner)
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC — {winner}")
    ax[0].legend()
    ax[1].plot(curves["recall"], curves["precision"], label=winner)
    ax[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision–Recall — {winner}")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS / "roc_pr.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    for name in ("uncalibrated", "isotonic"):
        ax.plot(cal[name]["curve_mean_predicted"], cal[name]["curve_fraction_positive"],
                marker="o", label=f"{name} (Brier {cal[name]['brier']:.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title=f"Calibration — {winner}")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "calibration.png", dpi=120); plt.close(fig)

    top = importance[:10][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([r["feature"] for r in top], [r["importance_mean"] for r in top],
            xerr=[r["importance_std"] for r in top])
    ax.set(xlabel="Mean ROC AUC drop when permuted", title=f"Permutation importance — {winner}")
    fig.tight_layout(); fig.savefig(RESULTS / "permutation_importance.png", dpi=120); plt.close(fig)


def run_shap(win_est, X_te, winner, n: int = 600):
    try:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pre = win_est.named_steps["preprocess"]
        clf = win_est.named_steps["clf"]
        Xs = X_te.sample(min(n, len(X_te)), random_state=RANDOM_STATE)
        Xt_df = pd.DataFrame(pre.transform(Xs), columns=pre.get_feature_names_out())
        sv = shap.TreeExplainer(clf).shap_values(Xt_df)
        if isinstance(sv, list):
            sv = sv[1]
        shap.summary_plot(sv, Xt_df, show=False, max_display=12)
        plt.tight_layout(); plt.savefig(RESULTS / "shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()
        print("SHAP summary written.")
    except Exception as exc:
        print(f"SHAP skipped: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--only", type=str, default=None, help="run nested CV for one model and cache it")
    ap.add_argument("--assemble", action="store_true", help="assemble cached nested results into final outputs")
    a = ap.parse_args()

    if a.only:
        run_one(a.only, a.sample, a.outer, a.inner)
    elif a.assemble:
        assemble(a.sample, a.inner)
    else:
        for name in build_models():
            run_one(name, a.sample, a.outer, a.inner)
        assemble(a.sample, a.inner)


if __name__ == "__main__":
    main()
