# Ensemble Classification

A rigorous tabular classification project: predicting U.S. work-visa case outcomes (certified vs denied) from employer and applicant attributes, using ensemble methods evaluated to a clinical-research standard.

The point of this repo is not the visa problem. It is the **evaluation discipline**: nested cross-validation, calibration, interpretability, and honest model comparison applied to a tabular classification task. It doubles as the classical-baseline template for the Clinical-QML Benchmark (forthcoming).

## Status

Rigor layer implemented. The original coursework baseline lives in `notebooks/01-ensemble-classification.ipynb` and is preserved for narrative continuity; the reusable, leakage-safe evaluation pipeline lives in `src/` and is driven by `scripts/run_benchmark.py`, which regenerates every metric and figure in `results/`. See `docs/learnings.md` for the modeling lessons and `results/SUMMARY.md` for the current numbers.

## Dataset

~25,480 records, 11 features, binary target (`case_status`: certified or denied). Features cover applicant education and experience, employer size and age, region, prevailing wage, and position type.

**Source and license.** `data/EasyVisa.csv` is a derived teaching dataset built from the U.S. Department of Labor OFLC permanent-labor-certification program disclosure data (FY2016 vintage). The underlying OFLC data is public-domain U.S. government work; the file here is a recoded, sampled subset of it rather than the raw disclosure release (for example, `continent` is rolled up from country of citizenship, `region_of_employment` from worksite state, and `case_id` is a synthetic identifier). Factual government data of this kind is not subject to copyright, so the file is redistributed here solely to make the analysis reproducible.

## Approach and roadmap

- [x] Baseline ensemble models (bagging, boosting, random forest) — `notebooks/01`.
- [x] Leakage-safe preparation: id dropped, employee-count sign error repaired, company age derived against a documented vintage, and prevailing wage kept raw with its unit retained as a categorical (annualizing was tested and rejected, it destroys the signal) — `src/data.py`.
- [x] All learned transforms (encoding, scaling) and resampling fit **inside** each CV fold via an imbalanced-learn pipeline — `src/preprocessing.py`, `src/models.py`.
- [x] Nested cross-validation: outer loop for the performance estimate, inner loop tuned on ROC AUC (the decision metric) — `src/evaluation.py:nested_cv_score`.
- [x] Expanded model set compared on one protocol: logistic regression, random forest, XGBoost, LightGBM, CatBoost — `src/models.py`.
- [x] Calibration: isotonic vs uncalibrated, Brier score and calibration curves — `src/evaluation.py:calibration_report`.
- [x] Interpretability: permutation importance (not biased impurity importance) plus SHAP on the winning model — `src/evaluation.py`, `scripts/run_benchmark.py`.
- [x] Honest comparison: ROC and PR curves with bootstrap confidence intervals on the held-out test set — `src/evaluation.py:compare_models`.

## Structure

```
ensemble-classification/
  data/        EasyVisa.csv
  notebooks/   01-ensemble-classification.ipynb   (original baseline, preserved)
  src/         data.py, preprocessing.py, models.py, evaluation.py  (the rigor layer)
  scripts/     run_benchmark.py                     (end-to-end driver)
  results/     metrics (json), SUMMARY.md, and figures written by the driver
  docs/        learnings.md, REVIEW-ensemble-classification.md
```

## Reproducing the results

Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python scripts/run_benchmark.py        # full nested-CV benchmark
uv run python scripts/run_benchmark.py --sample 6000   # fast smoke run
```

The driver writes `results/SUMMARY.md`, per-stage JSON metrics, and figures (ROC/PR, calibration, permutation importance, SHAP summary).

## Provenance

This project originated as coursework and has been rebuilt as an independent, rigor-upgraded portfolio piece. All third-party course materials (slides, templates, provided low-code scaffolds) have been removed; the work here is my own.
