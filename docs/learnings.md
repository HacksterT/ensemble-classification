# Modeling Learnings: Visa-Case Ensemble Classification

Lessons from the modeling work in this repo. Scope is the machine learning models and methodology, not the EDA. Target is `case_status` (1 = Certified, 67%, 0 = Denied, 33%). All resampled models balance the **training** split only; evaluation data keeps the natural distribution. Headline metric is ROC AUC, with denial (minority-class) recall as the business-critical secondary metric.

The work happened in two passes. The first is the single-split baseline in `notebooks/01-ensemble-classification.ipynb`. The second is the rigor layer in `src/`, which re-ran the comparison under nested cross-validation with a held-out test set, bootstrap confidence intervals, calibration, and permutation importance (`scripts/run_benchmark.py`, results in `results/`). Where the second pass sharpened or corrected a first-pass claim, it is noted inline rather than bolted on at the end.

## Results that anchor these learnings

Single-split baseline (notebook 01), tuned on the training split, scored on one held-out split:

| Stage / Model | ROC AUC | Accuracy | Denial Recall | Denial Precision |
|---|---|---|---|---|
| RF, original (default params) | 0.707 | 0.67 | 0.50 | 0.51 |
| RF, SMOTE oversample | 0.704 | 0.67 | 0.51 | 0.50 |
| RF, random undersample | 0.708 | 0.64 | 0.64 | 0.47 |
| **RF, tuned + undersample** | **0.788** | **0.722** | 0.672 | 0.568 |
| Gradient Boosting (undersample) | 0.782 | 0.709 | 0.702 | 0.549 |
| XGBoost (undersample) | 0.779 | 0.710 | 0.700 | 0.549 |
| AdaBoost (undersample) | 0.784 | 0.703 | **0.734** | 0.538 |
| Weighted Ensemble | 0.788 | 0.717 | 0.701 | 0.558 |

Rigor layer (`src/`), nested cross-validation for model selection, then a single held-out 20% test set for the final comparison with 95% bootstrap confidence intervals:

| Model | Test ROC AUC | 95% CI | PR AUC | Denial Recall | Brier |
|---|---|---|---|---|---|
| random_forest | 0.767 | [0.753, 0.781] | 0.869 | 0.664 | 0.197 |
| catboost | 0.765 | [0.751, 0.779] | 0.867 | 0.676 | 0.197 |
| xgboost | 0.759 | [0.745, 0.774] | 0.862 | 0.678 | 0.200 |
| lightgbm | 0.758 | [0.744, 0.771] | 0.864 | 0.686 | 0.204 |
| logistic_regression | 0.751 | [0.736, 0.764] | 0.856 | 0.684 | 0.203 |

The honest, leakage-free numbers sit slightly below the single-split baseline (0.75 to 0.77 versus 0.788). That gap is the optimism the nested protocol removes, and it is the point of the second pass.

## 1. Hyperparameter tuning was the dominant lever, not algorithm choice

Tuning a single Random Forest moved ROC AUC from 0.707 to 0.788, a +0.081 gain. Every subsequent algorithm swap landed inside a 0.009 band of that tuned RF. The single most consequential parameter was `max_depth`: the default unbounded forest was memorizing the training set and generalizing poorly. The nested-CV pass independently re-selected `max_depth=10, min_samples_leaf=4`, the same neighborhood the baseline grid search found, which is reassuring. The practical order of operations is regularize first, swap algorithms last. Time spent grid-searching depth and leaf size returned roughly ten times what time spent trying a different booster did.

## 2. All model families converge to the same ceiling (~0.79 AUC)

Random Forest, Gradient Boosting, XGBoost, AdaBoost, LightGBM, CatBoost, and even a regularized logistic regression all finish within a couple of points of each other. The nested-CV comparison makes this rigorous: the five models' 95% bootstrap confidence intervals on the held-out test set overlap almost completely (every interval sits inside roughly [0.74, 0.78]). Once the data is balanced and the models are reasonably regularized, the algorithm family stops mattering. The ceiling is set by the signal in the available features, not by model sophistication. Chasing marginal AUC by adding more boosters is a dead end; the lever is new features, not new models.

## 3. The ensemble bought nothing over the best single model

The first-pass weighted ensemble tied the tuned RF exactly. The second pass explains why and closes the door on it: a regularized linear model, a genuinely different family, lands inside the same confidence band as the tree ensembles. Ensembling pays off when base learners are decorrelated and make different errors; here even a linear learner agrees, so there is no diversity left to exploit by combination. An ensemble of agreeing models is not automatically better, and the combination effort was zero-yield. If ensembling is revisited it needs a genuinely different feature view, not another tree variant.

## 4. Random undersampling beat SMOTE for recovering the minority class

SMOTE oversampling produced essentially no change over the imbalanced baseline (0.704 vs 0.707 AUC; denial recall 0.51 vs 0.50). Random undersampling lifted denial recall from 0.50 to 0.64 at the same AUC, by forcing the model to stop defaulting to the majority "Certified" class. The named tradeoff: undersampling discards roughly half the majority rows, which usually risks variance, but here it did not cost AUC and materially improved the metric that matters. Synthetic oversampling was not worth its complexity; plain undersampling was the better-and-simpler choice, and it is what the `src/` pipeline uses inside every fold.

## 5. Accuracy is the wrong headline metric here

With 67% Certified, a model that predicts "Certified" for everyone scores 0.67 accuracy while being useless for the actual business question of flagging likely denials. The baseline RF's 0.67 accuracy was almost exactly that null model. ROC AUC and per-class recall exposed the difference that accuracy hid. The repo reports denial recall and ROC AUC as primary, and treats accuracy as a sanity check only.

## 6. Model selection is really a choice of operating point

Moving across the tuned models traces a single precision-recall tradeoff: denial recall rises while approval recall falls in step. These are not meaningfully different models so much as points on one curve. The "which algorithm" question collapses into "where on the recall tradeoff do you want to sit," which can equivalently be tuned by moving the decision threshold on one model rather than by switching algorithms. If the business cost of missing a denial exceeds the cost of missing an approval, push toward higher denial recall or lower the threshold; otherwise the balanced RF is the default.

## 7. Impurity-based feature importance is misleading; permutation importance corrected it

The untuned RF assigned `prevailing_wage` a ~0.65 Gini importance, dwarfing everything, a textbook case of impurity-importance bias toward high-cardinality continuous features. The rigor layer settled it with permutation importance on the held-out set, measured as ROC AUC drop: `education_of_employee` is the dominant driver (0.135), `has_job_experience` second (0.056), and `prevailing_wage` falls to the very bottom (0.002), below even `unit_of_wage` (0.018). The continuous wage variable that `feature_importances_` had crowned is, under an unbiased measure, almost irrelevant. Any feature-importance claim in this repo comes from permutation importance or SHAP, never the default attribute.

## 8. Leakage discipline is non-negotiable, and the honest score is the proof

In the baseline, resampling was applied to the training partition after the stratified split, never to the full dataset, the correct discipline. The rigor layer extends that guarantee to every learned transform: one-hot encoding, scaling, and resampling are all fit inside each cross-validation fold via an imbalanced-learn pipeline, and the full-data quantile binning of the wage was dropped in favor of the raw continuous value rather than bin edges computed across train and test together. The payoff is visible: the leakage-free test AUC (0.75 to 0.77) sits below the single-split baseline (0.788). That half-to-one point is exactly the inflation that careless pipelines ship as real performance.

## 9. The substantive finding: certification is qualification-driven

The models agree that education level and job experience carry the signal, with region, continent, and wage unit secondary and raw pay level near-irrelevant once measured without bias. The performance ceiling near 0.79 AUC and ~0.57 denial precision says the available features explain certification only moderately. This is not a tuning failure; it is the information limit of the dataset. Meaningful improvement requires new information (employer history, occupation/SOC detail, application timing, regional labor-market context), not further model iteration.

A specific corollary, because it is a transferable mistake: standardizing the wage across pay units was unnecessary and actively harmful. The obvious move is to put every wage on a common annual basis by multiplying Hourly by 2080, Weekly by 52, and so on. Doing it makes things worse. It bakes in an unverified full-time assumption and manufactures implausible magnitudes (Week-unit wages annualize to a median near $4M), and crucially it destroys signal rather than adding it: raw `prevailing_wage` separates the target at univariate AUC 0.55, while the annualized version scores 0.43, below chance. The reason is that raw wage magnitude partly proxies `unit_of_wage`, which is itself predictive (certification runs 35% for Hourly cases versus 70% for Annual), and rescaling to a common unit erases that proxy. The lesson generalizes: do not normalize across a grouping variable when the grouping variable is itself the signal, and always check a candidate transform against univariate separation before trusting it. The repo keeps raw wage and lets the model see the unit as a categorical.

## 10. Discrimination is not calibration

A model can rank cases well and still report probabilities you should not trust at face value. The tuned Random Forest scores a Brier of 0.197 uncalibrated; isotonic calibration improves it to 0.178 without changing its ranking (AUC is unaffected). For any setting where the predicted probability feeds a decision rule or a cost calculation, calibration is a separate property that has to be measured and, here, corrected. The repo reports Brier and calibration curves alongside AUC for this reason.

## 11. The tuning objective should match the decision metric

The baseline grid search optimized weighted F1, but the repo reports and decides on ROC AUC and denial recall. That is a quiet inconsistency: the hyperparameters that win under weighted F1 are not necessarily the ones that win under the metric the business cares about, and weighted F1 rewards majority-class performance in proportion to its prevalence, exactly the bias the rest of this analysis works to counteract. The `src/` nested CV tunes its inner loop on ROC AUC, the decision metric. A model selection procedure is only as honest as the metric it optimizes.

## Recommended posture for this repo

Treat the tuned Random Forest as the reference model and the operating-point choice as the real decision, set by the relative business cost of a missed denial versus a missed approval. Report ROC AUC and denial recall, report calibration alongside them, use permutation importance or SHAP for any feature story, and keep all learned transforms and resampling inside the training fold. Park further algorithm experimentation; direct future effort at feature acquisition, which is the only remaining lever on the ~0.79 ceiling.

## Final summary: can a better model or more tuning break the ceiling?

No, and the case is overdetermined. Six model families, Random Forest, gradient boosting, XGBoost, LightGBM, CatBoost, and a regularized logistic regression, all land in a ROC AUC band of roughly [0.74, 0.78] with held-out bootstrap confidence intervals that overlap almost completely. When a linear model sits inside the same interval as tuned gradient boosting, the accuracy is not being left on the table for want of a more sophisticated learner; the models have reached the information content of the features. The nested-CV tuning re-selected the same regularization the original grid search found, which is the signature of a ceiling rather than an under-explored search space. A direct test of the most defensible engineered feature confirms it: a within-unit wage rank (wage relative to others paid in the same unit, the principled version of the standardization rejected in learning 9) is uninformative, univariate AUC 0.49, and adding it to the Random Forest moved AUC down a hair, inside noise. Trees already capture whatever conditional wage signal exists by splitting on unit and then on raw wage, so the explicit feature is redundant.

What will not help, and why: more aggressive or Bayesian hyperparameter search will not break a band six families already converge to; stacking or a meta-learner will not help because the base learners are correlated and already agree (the reason the original weighted ensemble bought nothing); a neural network will underperform on tabular data of this size, as it nearly always does. Two moves people reach for, threshold tuning and probability calibration, are worth doing but do not change AUC at all, because AUC is threshold-independent: they shift denial recall and Brier, not discrimination.

The only lever that moves AUC meaningfully is new information, and the data points to where the missing signal lives. In rough order of expected payoff: employer-level history (the prior approval or denial rate for the same sponsor or filing attorney, almost certainly the strongest absent predictor, since outcomes cluster by sponsor); occupation or SOC code (what the job is, not merely its pay); filing timing (fiscal year and quarter, which encode policy regime and cap pressure); the ratio of offered wage to the actual prevailing wage for that occupation and area (the legitimate wage-competitiveness signal the raw column can only gesture at); and regional labor-market context such as local unemployment or occupation demand. These are data-acquisition tasks, not modeling tasks.

A final distinction, because it changes the answer depending on the goal. If the objective is the decision value of the model rather than AUC specifically, that is improvable without new data: cost-sensitive training, threshold optimization against the real cost of a missed denial, and the calibration already in place all make the model more useful at its operating point even though the ROC curve stays where the features put it. If the objective is AUC, the curve is fixed until the feature set changes.
