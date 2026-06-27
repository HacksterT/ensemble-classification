# Deeper Learnings: The Theory Behind the Models and the Method

A teaching document. The other file, `learnings.md`, records *what we found*. This file explains *what we are doing and why*, with definitions of the moving parts. It is written to be read top to bottom; each section assumes the one before it. Where a term first appears in bold, that is its working definition.

---

## 1. What problem are we actually solving

Every visa application in the data has an outcome: it was either **Certified** (approved) or **Denied**. We have a set of recorded attributes for each application (education, job experience, prevailing wage, region, continent, and so on). The goal is to learn a function that maps the attributes to the outcome, so that for a *new* application whose outcome we do not yet know, we can predict it.

This is **supervised learning**: we learn from examples where the answer is already known (the labels), in contrast to unsupervised learning where there are no labels and the goal is to find structure. It is specifically **binary classification**: the answer is one of two classes. The attributes we feed in are called **features**; the thing we predict is the **target** or **label**.

A crucial nuance that shapes everything downstream: the models do not actually output "Certified" or "Denied." They output a **probability**, a number between 0 and 1 representing the model's estimated chance that the application is Certified. The hard yes/no answer only appears when we apply a **threshold** (by default 0.5: above it we say Certified, below it Denied). Keeping the probability in mind rather than the yes/no is the key to understanding evaluation and calibration later.

---

## 2. Why decision trees, and why ensembles of them

### The decision tree

A **decision tree** is the atom of everything we are doing. It is a flowchart of yes/no questions about the features. "Is education at least a Master's? If yes, is there job experience? If yes, predict Certified with high probability." Each question splits the data into two groups; the tree keeps splitting until the groups at the bottom (the **leaves**) are reasonably pure, meaning mostly one class. To predict a new case, you drop it down the tree following the answers until it lands in a leaf, and the leaf's class mix becomes the prediction.

Trees are appealing because they capture **non-linear** relationships and **interactions** for free. An interaction is when the effect of one feature depends on another (wage matters differently for hourly versus salaried workers). A tree expresses that naturally by nesting questions. Trees are also **scale-invariant**: they only ask "is this feature above some cut point," so multiplying a feature by a constant or taking its log changes nothing. This is exactly why the wage standardization in the EDA was unnecessary for these models.

The fatal weakness of a single tree is **variance**. Variance, in this context, is the tendency of a model to change a lot if you retrain it on slightly different data. A deep tree will carve the training data into tiny, perfectly pure leaves that memorize noise. It looks brilliant on the data it was trained on and falls apart on new data. That failure mode is called **overfitting**: fitting the noise in the training set instead of the signal that generalizes.

### The bias-variance tradeoff

This is the single most important idea in the whole exercise, so it gets its own paragraph. Every model's error decomposes into three parts. **Bias** is error from the model being too simple to capture the true pattern (a straight line trying to fit a curve). **Variance** is error from the model being too sensitive to the particular training sample (the memorizing tree). **Irreducible error** is the noise inherent in the data that no model can remove. You cannot drive both bias and variance to zero at once; reducing one usually raises the other. A model that is too simple has high bias and underfits; a model that is too complex has high variance and overfits. The entire craft of modeling is finding the sweet spot, and most of the techniques below are tools for moving along this tradeoff deliberately.

### Ensembles: combining many trees

An **ensemble** is a model made of many models whose predictions are combined. The reason ensembles dominate tabular problems is that they attack the bias-variance tradeoff in ways a single model cannot. There are two main families, and they attack opposite ends.

**Bagging** (bootstrap aggregating) reduces *variance*. You train many trees, each on a different random resample of the training data, and average their predictions. Each individual tree still overfits, but they overfit to *different* noise, so when you average them the noise cancels and the stable signal survives. The **Random Forest** is bagging plus one extra twist: at each split, each tree is only allowed to consider a random subset of the features. This **decorrelates** the trees (stops them all from leaning on the same dominant feature), which makes the averaging even more effective. A random forest is, in essence, a committee of deliberately diverse overfit trees that vote.

**Boosting** reduces *bias*. Instead of training trees independently and averaging, boosting trains them in sequence, where each new tree focuses on the cases the previous trees got wrong. The first tree makes a crude prediction; the second tree learns to correct its errors; the third corrects what remains; and so on. The predictions are added up, not averaged. Because each tree is small (high bias on its own) but targeted at the current residual error, the sum becomes a low-bias, high-accuracy model. The risk is that boosting can also overfit if you let it run too long or make the trees too deep, so it needs careful regularization.

The boosting libraries in this repo are all variations on that theme:

**Gradient Boosting** is the general method, where "focus on the errors" is made precise as following the gradient (the direction of steepest error reduction) of a loss function. **XGBoost** is an optimized, regularized implementation that added second-order gradient information and built-in penalties against complexity; it was the library that made boosting dominate competitions. **LightGBM** is Microsoft's speed-focused variant that grows trees leaf-by-leaf (rather than level-by-level) and bins continuous features into histograms, making it very fast on large data. **CatBoost** is Yandex's variant whose signature feature is "ordered boosting," a scheme that reduces a subtle leakage problem in how boosting uses the target, and which handles categorical features natively. For our purposes these four are siblings: same core idea, different engineering tradeoffs. Part of the point of the benchmark was to test whether those engineering differences actually matter on *this* data, and the answer came back no: all four boosters, the random forest, and even a plain linear model landed inside overlapping confidence intervals, because the ceiling is set by the features, not the algorithm.

We also include a **logistic regression** in the benchmark, which is *not* a tree model at all. It is a linear model: it assigns each feature a weight and combines them additively, then squashes the result into a probability. It is included precisely because it is a different *family*. If it performs nearly as well as the trees, that tells us the signal in the data is mostly linear and simple; if the trees crush it, that tells us the non-linearities and interactions matter. It is the honest control group.

---

## 3. Class imbalance: when the classes are not equal

In our data, roughly 67% of applications are Certified and 33% are Denied. This is **class imbalance**: one class (here, Certified) is the **majority** and the other (Denied) is the **minority**. Imbalance is not extreme here, but it is enough to cause two problems.

The first problem is a training bias. Most learning algorithms minimize overall error, and when one class is twice as common, the cheapest way to reduce error is to favor the majority. A model can score 67% accuracy by predicting "Certified" for everyone, never once correctly flagging a denial. That model is useless for the actual business question (which denials should we worry about), yet it looks acceptable on a naive metric. This is why imbalance and metric choice are tied together.

The second problem is that the minority class, being rarer, simply gives the model fewer examples to learn its pattern from.

There are three standard remedies, and the repo deliberately chose among them.

**Oversampling** adds more minority examples. The naive version duplicates them; the sophisticated version, **SMOTE** (Synthetic Minority Over-sampling Technique), creates *new synthetic* minority points by interpolating between real ones (drawing a point somewhere on the line between a minority case and one of its near neighbors). The upside is you keep all your data; the downside is the synthetic points can be unrealistic and add noise.

**Undersampling** goes the other way: it throws away majority examples until the classes are balanced. **RandomUnderSampler**, the one used here, just drops random majority rows. The upside is simplicity and that it forces the model to take the minority seriously; the downside is you discard real data, which can raise variance.

**Class weighting** changes neither the data nor its size; it tells the algorithm to count each minority error as more costly, achieving balance through the loss function rather than the sample.

The modeling finding (recorded in `learnings.md`) was that on this data, undersampling beat SMOTE for recovering denial recall at no cost to ranking quality, so the production pipeline uses RandomUnderSampler. The deeper lesson is that more complex is not better by default; SMOTE's sophistication bought nothing here.

There is a critical methodological rule attached to resampling, which connects to the leakage discussion below: **you resample only the training data, never the test data**. The test set must keep the real-world class distribution, because that is what the model will actually face. Resampling the test set would mean grading the model on a fantasy population. In the repo this is enforced structurally by putting the resampler *inside* the modeling pipeline (more on why that matters in section 7).

---

## 4. How we measure success, and why accuracy is a trap

Because of imbalance, **accuracy** (the fraction of predictions that are correct) is misleading, as shown above. We need metrics that see the two classes separately. They all derive from the **confusion matrix**, a 2x2 table of true positives, false positives, true negatives, and false negatives. (Here "positive" is a convention for the class of interest; the code treats Denied as the business-critical class.)

From that table come two foundational metrics. **Precision** answers: of the cases the model *flagged* as denials, what fraction really were denials? It measures how much you can trust a positive prediction. **Recall** (also called sensitivity) answers: of all the cases that *truly were* denials, what fraction did the model catch? It measures how complete your net is. These two are in tension. If you flag everything as a denial, recall is perfect but precision is terrible. If you flag only the one case you are absolutely sure of, precision is high but recall is near zero. The **F1 score** is the harmonic mean of precision and recall, a single number that rewards having both.

The reason recall on the *denial* class is the business-critical metric here is domain logic: the costly mistake is approving an application that should have been denied (or failing to flag the risky ones for review). So we care most about catching denials, which is denial recall.

Now the two metrics that evaluate the *probability* rather than the thresholded yes/no:

**ROC AUC** (Area Under the Receiver Operating Characteristic Curve) is the headline metric. The ROC curve traces how the true-positive rate and false-positive rate change as you sweep the decision threshold from 0 to 1. The area under it has a beautifully concrete interpretation: it is the probability that the model assigns a higher score to a randomly chosen positive case than to a randomly chosen negative one. An AUC of 0.5 is random guessing; 1.0 is perfect ranking. The great virtue of AUC is that it is **threshold-independent**: it measures how well the model *ranks* cases, separate from where you draw the yes/no line. Our models score around 0.77 on honest held-out evaluation, meaning that given a random Denied and a random Certified application, the model ranks them in the correct order about 77% of the time.

**PR AUC** (Area Under the Precision-Recall Curve, also called average precision) is the close cousin that is often more informative under imbalance, because it focuses on the minority class and ignores the large pool of easy true negatives that can flatter ROC AUC.

**Brier score** measures something the others ignore: are the probabilities *numerically honest*. It is the mean squared error between the predicted probability and the actual 0/1 outcome. Lower is better. A model can rank perfectly (high AUC) yet output probabilities that are systematically too confident or too timid; Brier catches that, and it is the bridge to calibration in section 8.

The unifying insight, recorded as a learning, is that choosing between our models is mostly choosing an **operating point** on the precision-recall tradeoff, which you can also reach by moving the threshold on a single model. "Which model" and "which threshold" are partly the same question wearing two hats.

---

## 5. How we know the numbers are real: validation and leakage

A model's score on the data it was trained on is meaningless, because it can score well by memorizing. The whole game of honest evaluation is estimating how the model will do on data it has *never seen*.

The basic tool is the **train/test split**: hold out a random portion of the data (here 20%), train on the rest, and report the score on the held-out part. **Stratified** splitting means the split preserves the 67/33 class ratio in both halves, so neither is accidentally skewed.

A single split is noisy, because you might get lucky or unlucky in which rows landed in the test set. **Cross-validation** (CV) fixes this: split the data into k equal folds, then train k times, each time holding out a different fold as the test and training on the other k-1. Average the k scores. **K-fold cross-validation** with k=5 means five train/test cycles, each testing on a different 20%. You get a mean performance and, just as valuable, a standard deviation that tells you how stable the model is.

Here is the subtle trap that the repo's `evaluation.py` is built to avoid. If you use cross-validation to *tune* the model's settings and then report the best cross-validation score as your final number, you have cheated, slightly. By trying many settings and keeping the one that scored best on those folds, you have let information about those folds leak into your model choice. The reported number is optimistically biased.

The fix is **nested cross-validation**. There are two loops. The **inner loop** does the hyperparameter tuning (it tries different settings and picks the best). The **outer loop** evaluates the entire tuning-plus-training procedure on data the inner loop never touched. In other words, the outer loop grades not a single model but the whole *process* of building one. The number it produces is an honest estimate of how a freshly tuned model would perform on genuinely new data. This is the difference between a coursework answer and a defensible one, and it is why `run_benchmark.py` runs a 5-outer by 3-inner nested CV.

This all sits under the umbrella concept of **data leakage**: any situation where information from the evaluation data sneaks into the training process, inflating scores in a way that will not survive contact with reality. Resampling before splitting is leakage. Scaling features using statistics computed over the whole dataset is leakage. Tuning on the test set is leakage. Most catastrophic "the model worked in testing but failed in production" stories are leakage stories.

---

## 6. Hyperparameters and tuning

A **parameter** is something the model learns from data (the split points in a tree, the weights in a regression). A **hyperparameter** is a setting *you* choose before training that controls how the learning happens. For a random forest, the number of trees, the maximum depth of each tree, and the minimum samples required in a leaf are all hyperparameters.

Most important hyperparameters are knobs on the bias-variance tradeoff. `max_depth` caps how deep a tree can grow; shallow trees underfit (high bias), deep trees overfit (high variance). `min_samples_leaf` forces each leaf to contain at least so many cases, which prevents the tree from carving out tiny memorized pockets. These are **regularization** parameters: regularization is any deliberate constraint that simplifies the model to fight overfitting. The modeling finding that "tuning depth was the dominant lever" is precisely the observation that the default unbounded forest was overfitting, and that constraining depth moved it to a much better point on the tradeoff.

**Grid search** is the brute-force way to tune: you list candidate values for each hyperparameter, and the procedure tries every combination, scoring each by cross-validation, and keeps the winner. `GridSearchCV` in the code does exactly this. The grids in `models.py` are deliberately small, because the prior analysis showed that a few regularization settings capture most of the available gain, so an exhaustive search would burn compute for nothing.

One refinement the repo gets right: the inner grid search optimizes `roc_auc`, the same metric the project reports and decides on. A common quiet error is to tune on one metric (say, generic accuracy or weighted F1) and then report another. The hyperparameters that win under the tuning metric are not necessarily best for the metric you actually care about. Aligning the two is a small thing that makes the whole procedure honest.

---

## 7. Pipelines: making leakage prevention automatic

A **pipeline** chains preprocessing and modeling into a single object that behaves like one model. The repo's pipeline is `preprocess -> undersample -> classifier`. The reason this is more than tidiness is that cross-validation calls `fit` on the whole pipeline separately for each fold. That means the preprocessing (encoding, any scaling) and the undersampling are recomputed *using only that fold's training rows*, every time. The validation rows in each fold pass through transformations that were learned without them.

If you instead preprocessed and resampled once, up front, before cross-validation, every fold's "unseen" validation data would have been touched by statistics drawn from the whole set. That is the leakage from section 5, and the pipeline is the structural guarantee against it. Putting the resampler inside the pipeline is also what enforces the "resample training only" rule automatically: the resampler never sees test or validation rows because, by construction, it only runs during the fit phase on training rows.

---

## 8. Calibration: are the probabilities trustworthy

Discrimination and calibration are different virtues. **Discrimination** (what AUC measures) is whether the model *ranks* cases correctly, putting denials above certifications. **Calibration** is whether the predicted probabilities *mean what they say*: of all the cases the model says have a 70% chance of certification, do about 70% actually get certified.

These come apart constantly. A model can rank perfectly yet be systematically overconfident, outputting 0.95 for cases that are really 0.75. If anyone downstream is going to act on the probability itself (setting a review threshold, estimating expected outcomes, comparing risk across cases), miscalibrated probabilities mislead even when the ranking is excellent. Tree ensembles, and undersampled models in particular, are often poorly calibrated out of the box; undersampling distorts the base rate the model thinks it is seeing, which shifts the probabilities.

The repo checks this with a **calibration curve** (bin cases by predicted probability, plot predicted against actual; a perfectly calibrated model lies on the diagonal) and the **Brier score**. It then tries **isotonic regression**, a post-hoc method that learns a monotonic mapping to bend the model's probabilities back toward honesty, and compares Brier before and after. Calibration is fit on the training data only, for the same anti-leakage reason as everything else.

---

## 9. Feature importance and explainability

Once we have a model, we want to know *why* it predicts what it does. There are three levels of answer, in increasing sophistication.

**Impurity-based importance** (the default `feature_importances_` on tree models) measures how much each feature reduced impurity across all the splits that used it. It is free, because the model computes it as a byproduct of training, but it has a well-documented bias: it inflates the importance of high-cardinality continuous features (those with many distinct values, like a wage in dollars), because such features offer more possible split points and can carve the data more finely even when their true predictive value is modest. This is exactly the trap from `learnings.md`, where an untuned forest reported wage as overwhelmingly the most important feature, an artifact rather than a truth.

**Permutation importance** is the more honest, model-agnostic alternative the repo uses. The idea is causal and simple: take the trained model, measure its score, then randomly shuffle one feature's values (breaking its relationship to the target) and measure how much the score drops. A feature whose shuffling tanks performance was genuinely important; one whose shuffling changes nothing was not. Because it is measured on the held-out test data using the final model, it reflects real predictive contribution, not training-time artifacts. Its one caveat is that it can split credit oddly between two highly correlated features.

**SHAP values** are the most granular. Built on **Shapley values** from cooperative game theory (a principled way to fairly divide a payout among players who contributed unequally), SHAP attributes each *individual prediction* to its features. For a single application it can say "the Master's degree pushed this prediction +0.12 toward certification, the hourly wage unit pushed it -0.08." Permutation importance gives one global ranking; SHAP gives a signed, per-case explanation that also aggregates into global views. The `TreeExplainer` used in the benchmark is an exact, fast SHAP algorithm specialized for tree ensembles. It runs on the winning model only, because it is the expensive, deep-dive explanation reserved for the model we actually intend to use.

---

## 10. Quantifying uncertainty: the bootstrap

A single test-set AUC of 0.767 is a point estimate, and point estimates lie by omission: they hide how much that number would wobble if you had drawn a different test sample. The **bootstrap** is the tool for putting a margin of error on it. You repeatedly resample the test set *with replacement* (drawing the same number of rows, but allowing duplicates), recompute the metric on each resample, and look at the spread of the results. Two thousand such resamples give a distribution of plausible AUC values, and the middle 95% of that distribution is the **confidence interval**.

The practical payoff is in *comparing* models. Random forest scores 0.767 and XGBoost scores 0.759 on this repo's test set; is the forest genuinely better, or is that gap noise? Their bootstrap intervals, [0.753, 0.781] and [0.745, 0.774], overlap heavily, so the honest answer is that you cannot distinguish them on this data, and any story about why one wins is overfitting to randomness. That overlap is not incidental: every model in the benchmark, down to the linear baseline, sits inside a shared band, which is the statistical signature of a feature-set ceiling rather than a model gap. This is why `compare_models` reports the bootstrap interval, not just the number: the interval is what turns "this model won" into "this model won by more than chance," and here it did not.

---

## 11. How the whole benchmark hangs together

With the concepts in place, the flow of `scripts/run_benchmark.py` reads as a single coherent argument:

Load and prepare the data, then hold out a stratified test set that nothing will touch until the very end. Build a field of candidate models spanning two philosophies (one linear control, several tree ensembles), each wrapped in a leakage-proof pipeline that resamples only training rows. On the training data, run nested cross-validation so that the reported performance reflects the honest, tuned generalization of each *procedure*, not a lucky fit. Pick the winner by mean outer AUC. Refit everyone on the full training data and compare them on the untouched test set, with bootstrap confidence intervals so that "winner" means a real, not noise-sized, difference. Then interrogate the winner specifically: is it calibrated, what features actually drive it (permutation importance, not biased Gini), and how does it reason about individual cases (SHAP). Write all of it to `results/` as both numbers and figures.

Every choice in that chain is a defense against a specific way of fooling yourself: the held-out test against memorization, stratification against skewed splits, nested CV against tuning leakage, the pipeline against preprocessing leakage, undersampling-inside-the-fold against test contamination, bootstrap intervals against noise-as-signal, permutation importance against the Gini artifact, calibration against trusting dishonest probabilities. That accumulation of safeguards is what separates a result you can stake a decision on from a number that merely looks good.

---

## Glossary (quick reference)

- **Supervised learning**: learning a mapping from features to a known label.
- **Feature / target**: the inputs / the thing predicted.
- **Classification**: predicting a category; here, binary (two classes).
- **Decision tree**: a flowchart of feature questions ending in class predictions.
- **Overfitting / underfitting**: memorizing noise / being too simple to capture the pattern.
- **Bias / variance**: error from being too simple / error from being too sensitive to the training sample.
- **Ensemble**: a model combining many models.
- **Bagging / Random Forest**: averaging many independent trees to cut variance; the forest adds feature subsampling.
- **Boosting**: training trees in sequence, each fixing the last's errors, to cut bias.
- **Gradient Boosting / XGBoost / LightGBM / CatBoost**: the general boosting method and three engineered implementations.
- **Logistic regression**: a linear classifier; included as a non-tree control.
- **Class imbalance**: unequal class frequencies; majority vs minority.
- **SMOTE / undersampling / class weights**: three ways to counter imbalance (synthesize minority / drop majority / reweight the loss).
- **Confusion matrix**: the 2x2 table of correct and incorrect predictions per class.
- **Precision / recall / F1**: trustworthiness of positive predictions / completeness of catching positives / their harmonic mean.
- **ROC AUC**: probability the model ranks a random positive above a random negative; threshold-independent.
- **PR AUC**: precision-recall area; more informative under imbalance.
- **Brier score**: mean squared error of the predicted probabilities; lower is better.
- **Threshold / operating point**: where you draw the yes/no line on the probability.
- **Train/test split, stratification**: holding out unseen data; preserving class ratios when splitting.
- **Cross-validation**: rotating held-out folds to get a stable, averaged estimate.
- **Nested cross-validation**: an outer loop grading the entire tune-and-train procedure, an inner loop doing the tuning.
- **Data leakage**: evaluation information contaminating training, inflating scores.
- **Hyperparameter**: a setting you choose before training (vs a parameter the model learns).
- **Regularization**: deliberate constraints that simplify a model to fight overfitting.
- **Grid search**: trying every combination of candidate hyperparameters, scored by CV.
- **Pipeline**: chained preprocessing and modeling treated as one object, which makes leakage prevention automatic under CV.
- **Discrimination vs calibration**: ranking cases correctly vs probabilities meaning what they say.
- **Isotonic regression**: a post-hoc method to recalibrate probabilities.
- **Impurity (Gini) importance**: free but biased toward high-cardinality features.
- **Permutation importance**: score drop when a feature is shuffled; honest and model-agnostic.
- **SHAP / Shapley values**: fair per-prediction attribution of the output to each feature.
- **Bootstrap / confidence interval**: resampling with replacement to put a margin of error on a metric.
