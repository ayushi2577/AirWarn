# ML Project - — Corrected Plan (Sep 28 – Nov 17, 2026)

**Key dates:** Proposal due **Oct 15, 11:59 PM IST** (plan to submit by Oct 13). Final project due **Nov 17**. **Track:** ES 335 (website) or ES 678 (4-page LaTeX paper). Parts 0–9 are the same for both. Part 10 differs.

**Rules that apply to every part:**

- Anything learned from data (imputation, scaling, Lasso selection, Platt scaling) is fitted on **train only** (Platt scaling: on validation or CV only).
- Tune on **validation/CV**. Touch the **test set once**, at the end (Part 9).
- Results show association, **not causation**.

---

### Part 0 — Problem Setup (Sep 28–Oct 2, ~4 days)

1. Pick the final dataset (e.g. UCI Student Performance / Academic Success). Confirm you can download and use it.
2. Write what **one row** represents (e.g. one student).
3. Write **population vs dataset**: who does this generalize to, and what is the sampling bias?
4. Define target y (at-risk vs not-at-risk) and **justify the cutoff**.
5. Write **when the prediction is made** and **what information exists at that time**. Features that only exist later are not allowed.
6. List all features x and split them into groups (academic, behavioral, socioeconomic). You reuse these groups for Lasso and ablation.
7. Write one paragraph: "what does a wrong prediction cost?" (false negative vs false positive). This drives the threshold choice later.
8. Write the **limits of your claim** in 2–3 lines (what the model can and cannot say).

**Output:** short problem-statement doc (this becomes most of the proposal)

### Part 1 — Data Understanding, Preprocessing, Split (Oct 2–Oct 7, ~5 days)

1. Document the source, size, important variables, access limits, and data-quality issues.
2. Check missing values, duplicates, wrong values, and class balance (how many at-risk vs not).
3. Make basic plots: distributions of key features, relation with target.
4. **Leakage check:** remove any feature that directly or indirectly encodes the target (e.g. final grade if predicting failure) or is not known at prediction time.
5. Decide the **split before any modeling** (e.g. 70/15/15, stratified, fixed random seed). Save it to disk and reuse it everywhere.
6. Preprocessing (encoding, imputation, scaling): explain each step and **fit on train only**. Use a sklearn `Pipeline` so this cannot go wrong.

**Output:** EDA notebook section, saved split files (with seed), preprocessing pipeline

### Proposal (Oct 8–Oct 13, submit by Oct 13)

Submit these items:

1. Working title, course code, name and roll number (both students for an ES 335 pair).
2. The question and what you will predict or analyze.
3. Data source, what one row is, approximate size, and confirmation you can access it.
4. One simple baseline, your candidate methods, and the evaluation plan with a metric (e.g. recall, F1, ROC-AUC).
5. One likely challenge (e.g. class imbalance, leakage, small sample).
6. **ES 335:** the planned website interaction (e.g. threshold slider). **ES 678:** the planned depth analysis (e.g. calibration or ablation).

**Output:** proposal submitted (2-day buffer left before Oct 15)

### Part 2 — Baselines + Model 1: Logistic Regression (Oct 13–Oct 16, ~3 days)

1. **Simple baseline first:** a majority-class predictor. Record its metrics. Every later model is compared to this.
2. Derive on paper: Bernoulli likelihood → negative log-likelihood → cross-entropy loss.
3. Derive the gradient of the loss with respect to the weights.
4. Implement gradient descent by hand, then compare with sklearn's fit as a sanity check.
5. Fit on train. Get predicted **probabilities** (not only labels) on val and test.

**Output:** majority baseline + working logistic regression + derivation write-up

### Part 3 — Model 2: SVM (Oct 16–Oct 19, ~3 days)

1. Derive the soft-margin objective with slack variables.
2. One paragraph: hinge loss vs cross-entropy.
3. Fit linear SVM first, then RBF kernel. Choose C (and gamma) using **validation/CV only**. Compare the two.
4. SVM gives scores, not probabilities. Use decision scores, or Platt scaling fitted on validation/CV (needed for the calibration part).

**Output:** working SVM (linear + RBF), short write-up on hinge loss and margin geometry

### Part 4 — Model 3: KNN or Decision Tree (Oct 19–Oct 21, ~2 days)

1. Fit KNN (2–3 values of k) or a decision tree (2–3 depths). Choose using validation/CV only.
2. Note why it is useful: non-parametric, no assumption about boundary shape.
3. Get scores (KNN: neighbor fraction; tree: leaf probability) so thresholds can be swept.

**Output:** working third model

### Part 5 — Evaluation Suite (Oct 21–Oct 28, ~7 days — biggest chunk)

**Level 1 — tune each model on validation:**

1. Get probabilities/scores on the validation set.
2. Sweep thresholds (e.g. 0.05 to 0.95).
3. Plot precision and recall (and your cost) vs threshold. Add a precision-recall curve.
4. Pick each model's **own best threshold** using your Part 0 cost assumption, and justify it.

**Level 2 — final comparison (run once, in Part 9):** evaluate every model on the test set at its own chosen threshold.

Build these (first on validation; the same code is rerun on test in Part 9):

1. One metrics table for all models including the majority baseline: accuracy, precision, recall, F1, ROC-AUC.
2. Confusion matrices side by side.
3. Reliability diagram (calibration curve): logistic regression vs SVM with Platt scaling. Say which probabilities are more trustworthy.
4. Bootstrap confidence intervals for the main metrics (test sets are small, so show the uncertainty).

**Output:** evaluation notebook section with all plots, tables, and chosen thresholds (frozen before touching test)

### Part 6 — Regularization & Feature Selection (Oct 28–Oct 31, ~3 days)

1. Refit logistic regression with L2 (Ridge) and L1 (Lasso). Choose the penalty strength on validation/CV.
2. Show which features Lasso sets to zero and connect them to your feature groups.
3. One paragraph: does regularization change your model comparison?

**Output:** regularization table/plot

### Part 7 — Ablation Study (Oct 31–Nov 3, ~3 days)

1. Remove one feature group at a time (2–3 groups max). Refit the best model with the same split and settings.
2. Measure the change on **validation** (not test).
3. One paragraph: what this says about which data matters. Write it as association, not cause.

**Output:** ablation table

### Part 8 — Error Analysis & Limitations (Nov 3–Nov 6, ~3 days)

1. Look at real rows the best model got wrong (false negatives and false positives). Write what they have in common.
2. Write limitations: dataset bias, small sample, cutoff choice, threshold sensitivity, calibration issues.
3. State what you cannot claim (no causation, no proof it works on other schools).

**Output:** error-analysis and limitations section

### Part 9 — Final Test Run & Deployment Decision (Nov 6–Nov 8, ~2 days)

1. Freeze all models, thresholds, and settings. **No more changes after this.**
2. Run the test set **once** for all models at their chosen thresholds. Report the metrics table with confidence intervals.
3. Write "which model would you deploy and why": if false negatives cost more, deploy X; if interpretability matters, deploy Y. Tie it to Part 0's cost paragraph.
4. Compare against the majority baseline. If gains are small or negative, say so clearly; that is acceptable.

**Output:** final results, deployment paragraph

### Part 10 — Track Deliverable (Nov 8–Nov 15, ~7 days)

**ES 335 (website, solo or pair):**

1. Build a static site (HTML/CSS/JS) and publish on GitHub Pages.
2. Sections: question, data, preprocessing, baseline and methods, evaluation, results, errors, limitations, conclusion, reproduction steps.
3. Add at least one useful interaction, e.g. a threshold slider that updates the confusion matrix, precision, and recall using precomputed outputs.
4. Submit the live site link and the repository link.
5. If in a pair, split work clearly so each person can explain their part.

**Output:** finished website or paper

### Part 11 — Reproducibility, Disclosure & Viva Prep (Nov 13–Nov 16, ~3 days)

1. README: data access steps, split seed, preprocessing, how to run the main result.
2. Test by running everything from a clean clone.
3. Write an **AI-tools disclosure**: which tools and what kind of help. Cite external code.
4. Check that no passwords, API keys, private student data, or data you are not allowed to share is published. Use a permitted dataset or precomputed outputs.
5. Viva prep: be able to explain your contribution, key choices, evaluation, one limitation, and what happens if you change one thing (e.g. the cutoff or threshold), without notes.

**Output:** reproducible repo, disclosure note

### Buffer & Submission (Nov 16–Nov 17)

1. Final check and submit. **No new experiments.**

---

## Rubric Check

|Rubric item|Points|Covered in|
|---|---|---|
|Problem formulation and scope|10|Part 0|
|Data understanding|10|Part 1|
|Preprocessing and statistical reasoning|10|Part 1|
|Baseline and method selection|15|Parts 2, 3, 4|
|Evaluation design and metrics|20|Parts 1 (split), 5, 9|
|Results, errors, limitations|10|Parts 8, 9|
|Reproducibility and technical integrity|5|Part 11|
|Track deliverable (website/paper)|10 + 5|Part 10|
|Individual understanding / viva|5|Part 11|