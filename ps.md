
ILPD is small, and I don't want to hide that. It has one CSV of 583 patients, so it is **not a large dataset**. I'd still use it, and the last section explains how to get enough out of it.

### What it is (all from the official UCI page)

- The records come from the northeast of Andhra Pradesh, India, and the task is to predict whether a patient has liver disease from biochemical markers. [uci](https://archive-beta.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)
- There are 416 patients with liver disease and 167 without, stored in a label column called "Selector". That is about 71% vs 29%. [uci](https://archive-beta.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)
- Each patient has 10 variables: age, gender, total bilirubin, direct bilirubin, total proteins, albumin, A/G ratio, SGPT, SGOT and Alkphos. [uci](https://archive-beta.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)
- Of the 583 records, 441 are male and 142 are female. [uci](https://archive-beta.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)
- Ages above 89 are stored as 90. [uci](https://archive-beta.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)
- The license is CC BY 4.0, so you can use it freely, but you must cite it. Citation: Ramana & Venkateswarlu, UCI ML Repository, doi 10.24432/C5D02C.

### Where to get it

- **UCI (main source):** archive.ics.uci.edu/dataset/225. The CSV is about 23 KB.
- **Python (easiest):** `pip install ucimlrepo`, then `fetch_ucirepo(id=225)`. It gives you X and y as pandas tables.
- **Kaggle:** search "Indian Liver Patient Records" (same data).
- **OpenML:** dataset id 1480 (same data).

For reproducibility, download once, save the CSV in your repo, and note the source and date in the README.

### How it works for your project

- **One row:** one patient.
- **Target:** liver disease (yes) vs not (no). Recode it to 1 = disease, 0 = no disease, because the raw file uses 1 and 2.
- **Prediction time:** when a routine blood panel comes back, before a specialist confirms anything. All 10 features are available at that time, so leakage is low, but you must still say this explicitly.
- **Feature groups** (for your Lasso and ablation):
    - bilirubin: total, direct
    - liver enzymes: SGPT, SGOT, Alkphos
    - proteins: total proteins, albumin, A/G ratio
    - demographics: age, gender
- **Cost paragraph:** a false negative (a sick patient sent home) is far worse than a false positive (an extra test), so you tune the threshold toward high recall.
- **Baseline:** always predicting "disease" already gives about 71% accuracy, so accuracy alone means little. This is exactly what the rubric wants you to notice.
- **Data checks to do:** the Kaggle copy has a few blank A/G ratio values, and I believe it has some duplicate rows. Both are from memory, so check them in your EDA.
- **Good extra angle:** the dataset has been used to study gender-based differences in liver-disease prediction, since biochemical markers don't work equally well for men and women. Since only about 24% of rows are female, checking your model's errors by gender would make a strong error-analysis or ES 678 depth section. [uci](https://archive-beta.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)

### Making the small size work

- **Enough for your models:** you have 10 features, and even the smaller class has 167 patients, which is fine for logistic regression and SVM.
- **Change the split:** a 70/15/15 split leaves you a validation set of about 88 patients, which is too noisy for threshold tuning. Use **80% train+validation, 20% test** (about 117 test patients, roughly 83 with disease and 34 without). Tune with **5-fold stratified cross-validation** on the 80%. Fix the seed and save the split.
- **Show the uncertainty:** with about 34 healthy patients in the test set, one or two mistakes move your specificity a lot. Bootstrap confidence intervals will show this, and reporting wide intervals honestly earns credit.
- **Don't oversample before splitting.** If you use class weights or SMOTE, apply them inside the training folds only, never on the test data.

If you want a bigger dataset than this, the honest answer is that no Indian medical dataset I know of is both large and instantly accessible. NFHS is large but needs access approval, and CKD is smaller than ILPD.

Want me to write your Part 0 problem statement for ILPD next, and update the plan file with the 80/20 split and the 2-model change?