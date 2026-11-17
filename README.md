# Next-Day Unhealthy Air Warning for Indian Cities

**Course:** ES 335 (pair project) | **Author:** Ayushi Agrawal, 26120002 and Pranjali Budhe, 26120006

**Live website:** https://ayushi2577.github.io/AirWarn/
**Repository:** https://github.com/ayushi2577/AirWarn

## Question

Using a city's air-quality readings up to the end of day *t*, can we predict whether the AQI on day *t+1* will be above 200 ("Poor", "Very Poor" or "Severe"), better than the baseline "tomorrow will be like today" (persistence)?

Binary classification. One row = one city on one day *t*. Predictions use only information from day *t* or earlier.

## Data

- Kaggle: "Air Quality Data in India (2015-2020)", `rohanrao/air-quality-data-in-india`, file `city_day.csv`.
- Original source: Central Pollution Control Board (CPCB), Government of India.
- 29,531 city-days, 26 cities, 2015-01-01 to 2020-07-01.
- **The raw data is not included in this repository.** Licence: CC0 (public domain) on the Kaggle page, see `LICENSE.md`.

### Getting the data

1. Log in to Kaggle and download `city_day.csv` from the dataset page, or run:
   `kaggle datasets download -d rohanrao/air-quality-data-in-india`
2. Place the file at `data/raw/city_day.csv`.
3. Never commit `kaggle.json` or the raw data (both are in `.gitignore`).

## Target and rows used

- `poor_tomorrow = 1` if the city's AQI on day *t+1* is above 200 (official CPCB boundary, not tuned). The target is built by date, not by "next row".
- A row is used only if **both today's and tomorrow's AQI are observed**. This gives **24,343 rows, 6,330 positive (26.0%)**. Every model and baseline is trained, tuned and scored on exactly these rows.

## Split (fixed before modelling, by date, never shuffled)

| Part | Dates | Rows | Positives |
|---|---|---|---|
| Train | 2015-01-01 to 2018-12-30 | 13,024 | 4,300 |
| Test (used once) | 2019-01-01 to 2019-12-30 | 6,946 | 1,567 |
| 2020 pre-lockdown | 2020-01-01 to 2020-03-23 | 1,941 | 409 |
| 2020 lockdown and unlock | 2020-03-25 to 2020-07-01 | 2,369 | 32 |

Embargo days dropped (their label falls in the next period): 2018-12-31, 2019-12-31, 2020-03-24.

**Validation folds** (forward-chaining, date-based, inside 2015-2018; each trains on all earlier dates):

| Fold | Train up to | Validate on | Positives |
|---|---|---|---|
| 1 | 2016-12-30 | 2017-01-01 to 2017-06-30 | 506 |
| 2 | 2017-06-29 | 2017-07-01 to 2017-12-31 | 533 |
| 3 | 2017-12-30 | 2018-01-01 to 2018-06-30 | 797 |
| 4 | 2018-06-29 | 2018-07-01 to 2018-12-30 | 785 |

Seen cities: 18 (have rows in 2015-2018). New cities: 8 (never in training); reported separately.

## Preprocessing

- `features.py` (learns nothing, past values only): explicit calendar, drop Xylene, forward-fill within a city for at most 3 days, "was missing" flags (PM10, NH3, Toluene, Benzene), lags (1, 2, 3, 7 days) and 3/7-day rolling means for AQI, PM2.5 and PM10, change since yesterday, month, day of week, season.
- `pipeline.py` (fitted on training rows only, inside a scikit-learn `Pipeline`): cap at the training 99.5th percentile, median imputation, standard scaling, one-hot encoding of city/season/month/day of week (`handle_unknown="ignore"`, so unseen cities get all-zero columns).
- `tests/test_no_future.py` rebuilds features after deleting all data past a cutoff and checks nothing changes.

## Methods

- Baselines: majority class, persistence (AQI > 200 today), tuned single feature (AQI > c).
- Models: logistic regression (chosen), linear SVM, exact RBF SVM (one fit took 3 seconds, so no Nystroem). Platt scaling gives SVM probabilities.
- Model-choice rule written before results: choose logistic regression unless the best SVM is better in most folds and clearly better on average. Result: logistic regression chosen.
- Threshold: highest precision with recall at least 80% on pooled validation, then frozen (`outputs/results/frozen_settings.json`). Never re-tuned on test.
- Metrics: PR-AUC (main), precision, recall, F1, ROC-AUC. Uncertainty: block bootstrap over weeks.

## Headline results (2019 test, 18 seen cities)

| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| Logistic regression | 0.940 | 0.780 | 0.958 |
| RBF SVM (Platt) | 0.941 | 0.783 | 0.959 |
| Persistence | 0.847 | 0.849 | 0.754 |
| Tuned single feature (AQI > 234, cutoff frozen on validation) | 0.914 | 0.749 | 0.934 |

- At persistence's recall (0.849), logistic regression reached precision 0.913 vs 0.847 (+6.7 points, 95% interval +5.0 to +8.6). This check was run after seeing the test and was not used to change anything.
- Recall on days where clean air turns bad is only about 10% (no weather data).
- Full tables, new-city and 2020 results are on the website and in `outputs/results/`.

## How to run

```
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Run from the project root, in this order (all scripts are in `src/`):

```
python src/explore_data.py --csv data/raw/city_day.csv   # data summary and plots (outputs/data_summary.md)
python src/data.py                  # counts check
python src/splits.py                # split and fold counts
python -m pytest tests -q           # no-future-leakage test
python src/baseline.py              # baselines on validation folds
python src/models.py                # logistic regression vs baseline
python src/run_svm.py               # SVM tuning grids
python src/compare.py               # model-choice rule, Platt scaling
python src/run_eval.py              # threshold sweep, freezes thresholds and the single-feature cutoff
python src/run_bootstrap.py         # confidence intervals
python src/run_tables.py            # persistence comparison, per-city, reliability
python src/run_regularization.py    # L1 vs L2
python src/run_ablation.py          # feature-group ablation
python src/run_errors.py            # error analysis
```

Optional EDA notebook: `pip install jupyter`, then open `notebooks/01_eda.ipynb` (outputs are already saved in it).

Final test, run **once**, after the frozen settings exist and without changing them:

```
python src/run_final_test.py
python src/run_equal_recall.py
python src/export_site_data.py      # writes docs/data/*.json for the website
```

## Repository layout

```
notebooks/  01_eda.ipynb (data exploration; run from the project root after the data is in place)
src/        code (data, splits, features, pipeline, evaluate, models, run_* scripts)
tests/      no-future-leakage test
outputs/    figures and results tables
docs/       the website (GitHub Pages serves this folder)
data/raw/   city_day.csv goes here (not committed)
```

## Contributions

We worked as a pair and shared the work equally. Both of us can explain every part.

| Area | Ayushi Agrawal | Pranjali Budhe |
|---|---|---|
| Question, data choice and proposal | Shared | Shared |
| Data checks and EDA (`explore_data.py`, notebook) | Shared | Shared |
| Features, preprocessing pipeline, leakage test | Shared | Shared |
| Baselines, models, model-choice rule | Shared | Shared |
| Evaluation design, thresholds, final test (run once) | Shared | Shared |
| Error analysis, limitations | Shared | Shared |
| Website, README and reproduction checks | Shared | Shared |

## Limitations

No weather data; mostly follows today's AQI; selection bias from requiring AQI on both days; 26 monitored cities only; dataset AQI computed by the uploader and uncapped; threshold drift (33% bad days in training vs 22.6% in 2019); weaker transfer to unseen cities; lockdown period has only 32 positives; correlated days. This project predicts; it does not explain causes, and it is not shown to work for other cities or for today.

## AI assistance

See `AI_DISCLOSURE.md`.