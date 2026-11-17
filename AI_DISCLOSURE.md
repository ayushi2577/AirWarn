# AI Tools Disclosure

**Tools used:** ChatGPT (OpenAI) and Claude (Anthropic), both through their chat interfaces.

**In one line:** ChatGPT helped us explore and compare options. Claude helped with coding, debugging and other engineering work. We made every decision, ran every script and checked every result ourselves.

## Who did what

| Area | ChatGPT | Claude | We |
|---|---|---|---|
| Exploring and comparing options (dataset candidates, model families, metrics) | Explored and listed options | | Chose |
| Question, AQI > 200 target, time split, embargo days, validation folds | | | Decided |
| Feature and preprocessing decisions | | | Decided |
| Baselines, models, model-choice rule (written before seeing results), threshold rule | | | Decided |
| Writing code: data, features, pipeline, models, evaluation, extra experiments, site data export | | Drafted and refined | Reviewed and adapted |
| No-future leakage test | | Drafted | Ran and checked |
| Website layout, design and interaction code | | Drafted and refined | Reviewed and decided what to show |
| Website text, README and this disclosure | | Drafted wording | Decided content, edited and approved |
| Debugging errors and warnings | | Investigated and suggested fixes | Tested fixes in our own environment |
| Explaining concepts (data leakage, embargo days, Platt scaling, calibration, evaluation metrics) | | Explained | Learned and can explain them |
| Running all scripts and the final test (once, with frozen settings) | | | Did |
| Checking counts and consistency across stages | | | Did |
| Reading and interpreting the results | | | Did |

## What ChatGPT did

- Helped us explore and compare options when we needed to choose: candidate datasets, model families and evaluation metrics.
- It did not write project code and did not make any decision for us.

## What Claude did

- **Code:** drafted, refined and explained code for data loading, feature engineering, the preprocessing pipeline, baselines, models, evaluation scripts, the block bootstrap, the extra experiments and the website.
- **Debugging:** helped investigate errors, import issues, warnings and compatibility problems.
- **Concepts:** explained data leakage, embargo days, Platt scaling, probability calibration and temporal validation.
- **Documentation:** helped with the README, website wording and this disclosure.

## What we did ourselves

- Chose the dataset and formulated the question and the AQI > 200 target.
- Designed the time-based split, the embargo days and the four forward-chaining validation folds.
- Decided which features, preprocessing, baselines and models to use.
- Wrote the model-choice rule before seeing the final results: choose logistic regression unless the SVM wins in most folds and is clearly better on average.
- Chose the threshold rule and froze the settings before the test year was used.
- Ran all scripts and experiments on our own machines and inspected the outputs.
- Verified the counts against our proposal: 24,343 evaluation rows, 6,330 positives and 13,024 training rows.
- Ran the automated no-future test.
- Ran the final test once, without changing anything afterwards.
- Interpreted the results and decided what the website says.

## Verification and responsibility

We did not treat any AI output as automatically correct. We read and tested the code, compared numbers across scripts, tables and the website, and checked results against the rules we set in advance. We are responsible for the design, the decisions, the results and the interpretation, and we can explain and defend every part of the work.

## External code, libraries and data

**Libraries:** pandas, NumPy, scikit-learn, Matplotlib, pytest, pyarrow, joblib, SciPy and Chart.js.

**Dataset:** CPCB air-quality data from the Kaggle dataset `rohanrao/air-quality-data-in-india`, uploaded by Rohan Rao. The raw data is not included in this repository.

No other external code was copied into the project.