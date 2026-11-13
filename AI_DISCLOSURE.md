# AI tools disclosure

**Tool used:** Claude (Anthropic), via the Claude chat interface.

## What Claude helped with

- **Brainstorming and planning:** choosing the project topic, defining the target and the split design, and structuring the project plan.
- **Code:** drafting and explaining Python code for data loading, feature building, the preprocessing pipeline, baselines, models, evaluation scripts, the block bootstrap, and the website (HTML, CSS, JavaScript with Chart.js).
- **Debugging:** diagnosing errors and warnings (for example import errors and scikit-learn deprecation warnings).
- **Explanation:** explaining concepts such as leakage, the embargo day, Platt scaling and calibration.
- **Writing:** drafting the website text, the README and this disclosure.

## What I did myself

- Ran every script on my own machine and checked that the counts matched my proposal (24,343 evaluation rows, 6,330 positives, 13,024 training rows).
- Ran the automated no-future test and compared results across scripts for consistency.
- Made the project decisions: which dataset, the AQI > 200 target, the split dates, keeping all four folds, and choosing logistic regression using the rule written before results.
- Ran the final test once and reported the results without re-tuning.

## Checking and responsibility

I checked important outputs against each other (for example the same logistic regression scores appear in several scripts) and against the proposal numbers. I am responsible for every result and can explain any part of the work.

## External code and material

- Libraries: pandas, NumPy, scikit-learn, Matplotlib, pytest, pyarrow, Chart.js.
- Data: CPCB via the Kaggle dataset `rohanrao/air-quality-data-in-india` (uploaded by Rohan Rao).
- No other external code was copied.