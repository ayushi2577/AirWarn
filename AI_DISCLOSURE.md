# AI Tools Disclosure

**Tool used:** Claude (Anthropic), via the Claude chat interface.

## How We Used Claude

Claude was used as an assisting tool throughout the project, primarily to speed up implementation, handle repetitive coding tasks and support technical exploration.

- **Brainstorming and planning:** Discussed possible project directions and used Claude to explore implementation approaches. We evaluated the suggestions and finalized the research question, AQI > 200 prediction target, temporal split strategy and overall experimental design.
- **ML methodology and implementation:** We determined the machine learning approach, preprocessing requirements, feature construction strategy, baseline comparisons, evaluation methodology and model selection criteria. Claude assisted in translating these decisions into Python code, particularly for repetitive implementation tasks and structuring scripts.
- **Code exploration and development:** Used Claude to help draft, refine and explain code for data loading, feature engineering, preprocessing pipelines, baseline models, evaluation scripts, block bootstrap analysis and the project website. We reviewed the implementations and adapted them to the requirements of the project.
- **Debugging and technical assistance:** Used Claude to investigate implementation errors, import issues, warnings and compatibility problems. We reviewed the suggested fixes and tested them in the actual project environment.
- **ML concept clarification:** Discussed concepts such as data leakage, embargo days, Platt scaling, probability calibration and temporal validation to clarify their implications for the experimental design.
- **Documentation:** Used assistance for improving the clarity and structure of the README, website content and AI disclosure.

## What We Did Independently

We were responsible for the core research and experimental decisions throughout the project.

- Selected the dataset, formulated the prediction target (AQI > 200) and established the temporal evaluation strategy.
- Designed the training and evaluation split, including the split dates, embargo mechanism and decision to retain all four evaluation folds.
- Determined the feature engineering and preprocessing requirements based on the prediction task and leakage constraints.
- Decided which baselines and models to compare and established the model selection rule before examining the final results.
- Evaluated the suitability of logistic regression and selected it using the predefined selection criterion.
- Ran the scripts and experiments on our own machines, inspected outputs and checked consistency across different stages of the pipeline.
- Verified that the data counts matched the proposal: 24,343 evaluation rows, 6,330 positive samples and 13,024 training rows.
- Ran the automated no-future test and checked that the implementation respected the temporal constraints.
- Conducted the final evaluation once, without further tuning based on the final test results.

## Verification and Responsibility

We did not treat AI-generated code or explanations as automatically correct. We reviewed the implementations, tested the scripts, compared outputs across different components and checked the results against the project's predefined requirements.

We are responsible for the experimental design, methodological decisions, interpretation and reported results. We can explain the reasoning behind the ML pipeline, the evaluation strategy and the implementation choices.

## External Code, Libraries and Data

**Libraries and frameworks:** pandas, NumPy, scikit-learn, Matplotlib, pytest, pyarrow and Chart.js.

**Dataset:** CPCB air-quality data obtained through the Kaggle dataset `rohanrao/air-quality-data-in-india`, uploaded by Rohan Rao.

No other external code was copied into the project.