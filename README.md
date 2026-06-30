# Explainable Credit Card Fraud Detection with Machine Learning and SHAP

## Overview

This project builds an end-to-end machine learning pipeline for detecting fraudulent credit card transactions in a highly imbalanced financial dataset. It combines exploratory data analysis, feature engineering, imbalance-aware model training, threshold-aware evaluation, SHAP explainability, and an offline rule-based explanation layer (designed to be LLM-ready) that translates model outputs into readable, analyst-facing fraud investigation summaries.

The scope is intentional: the project is fully notebook-based, offline, and reproducible. Deployment artifacts (app, API, container, RAG) are documented as deliberate future work rather than left half-finished, so the repository reflects exactly what was built.

## Why This Project Matters

Financial fraud detection is a high-stakes classification problem where fraudulent transactions are rare but costly. A model can reach very high accuracy by mostly predicting the majority class while still missing the cases that matter most.

This project focuses on the questions that actually matter in fraud work:

- How many fraudulent transactions does the model catch?
- How reliable are the alerts it raises?
- Which features drive each decision?
- How can model outputs be packaged into analyst-friendly explanations?

## Dataset

The project uses the ULB credit card fraud dataset: **284,807 real transactions** made by European cardholders over two days in September 2013, with only **~0.17% labelled as fraud** -- a realistic, severe class imbalance.

Key columns:

- `V1`-`V28`: anonymized PCA transaction-behavior components
- `Time`: transaction timestamp relative to the dataset timeline
- `Amount`: transaction amount
- `Class`: target label -- `0` = legitimate, `1` = fraud

Data files are intentionally git-ignored and are not pushed to the repository.

## Repository Structure

```text
.
|-- EDA.ipynb
|-- 02_modeling.ipynb
|-- 02_modeling_executed.ipynb
|-- 03_explainability_shap.ipynb
|-- 03_explainability_shap_executed.ipynb
|-- 04_rule_based_explanation_layer.ipynb
|-- 04_rule_based_explanation_layer_executed.ipynb
|-- outputs/
|-- data/                (git-ignored)
|-- .gitignore
`-- README.md
```

### Completed Stages

| Stage | Notebook | Status |
|---|---|---|
| Exploratory data analysis and cleaning | `EDA.ipynb` | Complete |
| Modeling and evaluation | `02_modeling.ipynb`, `02_modeling_executed.ipynb` | Complete |
| SHAP explainability | `03_explainability_shap.ipynb`, `03_explainability_shap_executed.ipynb` | Complete |
| Rule-based explanation layer | `04_rule_based_explanation_layer.ipynb`, `04_rule_based_explanation_layer_executed.ipynb` | Complete |

## Feature Engineering

Engineered features were added to improve transaction-pattern detection:

- `LogAmount`: log-transformed transaction amount
- `IsZeroAmount`: indicator for zero-amount transactions
- `Day`: derived time-based feature
- `HourSin` / `HourCos`: cyclical hour encoding via sine and cosine transforms
- `ExtremeFlag` features: tested during experimentation

The final best-performing model did not use the `ExtremeFlag` features.

## Modeling Approach

The workflow follows fraud-detection best practice for imbalanced classification:

- Stratified train-test split
- Test set kept fully imbalanced and untouched
- SMOTE applied only to the training data (never to test) to avoid leakage
- Models compared with and without the engineered `ExtremeFlag` features
- Threshold behavior assessed on a validation split carved from training data, not on the test set

Models tested: Logistic Regression, Random Forest, and HistGradientBoostingClassifier.

Evaluation metrics: Precision, Recall, F1-score, ROC-AUC, and PR-AUC / Average Precision.

## Best Model Results

The best-performing model was Random Forest (Version B, without `ExtremeFlag` features), selected by PR-AUC.

| Model | Version | Threshold | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | Without ExtremeFlags | 0.50 | 0.924 | 0.768 | 0.839 | 0.986 | 0.816 |

### What these numbers mean for fraud operations

In practical terms, at the 0.50 threshold the model catches roughly **77% of fraudulent transactions** (recall 0.768), and about **92% of the transactions it flags are genuinely fraudulent** (precision 0.924) -- so only ~8% of alerts are false alarms, while ~23% of fraud still slips through. That trade-off is exactly what the threshold is for: lowering it catches more fraud at the cost of more false positives, and the right operating point depends on investigation capacity, false-positive cost, and fraud-loss exposure.

## Why Accuracy Is Not the Main Metric

With fraud at ~0.17% of transactions, a model could score over 99% accuracy by predicting "legitimate" almost every time while catching no fraud at all. This project therefore prioritizes **PR-AUC, recall, and F1** over accuracy, because they reflect performance on the rare class that actually matters.

## SHAP Explainability

SHAP was used to produce model-grounded explanations for individual fraud predictions, identifying which features pushed each score up or down.

Top SHAP features by mean absolute contribution:

1. `V14`
2. `V12`
3. `V4`
4. `V3`
5. `V10`
6. `V11`
7. `V17`
8. `V16`
9. `V7`
10. `V1`

These are anonymized PCA components and are **not** assigned invented real-world meanings. SHAP values are used here as model-grounded signals, not as causal claims.

## Rule-Based Fraud Explanation Layer (LLM-Ready Design)

The project includes an offline, rule-based prototype that converts model outputs and SHAP drivers into human-readable fraud investigation summaries.

This layer:

- Uses standard Python only -- no OpenAI API, paid API, or live LLM service
- Converts model predictions and SHAP contributions into structured narratives
- Produces a risk summary, key model drivers, a suggested investigation action, and a confidence caveat

It is deliberately built around a strict grounding contract (every statement tied to the model score and SHAP table for that transaction), so a live LLM could later be dropped in to replace the rule-based generator without changing the governance design.

## Key Takeaways

- Fraud detection should be evaluated with imbalance-aware metrics, not accuracy alone.
- Keeping the test set imbalanced and untouched gives a realistic estimate of model behavior.
- SMOTE belongs on the training data only, to avoid leakage.
- Random Forest performed best among the tested models on PR-AUC.
- SHAP adds transparency, but anonymized PCA features should not be over-interpreted.
- A lightweight, grounded explanation layer makes model results usable in an investigation workflow.

## Technologies Used

Python, Jupyter, pandas, NumPy, scikit-learn, imbalanced-learn, SHAP, Matplotlib, Seaborn.

## How to Run

1. Clone the repository.

```bash
   git clone https://github.com/nazishatta/Explainable-Credit-Card-Fraud-Detection-with-Machine-Learning-and-SHAP.git
   cd Explainable-Credit-Card-Fraud-Detection-with-Machine-Learning-and-SHAP
```

2. Place the credit card fraud dataset in a local `data/` directory (data files are git-ignored).

3. Install dependencies.

```bash
   pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn jupyter
```

4. Run the notebooks in order.

```text
   EDA.ipynb
   02_modeling.ipynb
   03_explainability_shap.ipynb
   04_rule_based_explanation_layer.ipynb
```

Executed notebook versions are included for review.

## Future Work (Deliberately Out of Current Scope)

Documented as next steps, not claimed as completed:

- A live LLM replacing the rule-based generator, using the same grounding contract
- Interactive dashboard / API / containerized deployment
- Hyperparameter tuning and cross-validated metric stability
- Model monitoring and drift detection
- Cost-based threshold selection driven by business assumptions

## Author

**Nazish Atta**
MS Data Science, George Washington University
