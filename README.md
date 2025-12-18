# Credit Card Fraud Detection (Machine Learning Project)

## Project Overview

This project focuses on detecting fraudulent credit card transactions using supervised machine learning techniques. The dataset is highly imbalanced, with fraudulent transactions accounting for less than 1% of all records, making accuracy an unreliable evaluation metric. Instead, the project emphasizes recall, precision, F1-score, and ROC-AUC to properly assess model performance.

The goal is to build and evaluate classification models that can effectively identify fraudulent transactions while minimizing false alarms.

---

## Dataset

* **Source:** Kaggle – Credit Card Fraud Detection Dataset
* **Size:** ~285,000 transactions
* **Features:**

  * 28 anonymized PCA components (`V1`–`V28`)
  * `Amount`: transaction amount
  * `Time`: seconds elapsed between transactions
* **Target:**

  * `Class` (0 = Normal, 1 = Fraud)

 The dataset is **severely imbalanced** (<1% fraud cases).

---

## Exploratory Data Analysis (EDA)

Key insights from EDA include:

* No strong linear correlations between features and the target
* Extreme class imbalance makes accuracy misleading
* Fraud and non-fraud transactions show overlapping distributions
* Fraud transactions tend to occur across a wide range of amounts, including very small values

These observations guided metric selection and model choice.

---

## Preprocessing Steps

1. Loaded and validated the dataset (no missing values)
2. Separated features (`X`) and target (`y`)
3. Performed stratified train/test split
4. Addressed class imbalance using **class weighting**
5. Applied feature scaling **only where required** (Logistic Regression)
6. Prevented data leakage by fitting all transformations on training data only

---

## Models Trained

### Logistic Regression (Baseline)

* Used as a simple, interpretable baseline
* Required feature scaling
* Used `class_weight='balanced'` to handle imbalance

### Random Forest (Final Model)

* Captures non-linear feature interactions
* Robust to noise and imbalance
* No feature scaling required
* Used `class_weight='balanced'`

---

##Evaluation Metrics

Due to class imbalance, evaluation focused on:

* **Recall** (priority): ability to detect fraudulent transactions
* **Precision**: false alarm control
* **F1-score**: balance between precision and recall
* **ROC-AUC**: overall discriminatory ability

Accuracy was reported but not used for model selection.

---

## Results Summary

* Random Forest significantly outperformed Logistic Regression on the fraud class
* Achieved high recall and precision, indicating effective fraud detection with low false-positive rates
* Demonstrated the importance of non-linear models for this dataset

As a result, **Random Forest** was selected as the final model.

---

## Conclusion

This project demonstrates that ensemble tree-based models are well-suited for fraud detection tasks involving highly imbalanced, non-linear data. Careful preprocessing, appropriate metric selection, and imbalance-aware modeling were critical to achieving reliable results.

---

## Tools & Libraries

* Python
* pandas, numpy
* matplotlib, seaborn
* scikit-learn

---

## Future Work

* Threshold tuning to further improve recall
* Experimenting with Gradient Boosting (e.g., XGBoost)
* Cost-sensitive evaluation
* Feature importance analysis


