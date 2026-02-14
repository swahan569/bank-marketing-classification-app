# Bank Marketing Classification

## Problem Statement
The objective of this project is to build multiple classification models to predict whether a client will subscribe to a term deposit based on bank marketing campaign data.

## Dataset Description
The dataset used is the Bank Marketing Dataset containing 45,211 instances and 16 input features.
It includes both numerical and categorical variables such as age, job, marital status, education, balance, contact type, and campaign details.
The target variable 'y' indicates whether the client subscribed to a term deposit (yes/no).

## Models Used
The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

## Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8457 | 0.9079 | 0.4182 | 0.8147 | 0.5527 | 0.5092 |
| Decision Tree | 0.8285 | 0.8712 | 0.3917 | 0.8431 | 0.5349 | 0.4959 |
| kNN | 0.8996 | 0.8513 | 0.6298 | 0.3440 | 0.4450 | 0.4169 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.8742 | 0.9261 | 0.4772 | 0.7911 | 0.5953 | 0.5497 |
| XGBoost | 0.8829 | 0.9305 | 0.4997 | 0.8195 | 0.6208 | 0.5802 |

## Observations

- Logistic Regression achieved strong AUC and recall, making it effective for imbalanced classification.
- Decision Tree captured non-linear relationships but showed slightly lower generalization.
- kNN showed high accuracy but poor recall due to bias toward the majority class.
- Naive Bayes provided a fast baseline with moderate performance.
- Random Forest improved stability and delivered strong AUC and MCC.
- XGBoost achieved the best overall performance across most evaluation metrics.