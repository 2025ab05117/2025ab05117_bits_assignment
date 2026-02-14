# ML Assignment 2 – Classification Models & Streamlit Deployment

## a. Problem Statement
Heart disease is one of the leading causes of death worldwide. Early detection can enable timely medical intervention and significantly improve patient survival rates.

The objective of this assignment is to implement multiple machine learning
classification models on a single dataset, evaluate their performance using
standard metrics, and deploy the models using an interactive Streamlit web
application.

## b. Dataset Description
- **Dataset Type:** Heart Disease Dataset  
- **Dataset Source:**  
  https://raw.githubusercontent.com/ageron/data/main/heart_disease.csv  
- **Number of Instances:** 1025  
- **Number of Features:** 13 (Numerical)  
- **Target Variable:** `target`

### Target Classes
- `0` → No Heart Disease  
- `1` → Presence of Heart Disease  

## c. Models Used and Evaluation Metrics

The following machine learning models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Comparison Table

|Model               | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.85 | 0.92 | 0.83 | 0.90 | 0.86 | 0.71 |
| Decision Tree       | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 |
| KNN                 | 0.92 | 0.98 | 0.91 | 0.94 | 0.92 | 0.85 |
| Naive Bayes         | 0.83 | 0.91 | 0.81 | 0.87 | 0.84 | 0.66 |
| Random Forest       | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 |
| XGBoost             | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 |

## d. Model Performance Observations

| Model | Observation |
|------|------------|
| Logistic Regression | Performed very well, indicating the dataset is largely linearly separable and well-structured. |
| Decision Tree | Achieved near-perfect accuracy, showing clear decision rules but with potential overfitting. |
| KNN | Delivered strong performance after feature scaling, proving effective for neighborhood-based classification. |
| Naive Bayes | Showed moderate performance due to the assumption of feature independence. |
| Random Forest | Provided robust and stable performance by reducing overfitting through ensemble learning. |
| XGBoost | Achieved the best overall performance by capturing complex patterns using boosting techniques. |


## e. Streamlit Deployment
The Streamlit application includes:
- CSV dataset upload option
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization

The app is deployed using Streamlit Community Cloud.
