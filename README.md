# Insurance Claims Classification and Value Prediction ML

## Background

Car insurance claims dataset taken from this source: `https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data`. We use this dataset containing driving history, demographic data, and past claims data to build a classification model to predict whether a claim will be made by a driver or not. We then build a regression model on the subset of data where a claim was made, to predict the value of claims. 

# Solution:

This problem is split into two parts. <br> 
1. The first part uses the entire dataset and tackles the binary classification, with the target variable `0` if a claim is not made and `1` if a claim is made. <br>
2. The second part uses only those entries where the value of claims is > 0, or where the target variable `is_claim=1`. Then, with the `claims_value` column as the target variable, a regression model is built. 

The repository contains a solution in the `python notebook`, where multiple ML classifiers and regressors are trained on a training dataset and the desired accuracy and RMSE is achieved. 
Two chosen models with the best performance are submitted as solutions. 

## Training and Evaluation: 
The preprocessing steps were: <br>
1. Data cleaning, removing duplicates. <br>
2. Data transformations for categorical data. <br>
3. Data imputation for null entries. <br>
4. Handling imbalance by over/undersampling. <br>
5. Model building. <br>
From an initial comparison, boosting models such as the  `XGBoost Classifier` and `GradientBoost Classifier` were performing better.  Model performance was measured by `accuracy`, `precision`, `recall`, and `f1-score`. To evaluate the best model, hyperparameter optimization was performed using `Random Search`  and `Grid Search` algorithms.
To boost model robustness, and avoid overfitting, `k-fold cross validation` and regularization methods such as `L1 regularization` were employed. 

# Conclusion:

For the given dataset, I found the best performance model to be an `XGBoost Classifier`. It gives an accuracy of 78%. However, the performance for class 1 was still very low, and can be further improved by smart feature selection. <br> 
For the regression problem, the `XGBoost Regressor` performed best, giving the lowest RMSE. 



