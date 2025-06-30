# Credit Card Fraud Detection

## Project Overview

This project focuses on building and evaluating machine learning models to detect fraudulent credit card transactions. By leveraging a comprehensive dataset of credit card transactions, the goal is to identify patterns indicative of fraud, thereby minimizing financial losses and enhancing security for financial institutions and customers.

## Dataset

The dataset used in this project is `CreditCard.csv`, which contains a large number of credit card transactions.

* **Shape**: (568,630 rows, 31 columns)
* **Memory Usage**: 134.49 MB
* **Missing Values**: None
* **Duplicate Rows**: None
* **Data Types**: 29 columns are `float64`, and 2 columns are `int64`.

### Class Distribution

The dataset exhibits a perfectly balanced class distribution, with an equal number of non-fraudulent and fraudulent transactions.

* **Non-Fraud (0)**: 284,315 (50.00%)
* **Fraud (1)**: 284,315 (50.00%)

## Exploratory Data Analysis (EDA)

EDA was performed to understand the characteristics of the dataset, identify important features, and analyze transaction patterns.

### Transaction Amount Distribution

The distribution of transaction amounts for both non-fraudulent and fraudulent transactions shows distinct patterns.

![Transaction Amount Distribution](https://github.com/nsk20/Fraud-Detection/blob/e5dbb4ec0611a8a7ed4d19ed8a945853fdc22900/Visualization/1.jpg)

* **Non-Fraudulent Transactions**: Mean amount is 12026.31, median is 11996.90, and standard deviation is 6929.50.
* **Fraudulent Transactions**: Mean amount is 12057.60, median is 12062.45, and standard deviation is 6909.75.

### Feature Importance Analysis

A preliminary Random Forest Classifier was used to identify the most important features for fraud detection. The top 8 most important features are: 'V14', 'V10', 'V12', 'V4', 'V11', 'V17', 'V16', and 'V7'.


### Correlation Heatmap of Top Features

A correlation heatmap was generated for the top features to visualize their relationships, including their correlation with the 'Class' variable.

### Box Plots of Top Features by Class

Box plots for the top 4 most important features (V14, V10, V12, V4) show their distribution across non-fraudulent and fraudulent classes, highlighting their discriminative power.

### Amount vs. Most Important Feature by Class

A scatter plot illustrating the relationship between the transaction `Amount` and the most important feature ('V14') for both non-fraudulent and fraudulent transactions.


## Merchant Category Analysis

[cite_start]Transactions were categorized into 5 bins (Cat_A to Cat_E) based on 'V10' (the second most important feature) to analyze fraud rates across different merchant categories. [cite: 9]

| Merchant_Category | Total_Transactions | Fraud_Count | Fraud_Rate |
|-------------------|--------------------|-------------|------------|
| Cat_A             | 566921             | 284315      | 0.5015     |
| Cat_B             | 1559               | 0           | 0.0000     |
| Cat_C             | 149                | 0           | 0.0000     |
| Cat_D             | 0                  | 0           | NaN        |
| Cat_E             | 1                  | 0           | 0.0000     | 

* **Safest Category**: Cat_B, Cat_C, Cat_E (0.0% fraud rate)
* **Riskiest Category**: Cat_A (50.1% fraud rate)

## Transaction Type Analysis

[cite_start]Transactions were categorized by amount into 'Small' (0-100), 'Medium' (100-500), 'Large' (500-2000), and 'Very_Large' (>2000) to understand fraud patterns across different transaction sizes.

| Transaction_Type | Total_Transactions | Fraud_Count | Fraud_Rate |
|------------------|--------------------|-------------|------------|
| Small            | 1190               | 594         | 0.4992     |
| Medium           | 9599               | 4719        | 0.4916     |
| Large            | 34997              | 17298       | 0.4943     |
| Very_Large       | 522844             | 261704      | 0.5005     |

* **Safest Transaction Type**: Medium (49.2% fraud rate)
* **Riskiest Transaction Type**: Very_Large (50.1% fraud rate)

## Data Preprocessing

For model training, only the top 8 most important features were selected: 'V14', 'V10', 'V12', 'V4', 'V11', 'V17', 'V16', and 'V7'. Features were scaled using `StandardScaler`. The dataset was split into training and testing sets (80% train, 20% test) with stratification to maintain class balance.

## Machine Learning Models

The following models were trained and evaluated for fraud detection:

* **Random Forest Classifier**
* **Gradient Boosting Classifier**
* **Decision Tree Classifier**
* **Gaussian Naive Bayes**

### Model Performance Metrics

The models were evaluated based on Accuracy, AUC Score, Precision, Recall, and F1-score.

#### Random Forest
* **Accuracy**: 0.9783
* **AUC Score**: 0.9980
* **Precision (Fraud)**: 1.00
* **Recall (Fraud)**: 0.96
* **Specificity (Non-Fraud Accuracy)**: 0.996

#### Gradient Boosting
* **Accuracy**: 0.9908
* **AUC Score**: 0.9994
* **Precision (Fraud)**: 0.99
* **Recall (Fraud)**: 0.99
* **Specificity (Non-Fraud Accuracy)**: 0.993

#### Decision Tree
* **Accuracy**: 0.9779
* **AUC Score**: 0.9965
* **Precision (Fraud)**: 0.98
* **Recall (Fraud)**: 0.98
* **Specificity (Non-Fraud Accuracy)**: 0.977

#### Naive Bayes
* **Accuracy**: 0.93
* **AUC Score**: 0.9850
* **Precision (Fraud)**: 0.99
* **Recall (Fraud)**: 0.88
* **Specificity (Non-Fraud Accuracy)**: 0.989

## Summary & Insights

1.  **Fraud Detection Model Performance**:
    * **Best Model**: Gradient Boosting (AUC: 0.999, Accuracy: 99.1%) 
    * **All Models**:
        * Gradient Boosting: 99.1% accuracy, 0.999 AUC 
        * Random Forest: 97.8% accuracy, 0.998 AUC 
        * Decision Tree: 97.8% accuracy, 0.996 AUC 
        * Naive Bayes: 93.0% accuracy, 0.930 AUC

2.  **Merchant Category Analysis**:
    * Safest Category: Cat_B, Cat_C, Cat_E (0.0% fraud rate)
    * Riskiest Category: Cat_A (50.1% fraud rate)

3.  **Transaction Type Analysis**:
    * Safest Transaction: Medium (49.2% fraud rate)
    * Riskiest Transaction: Very_Large (50.1% fraud rate)

