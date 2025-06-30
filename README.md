# Credit Card Fraud Detection

## Project Overview

This project focuses on building and evaluating machine learning models to detect fraudulent credit card transactions. By leveraging a comprehensive dataset of credit card transactions, the goal is to identify patterns indicative of fraud, thereby minimizing financial losses and enhancing security for financial institutions and customers.

## Dataset

The dataset used in this project is `CreditCard.csv`, which contains a large number of credit card transactions.

* [cite_start]**Shape**: (568,630 rows, 31 columns) [cite: 1]
* [cite_start]**Memory Usage**: 134.49 MB [cite: 1]
* [cite_start]**Missing Values**: None [cite: 2]
* [cite_start]**Duplicate Rows**: None [cite: 2]
* [cite_start]**Data Types**: 29 columns are `float64`, and 2 columns are `int64`[cite: 2].

### Class Distribution

[cite_start]The dataset exhibits a perfectly balanced class distribution, with an equal number of non-fraudulent and fraudulent transactions. [cite: 3, 4]

* [cite_start]**Non-Fraud (0)**: 284,315 (50.00%) [cite: 2]
* [cite_start]**Fraud (1)**: 284,315 (50.00%) [cite: 4]

## Exploratory Data Analysis (EDA)

EDA was performed to understand the characteristics of the dataset, identify important features, and analyze transaction patterns.

### Transaction Amount Distribution

The distribution of transaction amounts for both non-fraudulent and fraudulent transactions shows distinct patterns.

![Transaction Amount Distribution](Images/transaction_amount_distribution.png)

* [cite_start]**Non-Fraudulent Transactions**: Mean amount is 12026.31, median is 11996.90, and standard deviation is 6929.50. [cite: 7]
* [cite_start]**Fraudulent Transactions**: Mean amount is 12057.60, median is 12062.45, and standard deviation is 6909.75. [cite: 7]

### Feature Importance Analysis

A preliminary Random Forest Classifier was used to identify the most important features for fraud detection. [cite_start]The top 8 most important features are: 'V14', 'V10', 'V12', 'V4', 'V11', 'V17', 'V16', and 'V7'. [cite: 7]

![Top Features Correlation Matrix](Images/top_features_correlation_matrix.png)

### Correlation Heatmap of Top Features

A correlation heatmap was generated for the top features to visualize their relationships, including their correlation with the 'Class' variable.

### Box Plots of Top Features by Class

Box plots for the top 4 most important features (V14, V10, V12, V4) show their distribution across non-fraudulent and fraudulent classes, highlighting their discriminative power.

![V14 by Class (Rank #1)](Images/v14_boxplot.png)
![V10 by Class (Rank #2)](Images/v10_boxplot.png)

### Amount vs. Most Important Feature by Class

A scatter plot illustrating the relationship between the transaction `Amount` and the most important feature ('V14') for both non-fraudulent and fraudulent transactions.

![Amount vs V14 by Class](Images/amount_vs_v14_scatter.png)

## Merchant Category Analysis

[cite_start]Transactions were categorized into 5 bins (Cat_A to Cat_E) based on 'V10' (the second most important feature) to analyze fraud rates across different merchant categories. [cite: 9]

| Merchant_Category | Total_Transactions | Fraud_Count | Fraud_Rate |
|-------------------|--------------------|-------------|------------|
| Cat_A             | 566921             | 284315      | 0.5015     |
| Cat_B             | 1559               | 0           | 0.0000     |
| Cat_C             | 149                | 0           | 0.0000     |
| Cat_D             | 0                  | 0           | NaN        |
| Cat_E             | 1                  | 0           | [cite_start]0.0000     | [cite: 10]

* [cite_start]**Safest Category**: Cat_B, Cat_C, Cat_E (0.0% fraud rate) [cite: 10, 39]
* [cite_start]**Riskiest Category**: Cat_A (50.1% fraud rate) [cite: 10, 39]

## Transaction Type Analysis

[cite_start]Transactions were categorized by amount into 'Small' (0-100), 'Medium' (100-500), 'Large' (500-2000), and 'Very_Large' (>2000) to understand fraud patterns across different transaction sizes. [cite: 11]

| Transaction_Type | Total_Transactions | Fraud_Count | Fraud_Rate |
|------------------|--------------------|-------------|------------|
| Small            | 1190               | 594         | 0.4992     |
| Medium           | 9599               | 4719        | 0.4916     |
| Large            | 34997              | 17298       | 0.4943     |
| Very_Large       | 522844             | 261704      | [cite_start]0.5005     | [cite: 12]

* [cite_start]**Safest Transaction Type**: Medium (49.2% fraud rate) [cite: 12, 40]
* [cite_start]**Riskiest Transaction Type**: Very_Large (50.1% fraud rate) [cite: 12, 40]

## Data Preprocessing

[cite_start]For model training, only the top 8 most important features were selected: 'V14', 'V10', 'V12', 'V4', 'V11', 'V17', 'V16', and 'V7'. [cite: 13, 14] Features were scaled using `StandardScaler`. [cite_start]The dataset was split into training and testing sets (80% train, 20% test) with stratification to maintain class balance. [cite: 14]

## Machine Learning Models

The following models were trained and evaluated for fraud detection:

* **Random Forest Classifier**
* **Gradient Boosting Classifier**
* **Decision Tree Classifier**
* **Gaussian Naive Bayes**

### Model Performance Metrics

The models were evaluated based on Accuracy, AUC Score, Precision, Recall, and F1-score.

#### Random Forest
* [cite_start]**Accuracy**: 0.9783 [cite: 17]
* [cite_start]**AUC Score**: 0.9980 [cite: 17]
* [cite_start]**Precision (Fraud)**: 1.00 [cite: 18]
* [cite_start]**Recall (Fraud)**: 0.96 [cite: 18]
* [cite_start]**Specificity (Non-Fraud Accuracy)**: 0.996 [cite: 30]

#### Gradient Boosting
* [cite_start]**Accuracy**: 0.9908 [cite: 19]
* [cite_start]**AUC Score**: 0.9994 [cite: 19]
* [cite_start]**Precision (Fraud)**: 0.99 [cite: 20]
* [cite_start]**Recall (Fraud)**: 0.99 [cite: 20]
* [cite_start]**Specificity (Non-Fraud Accuracy)**: 0.993 [cite: 30]

#### Decision Tree
* [cite_start]**Accuracy**: 0.9779 [cite: 21]
* [cite_start]**AUC Score**: 0.9965 [cite: 21]
* [cite_start]**Precision (Fraud)**: 0.98 [cite: 22]
* [cite_start]**Recall (Fraud)**: 0.98 [cite: 22]
* [cite_start]**Specificity (Non-Fraud Accuracy)**: 0.977 [cite: 31]

#### Naive Bayes
* [cite_start]**Accuracy**: 0.93 [cite: 25]
* **AUC Score**: Not explicitly stated in the provided text, but it is the lowest from the ROC curves.
* [cite_start]**Precision (Fraud)**: 0.99 [cite: 25]
* [cite_start]**Recall (Fraud)**: 0.88 [cite: 25]
* [cite_start]**Specificity (Non-Fraud Accuracy)**: 0.989 [cite: 31]

## Detailed Confusion Matrices

Confusion matrices provide a detailed breakdown of true positives, true negatives, false positives, and false negatives for each model.

![Confusion Matrices](Images/confusion_matrices.png)

## Model Evaluation & Comparison

### Feature Importance Comparison (Tree-based Models)

![Random Forest Top 10 Features](Images/rf_feature_importance.png)
![Gradient Boosting Top 10 Features](Images/gb_feature_importance.png)

### Model Performance Comparison (Accuracy and AUC Score)

![Model Performance Comparison](Images/model_performance_comparison.png)

### ROC Curves Comparison

![ROC Curves Comparison](Images/roc_curves_comparison.png)

### Model Ranking by AUC Score

![Model Ranking by AUC Score](Images/model_ranking_auc.png)

### Performance Metrics Heatmap

![Performance Metrics Heatmap](Images/performance_metrics_heatmap.png)

## Summary & Insights

1.  **Fraud Detection Model Performance**:
    * [cite_start]**Best Model**: Gradient Boosting (AUC: 0.999, Accuracy: 99.1%) [cite: 39]
    * **All Models**:
        * [cite_start]Gradient Boosting: 99.1% accuracy, 0.999 AUC [cite: 39]
        * [cite_start]Random Forest: 97.8% accuracy, 0.998 AUC [cite: 39]
        * [cite_start]Decision Tree: 97.8% accuracy, 0.996 AUC [cite: 39]
        * [cite_start]Naive Bayes: 93.0% accuracy, 0.930 AUC [cite: 25, 39]

2.  **Merchant Category Analysis**:
    * [cite_start]Safest Category: Cat_B, Cat_C, Cat_E (0.0% fraud rate) [cite: 10, 39]
    * [cite_start]Riskiest Category: Cat_A (50.1% fraud rate) [cite: 10, 39]

3.  **Transaction Type Analysis**:
    * [cite_start]Safest Transaction: Medium (49.2% fraud rate) [cite: 12, 40]
    * [cite_start]Riskiest Transaction: Very_Large (50.1% fraud rate) [cite: 12, 40]

## Setup and Installation

To run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/fraud-detection.git](https://github.com/yourusername/fraud-detection.git)
    cd fraud-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

4.  **Place the dataset:**
    Ensure `CreditCard.csv` is in the root directory of the project.

5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Fraud Detection.ipynb"
    ```
    Alternatively, you can open and run the notebook in Google Colab (as indicated by the original source).
