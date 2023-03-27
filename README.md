# Credit Card Fraud Detection using Machine Learning 💳⛔

This project aims to build machine learning models that can detect fraudulent transactions in credit card data. The dataset used in this project contains transactions made by European cardholders in September 2013, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

## Requirements 

To run the code in this project, you will need to have the following Python libraries installed:

- pandas
- seaborn
- matplotlib
- numpy
- scikit-learn
- xgboost

You can install these libraries using pip (each module at a time):

```
pip install pandas/seaborn/matplotlib/numpy/scikit-learn/xgboost
```

## Usage

To use this project, follow these steps:

1. Clone this repository to your local machine.
2. Download the dataset from Kaggle [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3. Open the `credit_card_fraud_detection.ipynb` file in Jupyter Notebook or JupyterLab.
4. Run each cell in the notebook to see how the different machine learning models perform on the credit card data.
5. Modify the code as needed to experiment with different models or parameters.

## Dataset 💾

The credit card fraud dataset used in this project contains only numerical input variables which are the result of a PCA transformation and can be downloaded [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Unfortunately, due to confidentiality issues, it was not possible provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, while 'Time' and 'Amount' are not transformed with PCA. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction amount.

## Models 🤖

This project uses several machine learning models to detect fraudulent transactions, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

Given the class imbalance ratio, it is recommended to measurie the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

## Results 📊

After analyzing each area under the curve (AUC) of Precision vs. Recall plot for each trained model, it was found that XGBoostClassifier was the best model trained, followed by RandomForest and LogisticRegression (balanced classes) models.

Overall, this project demonstrates that machine learning models can be effective in detecting credit card fraud, even in highly unbalanced datasets. The XGBoostClassifier model performed the best in this particular dataset, but other models such as Random Forest and Logistic Regression can also be effective depending on the specific use case. It is important to note that measuring accuracy using AUPRC is more meaningful for unbalanced classification than using confusion matrix accuracy.

The main results for each model considering the business needs are shown below.

|                                | LogReg | LogReg_balanced | DecisionTree | RandomForest | XGBoostClass | XGBoostRF |
|--------------------------------|--------|----------------|--------------|--------------|--------------|-----------|
| Fraudulent Caught losses prevented ($) | 7562.79 | 13165.65 | 10796.3 | 11077.85 | 12345.7 | 11141.69 |
| Fraudulent not Caught losses ($) | 8515.61 | 2912.75 | 5282.1 | 5000.55 | 3732.7 | 4936.71 |
| Transactions labeled as fraud (Qty.) | 96.00 | 2155.00 | 101.0 | 75.00 | 85.0 | 255.00 |
| Net Revenue Fraud detection model ($) | -1048.82 | 8097.90 | 5413.2 | 6002.30 | 8528.0 | 5949.98 |
|

