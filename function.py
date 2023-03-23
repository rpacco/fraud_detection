import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_importances(model):
    """
    Plots the top 10 feature importances of a trained model as a horizontal bar chart using Seaborn.

    Parameters:
    -----------
    model : object
        A trained machine learning model.

    Returns:
    --------
    None
    """
    # Get feature importances from the model
    importances = pd.Series(model.feature_importances_, model.feature_names_in_)
    # Sort the values in descending order
    importances = importances.sort_values(ascending=False)
    # Get the top 10 feature importances and multiply by 100 for percentage
    top_importances = importances.head(10) * 100
    # Create a horizontal bar chart of the top 10 feature importances using Seaborn
    sns.barplot(x=top_importances, y=top_importances.index)
    # Set the chart title and axis labels
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    # Show the plot
    plt.show()


def plot_precision_recall(actual, preds):
    """Plots precision and recall for binary classification.
    
    Parameters:
    actual (array-like): Array of true labels.
    preds (array-like): Array of predicted labels.
    
    Returns:
    None
    
    """
    plt.figure(figsize=(16, 4))

    # Precision plot
    plt.subplot(1, 2, 1)
    precision_0 = np.sum((actual == 0) & (preds == 0)) / np.sum(preds == 0)
    precision_1 = np.sum((actual == 1) & (preds == 1)) / np.sum(preds == 1)
    sns.barplot(x=['Class 0', 'Class 1'], y=[precision_0, precision_1])
    plt.xticks(fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.ylabel('Precision', fontsize=20)
    plt.title(f'Precision Class 0: {round(precision_0,2)}\nPrecision Class 1: {round(precision_1,2)}', fontsize=20)

    # Recall plot
    plt.subplot(1, 2, 2)
    recall_0 = np.sum((actual == 0) & (preds == 0)) / np.sum(actual == 0)
    recall_1 = np.sum((actual == 1) & (preds == 1)) / np.sum(actual == 1)
    sns.barplot(x=['Class 0', 'Class 1'], y=[recall_0, recall_1])
    plt.xticks(fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.ylabel('Recall', fontsize=20)
    plt.title(f'Recall Class 0: {round(recall_0,2)}\nRecall Class 1: {round(recall_1,2)}', fontsize=20)

    plt.tight_layout()
    plt.show()


def calculate_net_revenue(amount, actual_class, predicted_class, results=True):
    """
    Calculates the net revenue of a business based on the predicted and actual class of credit card transactions,
    considering the cost of additional authentication procedures for transactions labeled as fraud.

    Parameters:
    amount (pd.Series): Series with the amount of each transaction.
    actual_class (pd.Series): Series with the actual class (0: legit, 1: fraud) of each transaction.
    predicted_class (pd.Series): Series with the predicted class (0: legit, 1: fraud) of each transaction.

    Returns:
    Prints the following metrics:
    - Net Revenue Fraud detection model
    - Fraudulent Caught losses prevented
    - Fraudulent not Caught losses
    - Legit transactions labeled as fraud (lost revenue due to second authentication procedures)
    """

    # Calculate masks for legit and fraud transactions, as well as predicted legit and predicted fraud transactions
    legit_mask = (actual_class == 0)
    fraud_mask = (actual_class == 1)
    predicted_fraud_mask = (predicted_class == 1)
    predicted_legit_mask = (predicted_class == 0)

    # Calculate the total amount of legit and fraud transactions
    total_legit = amount[legit_mask].sum()
    total_fraud = amount[fraud_mask].sum()

    # Calculate the cost of sending 2 factor authentication method for each transaction labeled as fraud
    cost_fraud_auth = 1  # assuming $1 cost per additional authentication procedure
    fraud_cost = predicted_fraud_mask.sum() * cost_fraud_auth

    # Calculate the amount of fraud that was correctly detected by the model and the amount of fraud that was not detected
    fraud_as_fraud = amount[predicted_fraud_mask & fraud_mask].sum()
    fraud_not_caught = total_fraud - fraud_as_fraud

    # Calculate the net revenue for the fraud detection model
    revenue_fraud = fraud_as_fraud - fraud_not_caught - fraud_cost

    if results == True:
        # Print the results
        print("Fraudulent Caught losses prevented: ${:.2f}".format(fraud_as_fraud))
        print("Fraudulent not Caught losses: ${:.2f}".format(fraud_not_caught))
        print("Transactions labeled as fraud (cost due to second authentication procedures): ${:.2f}".format(fraud_cost))
        print("Net Revenue Fraud detection model: ${:.2f}".format(revenue_fraud))
        
    else:
        return [round(fraud_as_fraud, 2), round(fraud_not_caught, 2), round(fraud_cost, 2), round(revenue_fraud, 2)]

if __name__ == "__main__":
    pass