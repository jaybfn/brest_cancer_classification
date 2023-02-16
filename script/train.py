#Prompt2 : refactor the entire code in to functions

# Output: Sure, here's the refactored code with functions:

# Promt3 : you forgot to import gridsearchCV

# Output : You're right, my apologies for the oversight. Here's the updated code with the missing import.

# Prompt 4: Now add Confusion Matix, precision, recall and F1 score and ROC curve to function evaluate model.

# Output: Sure! Here's the complete code with the confusion matrix, precision, recall, F1-score, and ROC curve:

# Prompt 5: After running your code I am getting this error "  File "/home/jaybfn/project/brest_cancer_classification/script/test.py", line 74, in <module>
#     accuracy, confusion, classification, fpr, tpr, auc_score = evaluate_model(model, X_test, y_test)
#   File "/home/jaybfn/project/brest_cancer_classification/script/test.py", line 49, in evaluate_model
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#   File "/home/jaybfn/anaconda3/envs/chatgpt/lib/python3.10/site-packages/sklearn/metrics/_ranking.py", line 992, in roc_curve
#     fps, tps, thresholds = _binary_clf_curve(
#   File "/home/jaybfn/anaconda3/envs/chatgpt/lib/python3.10/site-packages/sklearn/metrics/_ranking.py", line 766, in _binary_clf_curve
#     pos_label = _check_pos_label_consistency(pos_label, y_true)
#   File "/home/jaybfn/anaconda3/envs/chatgpt/lib/python3.10/site-packages/sklearn/metrics/_base.py", line 243, in _check_pos_label_consistency
#     raise ValueError(
# ValueError: y_true takes value in {'B', 'M'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly."

# Output: he error is being caused because the roc_curve() function requires binary labels to calculate the ROC curve. In your case, 
#the y_true values are 'B' and 'M'. To fix this issue, you need to convert 'B' and 'M' to binary labels before calculating the ROC curve. 
#You can do this by replacing 'B' with 0 and 'M' with 1 in both y_true and y_pred arrays.
# To do this, you can modify the evaluate_model() function as follows:

# prompt 5: Now add the code for saving the model and scaling model `scaler` 
# output: ure, here's the modified code with the trained model and scaler saved to disk:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import pickle

def load_data(file_path):
    """Load the dataset from a csv file."""
    df = pd.read_csv(file_path)
    return df

def drop_columns(df, cols):
    """Drop columns from the dataset."""
    df = df.drop(cols, axis=1)
    return df

def prepare_data(df, target_col, test_size=0.3, random_state=42):
    """Prepare the data for training by splitting it into training and testing sets and scaling it."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train a logistic regression model on the training data."""
    model = LogisticRegression(max_iter = 10000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the testing data and return the accuracy, confusion matrix, classification report, and ROC curve."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Convert 'B' and 'M' to binary labels
    y_test = np.where(y_test == 'B', 0, 1)
    y_pred = np.where(y_pred == 'B', 0, 1)

    # Calculate confusion matrix, classification report, and ROC curve
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)

    return accuracy, confusion, classification, fpr, tpr, auc_score

def fine_tune_model(model, X_train, y_train):
    """Fine-tune the model using grid search."""
    param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    return grid

if __name__ == "__main__":
    # Load the dataset
    df = load_data("../data/breast-cancer.csv")

    # Drop the unnecessary columns
    df = drop_columns(df, ["id"])

    # Prepare the data for training
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, "diagnosis")

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # save the scaler model
    with open("scaler_model.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Evaluate the model
    accuracy, confusion, classification, fpr, tpr, auc_score = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(classification)
    print("AUC Score:", auc_score)

    # Fine-tune the model (optional)
    grid = fine_tune_model(model, X_train, y_train)
    print("Best hyperparameters:", grid.best_params_)
    print("Accuracy:", grid.best_score_)

    # Plot the ROC curve
    plt.plot(fpr, tpr)
