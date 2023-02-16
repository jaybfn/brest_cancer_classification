# Now build a prediction python code which takes in data and make a predition  and refactor it using functions 

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def load_model(model_path):
    """Load the trained model from a file."""
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_scaler(scaler_path):
    """Load the scaler used to preprocess the data."""
    with open("scaler_model.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

def drop_columns(df, cols):
    """Drop columns from the dataset."""
    df = df.drop(cols, axis=1)
    return df

def preprocess_data(df, scaler):
    """Preprocess the data by scaling it."""
    X = df.values
    X = scaler.transform(X)
    return X

def make_prediction(model, X):
    """Make a prediction using the trained model."""
    y_pred = model.predict(X)
    return y_pred

if __name__ == "__main__":
    # Load the new data to make a prediction on
    df = pd.read_csv("../data/breast-cancer-predict.csv")

    # Drop the unnecessary columns
    df = drop_columns(df, ["id"])

    # Load the trained model and scaler
    model = load_model("trained_model.pkl")
    scaler = load_scaler("scaler.pkl")

    # Preprocess the new data
    X = preprocess_data(df, scaler)

    # Make a prediction
    y_pred = make_prediction(model, X)

    print(y_pred)