import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None

    # Remove 'id' column if it's not useful for training/prediction
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # List of categorical columns that need encoding
    categorical_columns = ["Gender", "City", "Profession", "Dietary Habits", "Degree", 
                           "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]
    
    # Convert all categorical columns to lowercase for uniformity
    for col in categorical_columns:
        df[col] = df[col].astype(str).str.lower()

    # Map sleep durations to numerical values
    sleep_mapping = {"less than 5 hours": 4, "5-6 hours": 5.5, "7-8 hours": 7.5, "more than 8 hours": 9}
    df["Sleep Duration"] = df["Sleep Duration"].map(sleep_mapping)

    # Initialize label encoders for categorical columns
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Impute missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

    return df, encoders

def train_model(df):
    # Define feature set X and target variables y
    X = df.drop(columns=["Depression", "Have you ever had suicidal thoughts ?"])
    y_depression = df["Depression"]
    y_suicidal = df["Have you ever had suicidal thoughts ?"]

    # Split data into training and testing sets
    X_train, X_test, y_train_depression, y_test_depression = train_test_split(X, y_depression, test_size=0.2, random_state=42)
    X_train_suicidal, X_test_suicidal, y_train_suicidal, y_test_suicidal = train_test_split(X, y_suicidal, test_size=0.2, random_state=42)

    # Initialize Random Forest classifiers
    depression_model = RandomForestClassifier(n_estimators=100, random_state=42)
    suicidal_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train models
    depression_model.fit(X_train, y_train_depression)
    suicidal_model.fit(X_train_suicidal, y_train_suicidal)

    # Make predictions
    depression_pred = depression_model.predict(X_test)
    suicidal_pred = suicidal_model.predict(X_test_suicidal)

    # Output model accuracies
    print(f"Depression Model Accuracy: {accuracy_score(y_test_depression, depression_pred):.2f}")
    print(f"Suicidal Thoughts Model Accuracy: {accuracy_score(y_test_suicidal, suicidal_pred):.2f}")

    return depression_model, suicidal_model

def predict_mental_health(model_depression, model_suicidal, encoders, input_data):
    # Prepare the input data for prediction
    df_input = pd.DataFrame([input_data])

    # Remove 'id' if it's present in the input data
    if 'id' in df_input.columns:
        df_input = df_input.drop(columns=['id'])

    # Handle unseen labels by encoding them as a new category
    for col in encoders:
        if col in df_input:
            value = df_input[col].values[0]
            if value not in encoders[col].classes_:
                unseen_index = len(encoders[col].classes_)
                df_input[col] = unseen_index
            else:
                df_input[col] = encoders[col].transform([value])

    # Map sleep duration to numeric values
    sleep_mapping = {"less than 5 hours": 4, "5-6 hours": 5.5, "7-8 hours": 7.5, "more than 8 hours": 9}
    df_input["Sleep Duration"] = df_input["Sleep Duration"].map(sleep_mapping)

    # Make predictions using the trained models
    depression_pred = model_depression.predict(df_input)[0]
    suicidal_pred = model_suicidal.predict(df_input)[0]

    # Map predictions to labels
    depression_status = "Depressed" if depression_pred == 1 else "Not Depressed"
    suicidal_status = "Has Suicidal Thoughts" if suicidal_pred == 1 else "No Suicidal Thoughts"

    return depression_status, suicidal_status