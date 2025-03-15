import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

iris_model = RandomForestClassifier(n_estimators=100, random_state=42)
iris_model.fit(X_train_iris, y_train_iris)

y_pred_iris = iris_model.predict(X_test_iris)

accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
print(f"Iris Dataset Model Accuracy: {accuracy_iris:.2f}")

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None

    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    categorical_columns = ["Gender", "City", "Profession", "Dietary Habits", "Degree", 
                           "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]
    
    for col in categorical_columns:
        df[col] = df[col].astype(str).str.lower()

    sleep_mapping = {"less than 5 hours": 4, "5-6 hours": 5.5, "7-8 hours": 7.5, "more than 8 hours": 9}
    df["Sleep Duration"] = df["Sleep Duration"].map(sleep_mapping)

    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    imputer = KNNImputer(n_neighbors=5)
    df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

    return df, encoders

def train_model(df):
    X = df.drop(columns=["Depression", "Have you ever had suicidal thoughts ?"])
    y_depression = df["Depression"]
    y_suicidal = df["Have you ever had suicidal thoughts ?"]

    X_train, X_test, y_train_depression, y_test_depression = train_test_split(X, y_depression, test_size=0.2, random_state=42)
    X_train_suicidal, X_test_suicidal, y_train_suicidal, y_test_suicidal = train_test_split(X, y_suicidal, test_size=0.2, random_state=42)

    depression_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=15, min_samples_split=4)
    suicidal_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=15, min_samples_split=4)

    depression_model.fit(X_train, y_train_depression)
    suicidal_model.fit(X_train_suicidal, y_train_suicidal)

    depression_pred = depression_model.predict(X_test)
    suicidal_pred = suicidal_model.predict(X_test_suicidal)

    print(f"Depression Model Accuracy: {accuracy_score(y_test_depression, depression_pred):.2f}")
    print(f"Suicidal Thoughts Model Accuracy: {accuracy_score(y_test_suicidal, suicidal_pred):.2f}")

    return depression_model, suicidal_model

def predict_mental_health(model_depression, model_suicidal, encoders, input_data):
    df_input = pd.DataFrame([input_data])

    if 'id' in df_input.columns:
        df_input = df_input.drop(columns=['id'])

    for col in encoders:
        if col in df_input:
            value = df_input[col].values[0]
            if value not in encoders[col].classes_:
                unseen_index = len(encoders[col].classes_)
                df_input[col] = unseen_index
            else:
                df_input[col] = encoders[col].transform([value])

    sleep_mapping = {"less than 5 hours": 4, "5-6 hours": 5.5, "7-8 hours": 7.5, "more than 8 hours": 9}
    df_input["Sleep Duration"] = df_input["Sleep Duration"].map(sleep_mapping)

    depression_pred = model_depression.predict(df_input)[0]
    suicidal_pred = model_suicidal.predict(df_input)[0]

    depression_status = "Depressed" if depression_pred == 0 else "Not Depressed"
    suicidal_status = "Has Suicidal Thoughts" if suicidal_pred == 0 else "No Suicidal Thoughts"

    return depression_status, suicidal_status