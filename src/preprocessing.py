# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_preprocessing_pipeline(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return X, y, preprocessor

def split_and_preprocess(X, y, preprocessor, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Save the full pipeline for reuse (e.g., during inference)
    joblib.dump(pipeline, "models/preprocessing_pipeline.joblib")

    return X_train_transformed, X_test_transformed, y_train, y_test
