# src/model.py

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def train_logistic_regression(X_train, y_train):
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["liblinear", "lbfgs"],
    }

    pipeline = ImbPipeline(
        [("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    )

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    joblib.dump(grid.best_estimator_, "../models/logistic_model_tuned.joblib")
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from imblearn.pipeline import Pipeline as ImbPipeline
    import joblib

    param_grid = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [10, 20, 30, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    pipeline = ImbPipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced", random_state=42, n_jobs=-1
                ),
            )
        ]
    )

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    joblib.dump(grid.best_estimator_, "../models/random_forest_tuned.joblib")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc


def train_xgboost_model(X_train, y_train):
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    from imblearn.pipeline import Pipeline as ImbPipeline
    import joblib

    param_grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [3, 5, 8],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__subsample": [0.8, 1.0],
    }

    pipeline = ImbPipeline(
        [
            (
                "clf",
                XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    scale_pos_weight=1,
                    random_state=42,
                ),
            )
        ]
    )

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    joblib.dump(grid.best_estimator_, "../models/xgboost_model_tuned.joblib")
    return grid.best_estimator_
