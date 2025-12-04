import os
import pickle

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def create_baseline_model(clean_data):
    market_cols = ['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home']
    market_probs = clean_data[market_cols].values
    y = clean_data['y'].values

    baseline_pred = np.argmax(market_probs, axis=1)

    accuracy = accuracy_score(y, baseline_pred)

    print(f"Baseline Market Accuracy: {accuracy:.3f}")

    return accuracy


def get_param_grids():
    return {
        'LR': {'model__C': [0.0001, 0.001, 0.01, 0.1, 0.3, 1, 2, 5, 10],
               'model__penalty': ['l2'],
               'model__solver': ['lbfgs', 'liblinear'],
               'model__class_weight': [None, 'balanced']},
        'RF': {'n_estimators': [200, 400, 600, 800],
               'max_depth': [5, 8, 12, 16, 20],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'max_features': ['sqrt', 'log2', None]},
        'XGB': {'n_estimators': [400, 800, 1200],
                'max_depth': [5, 8],
                'learning_rate': [0.02, 0.05, 0.1, 0.2],
                'gamma': [0, 0.5, 1.0, 2.0],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'reg_alpha': [0, 1],
                'reg_lambda': [1, 3], },
        'Ensemble': {}
    }


def calibrate_model(model, X_cal, y_cal, method='isotonic'):
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv='prefit')
    calibrated.fit(X_cal, y_cal)

    return calibrated


def run_save_pipelines(model, prepare_features, clean_data, X, y):
    market_features = ['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home']
    X_no_market = X.drop(columns=[f for f in market_features if f in X.columns])

    model.create_models()

    model.tune_hyperparameters(X_no_market, y)

    print("Average Accuracy per model:")
    for model_name, acc in model.results.items():
        print(f"{model_name}: {acc:.2f}")

    best_model_name = max(model.results, key=model.results.get)
    best_model = model.models[best_model_name]
    print("Best model:", best_model_name)

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))

    train_idx, test_idx = splits[-1]

    X_train_last = X.iloc[train_idx]
    X_test_last = X.iloc[test_idx]
    y_train_last = y[train_idx]
    y_test_last = y[test_idx]

    best_model.fit(X_train_last, y_train_last)
    calibrated_model = calibrate_model(best_model, X_test_last, y_test_last)

    X_test_no_market_df = X_test_last.copy()

    test_indexed = clean_data.iloc[X_test_last.index[0]: X_test_last.index[-1] + 1].copy()
    test_indexed.index = X_test_no_market_df.index
    os.makedirs("models", exist_ok=True)

    with open("models/calibrated_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "calibrated_model": calibrated_model,
            "prepare_features": prepare_features,
            "X_test_last": X_test_last,
            "y_test_last": y_test_last,
            "test_indexed": test_indexed,
        }, f)

    print("Model and PrepareFeatures saved.")
    return model, calibrated_model, prepare_features, X_test_last, y_test_last, test_indexed


def load_pipelines():
    with open("models/calibrated_model.pkl", "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    calibrated_model = saved["calibrated_model"]
    prepare_features = saved["prepare_features"]
    X_test_last = saved["X_test_last"]
    y_test_last = saved["y_test_last"]
    test_indexed = saved["test_indexed"]

    print("Calibrated model loaded.")

    return model, calibrated_model, prepare_features, X_test_last, y_test_last, test_indexed


def check_for_leakage(clean_data, feature_columns):
    print("\n" + "=" * 60)

    # Convert result to numeric for correlation
    result_numeric = clean_data['Result'].map({'H': 0, 'D': 1, 'A': 2})

    for feature in feature_columns:
        if feature in clean_data.columns:
            correlation = clean_data[feature].corr(result_numeric)
            if abs(correlation) > 0.3:  # Strong correlation
                print(f"High correlation: {feature} - Result: {correlation:.3f}")
            elif abs(correlation) > 0.1:
                print(f"Moderate correlation: {feature} - Result: {correlation:.3f}")


class ModelTrainer:
    def __init__(self, label_encoder):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.label_encoder = label_encoder
        self.scale = StandardScaler()

    def create_models(self):
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(solver='lbfgs', random_state=42))
        ])

        rf_model = RandomForestClassifier(random_state=42)

        xgb_model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False,
                                  random_state=42)

        ensemble_model = VotingClassifier(
            estimators=[
                ('lr', pipeline_lr),
                ('rf', rf_model),
                ('xgb', xgb_model)
            ],
            voting='soft'
        )
        self.models = {
            'LR': pipeline_lr,
            'RF': rf_model,
            'XGB': xgb_model,
            'Ensemble': ensemble_model
        }

        return self.models

    def tune_hyperparameters(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        param_grids = get_param_grids()

        for name, model in self.models.items():
            grid = param_grids.get(name, {})
            if grid:
                gscv = GridSearchCV(model, param_grid=grid, cv=tscv, scoring='accuracy', n_jobs=-1)
                gscv.fit(X, y)
                self.trained_models[name] = gscv.best_estimator_
                self.results[name] = gscv.best_score_
            else:
                model.fit(X, y)
                self.trained_models[name] = model
