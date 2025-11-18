import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(self, label_encoder):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.label_encoder = label_encoder
        self.scale = StandardScaler()

    def create_baseline_model(self, X, y_encoded):
        baseline_prediction = X[['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home']].values

        baseline_pred_encoded = baseline_prediction.argmax(axis=1)
        print(f"Label encoding: {list(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        accuracy = accuracy_score(y_encoded, baseline_pred_encoded)

        self.results['Baseline_Market'] = {
            'cv_means': accuracy,

        }

        print(f"   Baseline Accuracy: {accuracy:.3f}")

        return baseline_pred_encoded

    def create_models(self):

        pipeline_lr_poly = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('model', LogisticRegression(C=0.01, max_iter=2000, random_state=42))
        ])

        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=0.01, max_iter=2000, random_state=42))
        ])

        rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

        xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                  objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False,
                                  random_state=42)

        ensemble_model = VotingClassifier(
            estimators=[
                ('lr_poly', pipeline_lr_poly),
                ('lr', pipeline_lr),
                ('rf', rf_model),
                ('xgb', xgb_model)
            ],
            voting='soft'
        )
        self.models = {
            'LR_poly': pipeline_lr_poly,
            'LR': pipeline_lr,
            'RF': rf_model,
            'XGB': xgb_model,
            'Ensemble': ensemble_model
        }

        return self.models

    def calibrate_model(self, model, X_cal, y_cal, method='isotonic'):
        calibrated = CalibratedClassifierCV(estimator=model, method=method, cv='prefit')
        calibrated.fit(X_cal, y_cal)

        return calibrated

    def time_series_train_test_split(self, X, y, n_splits=5, date_col='Date'):
        # If X is a DataFrame and has a date column, sort by it
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if date_col in X.columns:
            X = X.sort_values(date_col).reset_index(drop=True)

            y = y.loc[X.index].reset_index(drop=True)
            X = X.drop(columns=[date_col])  # drop date after sorting

        n_samples = len(X)
        fold_size = n_samples // (n_splits + 1)

        for i in range(1, n_splits + 1):
            train_end = i * fold_size
            test_end = (i + 1) * fold_size

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]

            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            yield X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X, y, ):

        for model_name, model_obj in self.models.items():
            fold_accuracies = []

            for X_train, X_test, y_train, y_test in self.time_series_train_test_split(X, y):
                model_obj = model_obj.fit(X_train, y_train)
                self.trained_models[model_name] = model_obj
                preds = model_obj.predict(X_test)
                acc = accuracy_score(y_test, preds)
                fold_accuracies.append(acc)

            # Store average accuracy
            self.results[model_name] = np.mean(fold_accuracies)

        return self.results

    # Check if any features strongly correlate with the result
    def check_for_leakage(self, clean_data, feature_columns):
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

    def predict_match(self, clf, home_team, away_team, home_odds, draw_odds, away_odds, home_strength, away_strength,
                      feature_names):

        all_features = {

            'Home_Avg_Goals_For': home_strength['Avg_Goals_For'],
            'Home_Avg_Goals_Against': home_strength['Avg_Goals_Against'],
            'Home_Avg_Points': home_strength['Avg_Points'],
            'Home_Form': home_strength['Form'],
            'Away_Avg_Goals_For': away_strength['Avg_Goals_For'],
            'Away_Avg_Goals_Against': away_strength['Avg_Goals_Against'],
            'Away_Avg_Points': away_strength['Avg_Points'],
            'Away_Form': away_strength['Form'],

            'Home_Attack_Strength': home_strength['Avg_Goals_For'] / (away_strength['Avg_Goals_Against'] + 0.1),

            'Away_Attack_Strength': away_strength['Avg_Goals_For'] / (home_strength['Avg_Goals_Against'] + 0.1),

            'Strength_Difference': home_strength['Avg_Points'] - away_strength['Avg_Points'],
            'Goals_Ratio': home_strength['Avg_Goals_For'] / (away_strength['Avg_Goals_For'] + 0.1),
            'Form_Ratio': home_strength['Form'] / (away_strength['Form'] + 0.1),
            'Home_Advantage_x_Form': home_strength['Form'] * 1.5,
            'Away_Disadvantage_x_Form': away_strength['Form'] * 0.8,
            'Points_Difference': home_strength['Avg_Points'] - away_strength['Avg_Points'],
            'Is_Strong_vs_Weak': 1.0 if (home_strength['Avg_Points'] - away_strength['Avg_Points']) > 1.0 else 0.0,
            'Is_Even_Contest': 1.0 if abs(home_strength['Avg_Points'] - away_strength['Avg_Points']) <= 0.5 else 0.0,
            'Home_Goal_Difference': home_strength['Avg_Goals_For'] - home_strength['Avg_Goals_Against'],
            'Away_Goal_Difference': away_strength['Avg_Goals_For'] - away_strength['Avg_Goals_Against'],
            'Goal_Diff_Advantage': (home_strength['Avg_Goals_For'] - home_strength['Avg_Goals_Against']) -
                                   (away_strength['Avg_Goals_For'] - away_strength['Avg_Goals_Against'])

        }
        feature_values = [all_features.get(f, 1.0) for f in feature_names]
        X_match = np.array([feature_values])

        # Predict probabilities and class
        probs = clf.predict_proba(X_match)[0]
        predicted_class = clf.predict(X_match)[0]
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        model_probs = {
            'Home': probs[2],
            'Draw': probs[1],
            'Away': probs[0]
        }

        total = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)

        market_probs = {
            'Home': (1 / home_odds) / total,
            'Draw': (1 / draw_odds) / total,
            'Away': (1 / away_odds) / total
        }

        edge_home = model_probs['Home'] - market_probs['Home']
        edge_draw = model_probs['Draw'] - market_probs['Draw']
        edge_away = model_probs['Away'] - market_probs['Away']

        edges = {
            'Home': edge_home,
            'Draw': edge_draw,
            'Away': edge_away
        }

        best_outcome = max(edges, key=edges.get)
        best_edge = edges[best_outcome]

        is_value = best_edge > 0

        return {
            "match": f"{home_team} vs {away_team}",
            "predicted_result": predicted_label,
            "probabilities": model_probs,
            "confidence": max(model_probs.values()),
            "market_probs": market_probs,
            "edges": edges,
            "best_value_bet": best_outcome if is_value else None,
            "best_edge": best_edge,
            "is_value_bet": is_value
        }

    # def analyze_model_insights(self, model, feature_names, clean_data):
    #     if hasattr(model, 'coef_'):
    #
    #         coefficients = model.coef_
    #         feature_effects = {}
    #
    #         for i, feature in enumerate(feature_names):
    #             # Average effect on predictions
    #             avg_effect = coefficients[0][i] * clean_data[feature].mean()
    #             feature_effects[feature] = avg_effect
    #
    #         print("Feature contributions to predictions:")
    #         for feature, effect in sorted(feature_effects.items(), key=lambda x: abs(x[1]), reverse=True):
    #             direction = "increases" if effect > 0 else "decreases"
    #             print(f"   {feature}: {effect:.4f} ({direction} home win probability)")
