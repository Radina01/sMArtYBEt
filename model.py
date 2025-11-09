import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()

    def prepare_features(self, clean_data):
        feature_columns = ['Market_Prob_Home', 'Market_Prob_Draw', 'Market_Prob_Away',
            'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF',
            'HY', 'AY', 'HR', 'AR', 'Total_Goals',
            'Total_Shots', 'Shot_Ratio']

        x = clean_data[feature_columns]
        y = clean_data['Result']

        y_encoded = self.label_encoder.fit_transform(y)
# Verify encoding works
        print(f"Features: {len(feature_columns)} columns")
        print(f"Target: {len(y)} games")
        print(f"Label encoding: {list(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        return x,y_encoded,feature_columns

    def create_baseline_model(self, X, y_encoded):
        baseline_prediction = X[['Market_Prob_Home', 'Market_Prob_Draw', 'Market_Prob_Away']]
        n_classes = len(self.label_encoder.classes_)
        print(f"Number of classes in encoder: {n_classes}")
        print(f"Label encoder classes: {self.label_encoder.classes_}")

        baseline_pred_encoded = baseline_prediction.values.argmax(axis=1)

#adjust if there are only two classes
        if n_classes == 3:
            baseline_pred_encoded= baseline_prediction.values.argmax(axis=1)
        elif n_classes == 2:
            print(" Only 2 classes detected")
            home_away_probs = baseline_prediction[['Market_Prob_Home', 'Market_Prob_Away']].values

            home_away_probs_normalized = home_away_probs/home_away_probs.sum(axis=1,keepdims=True)

            baseline_prediction = home_away_probs_normalized.argmax(axis=1)
        else:
            raise ValueError(f"Unexpected number of classes: {n_classes}")

        accuracy = accuracy_score(y_encoded, baseline_pred_encoded)
        if n_classes == 3:
            log_loss_score = log_loss(y_encoded, baseline_prediction.values)
        else:
            log_loss_score = log_loss(y_encoded, home_away_probs_normalized)


        self.results['Baseline_Market'] = {
            'accuracy': accuracy,
            'log_loss': log_loss_score,
        }

        print(f"   Baseline Accuracy: {accuracy:.3f}")
        print(f"   Baseline Log Loss: {log_loss_score:.3f}")

        return baseline_prediction

    def train_models(self, X, y_encoded):
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,test_size=0.2, random_state=42,stratify=y_encoded)

        print(f"   Training set: {len(X_train)} games")
        print(f"   Test set: {len(X_test)} games")
        n_classes = len(np.unique(y_encoded))
        print(f"🔍 Training with {n_classes} classes")
        models = {
            'Logistic Regression': LogisticRegression(random_state=42,max_iter=10000),
            'Random Forest': RandomForestClassifier(random_state=42,n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42,eval_metric = 'mlogloss'),
        }

        for name, model in models.items():
            print(f"   Training {name}")
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            log_loss_score = log_loss(y_test, y_pred_proba)

            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'log_loss': log_loss_score,
                'model': model
            }

            print(f"     {name} Accuracy: {accuracy:.3f}")
            print(f"     {name} Log Loss: {log_loss_score:.3f}")

    def compare_models(self):
        for model_name, metrics in self.results.items():
            print(f"{model_name:20} | Accuracy: {metrics['accuracy']:.3f} | Log Loss: {metrics['log_loss']:.3f}")

        if len(self.results)>1:
            ml_results = {k: v for k ,v in self.results.items() if k != 'Baseline_Market'}
            if ml_results:
                best_model = min(ml_results.items(), key = lambda x: x[1]['log_loss'])

                print(f"Best model: {best_model[0]}")
                return best_model[0]
        return None