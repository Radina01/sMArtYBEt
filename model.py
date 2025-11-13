import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    def __init__(self,label_encoder):
        self.models = {}
        self.results = {}
        self.label_encoder = label_encoder

    def create_baseline_model(self, X, y_encoded):
        baseline_prediction = X[['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home']].values

        baseline_pred_encoded = baseline_prediction.argmax(axis=1)
        print(f"Label encoding: {list(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        accuracy = accuracy_score(y_encoded, baseline_pred_encoded)
        log_loss_score = log_loss(y_encoded, baseline_prediction)

        self.results['Baseline_Market'] = {
            'accuracy': accuracy,
            'log_loss': log_loss_score,
        }

        print(f"   Baseline Accuracy: {accuracy:.3f}")
        print(f"   Baseline Log Loss: {log_loss_score:.3f}")

        return baseline_pred_encoded

    def train_models(self, X, y_encoded):
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,test_size=0.2, random_state=42,stratify=y_encoded)

        print(f"   Training set: {len(X_train)} games")
        print(f"   Test set: {len(X_test)} games")

        models = {
            'Logistic Regression': LogisticRegression(random_state=42,max_iter=1000,C = 0.1),
            'Random Forest': RandomForestClassifier(random_state=42,n_estimators=100,max_depth=10),
            'XGBoost': xgb.XGBClassifier(random_state=42,eval_metric = 'mlogloss',max_depth=6,learning_rate=0.1),
        }

        for name, model in models.items():
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

    def  predict_match(self,model,home_team,away_team,home_odds,away_odds,draw_odds,home_strength,away_strength,feature_names):
        total_prob=(1/home_odds)+(1/away_odds)+(1/draw_odds)
        market_prob_home = (1/home_odds)/total_prob
        market_prob_draw = (1/draw_odds)/total_prob
        market_prob_away = (1/away_odds)/total_prob


        all_features = {
            'Market_Prob_Away': market_prob_away,
            'Market_Prob_Draw': market_prob_draw,
            'Market_Prob_Home': market_prob_home,
            'Home_Avg_Goals_For': home_strength['Avg_Goals_For'],
            'Home_Avg_Goals_Against': home_strength['Avg_Goals_Against'],
            'Home_Avg_Points': home_strength['Avg_Points'],
            'Home_Form': home_strength['Form'],
            'Away_Avg_Goals_For': away_strength['Avg_Goals_For'],
            'Away_Avg_Goals_Against': away_strength['Avg_Goals_Against'],
            'Away_Avg_Points': away_strength['Avg_Points'],
            'Away_Form': away_strength['Form'],

            'Home_Attack_Strength' : home_strength['Avg_Goals_For'] / (away_strength['Avg_Goals_Against'] + 0.1),

            'Away_Attack_Strength' : away_strength['Avg_Goals_For'] / (home_strength['Avg_Goals_Against'] + 0.1),

            'Strength_Difference' : home_strength['Avg_Points'] - away_strength['Avg_Points']
        }
        feature_values = []
        for feature in feature_names:
            if feature in all_features:
                feature_values.append(all_features[feature])
            else:
                feature_values.append(1.0)

        prediction_array = np.array([feature_values])

        probabilities = model.predict_proba(prediction_array)[0]
        predicted_class = model.predict(prediction_array)[0]
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        return {
            'match': f"{home_team} vs {away_team}",
            'predicted_result': predicted_label,
            'probabilities': {
                'Home': probabilities[2],
                'Draw': probabilities[1],
                'Away': probabilities[0]
            },
            'confidence': np.max(probabilities),
            'market_odds': f"H:{home_odds:.2f} D:{draw_odds:.2f} A:{away_odds:.2f}"
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


