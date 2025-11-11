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
        feature_columns = sorted(['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home'])
            # 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF',
            # 'HY', 'AY', 'HR', 'AR',
            # 'Total_Shots', 'Shot_Ratio']

       # forbidden_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'Total_Goals', 'Goal_Difference']

        x = clean_data[feature_columns]
        y = clean_data['Result']

        y_encoded = self.label_encoder.fit_transform(y)
# Verify encoding works
        print(f"Features: {len(feature_columns)} columns")
        print(f"Label encoding: {list(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        return x,y_encoded,feature_columns

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

#Find potential value bets where model disagrees with market
    def find_value_bets(self, model, X, clean_data, threshold=0.03):

        model_proba = model.predict_proba(X)

        market_home = clean_data['Market_Prob_Home'].values
        market_draw = clean_data['Market_Prob_Draw'].values
        market_away = clean_data['Market_Prob_Away'].values

        # # DEBUG: Check if we need to normalize
        # market_sums = market_home + market_draw + market_away
        # print(f" Market probability sums: min={market_sums.min():.3f}, max={market_sums.max():.3f}")
        #
        # # If market probabilities don't sum to 1, we have a problem
        # if abs(market_sums.mean() - 1.0) > 0.01:
        #     print("Market probabilities don't sum to 1 - this explains the huge edges!")
        #
        # # DEBUG: Check for impossible probabilities
        # overconfident_predictions = (model_proba > 0.90).sum()
        # if overconfident_predictions > 0:
        #     print(f"{overconfident_predictions} overconfident predictions (>90%)")

        model_df = pd.DataFrame(model_proba, columns=self.label_encoder.classes_,index= X.index)

        value_bets=[]

        for index in X.index:
            model_home = model_df.loc[index,'H'] if 'H' in model_df.columns else 0
            model_draw = model_df.loc[index,'D'] if 'D' in model_df.columns else 0
            model_away = model_df.loc[index,'A'] if 'A' in model_df.columns else 0

            market_home = clean_data.loc[index,'Market_Prob_Home']
            market_draw = clean_data.loc[index,'Market_Prob_Draw']
            market_away = clean_data.loc[index,'Market_Prob_Away']

            home_edge = model_home - market_home
            away_edge = model_away - market_away
            draw_edge = model_draw - market_draw

            if home_edge > threshold:
                value_bets.append({
                    'game_index': index,
                    'home_team':clean_data.loc[index,'HomeTeam'],
                    'away_team':clean_data.loc[index,'AwayTeam'],
                    'bet_type':'HOME',
                    'model_prob': model_home,
                    'market_prob': market_home,
                    'edge': home_edge,
                    'odds': clean_data.loc[index,'B365H'],
                })
            if away_edge > threshold:
                value_bets.append({
                    'game_index': index,
                    'home_team':clean_data.loc[index,'HomeTeam'],
                    'away_team':clean_data.loc[index,'AwayTeam'],
                    'bet_type':'AWAY',
                    'model_prob': model_away,
                    'market_prob': market_away,
                    'edge': away_edge,
                    'odds': clean_data.loc[index,'B365A'],
                })

            if draw_edge > threshold:
                value_bets.append({
                    'game_index': index,
                    'home_team':clean_data.loc[index,'HomeTeam'],
                    'away_team':clean_data.loc[index,'AwayTeam'],
                    'bet_type':'DRAW',
                    'model_prob': model_draw,
                    'market_prob': market_draw,
                    'edge': draw_edge,
                    'odds': clean_data.loc[index,'B365D'],
                })

        value_df = pd.DataFrame(value_bets)
        if len(value_df)>0:
             print(f"Found {len(value_df)} potential value bets!")
             print("\n Top 10 Value Bets:")
             print(value_df[['home_team', 'away_team', 'bet_type', 'edge', 'odds']].head(10))
        else:
             print(" No value bets found with current threshold")
        return value_df

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


