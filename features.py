import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PrepareFeatures:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def prepare_features(self, clean_data):
        leaky_features = [
            'FTHG', 'FTAG', 'Total_Goals', 'Goal_Difference',
            'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR',
            'Shot_Ratio', 'Total_Shots'
        ]
        meta_data = [
            'Date', 'HomeTeam', 'AwayTeam', 'Result', 'FTHG', 'FTAG', 'Season',
        ]
        exclude_columns = leaky_features + meta_data
        feature_columns = [col for col in clean_data.columns if col not in exclude_columns]
        feature_columns = sorted(feature_columns)

        x = clean_data[feature_columns]
        y = clean_data['Result']

        y_encoded = self.label_encoder.fit_transform(y)
        # Verify encoding works
        print(f"Features: {len(feature_columns)} columns")
        for feature in feature_columns:
            print(feature + "\n")
        print(f"Label encoding: {list(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        return x, y_encoded, feature_columns

    def game_history(self, team, current_date, df, window):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        current_date = pd.to_datetime(current_date)
        history = df[
            (df['Date'] < current_date) &
            (df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(window)
        return history

    def calculate_historical_performance(self, history, team):
        if len(history) == 0:
            return None

        goals_for = []
        goals_against = []
        points = []

        for _, match in history.iterrows():
            if match['HomeTeam'] == team:
                goals_for.append(match['FTHG'])
                goals_against.append(match['FTAG'])
                points.append(3 if match['FTHG'] > match['FTAG']
                              else (1 if match['FTHG'] == match['FTAG'] else 0))


            else:
                goals_for.append(match['FTAG'])
                goals_against.append(match['FTHG'])
                points.append(3 if match['FTAG'] > match['FTHG']
                              else (1 if match['FTAG'] == match['FTHG'] else 0))

        return goals_for, goals_against, points

    def create_team_strength_features(self, clean_data, window=10):
        df = clean_data.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)

        teams = pd.unique(df[['AwayTeam', 'HomeTeam']].values.ravel())

        home_strength_features = []
        away_strength_features = []

        for index, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            current_date = row['Date']

            home_history = self.game_history(home_team, current_date, df, window)
            away_history = self.game_history(away_team, current_date, df, window)

            home_performance = self.calculate_historical_performance(home_history, home_team)
            home_goals_for, home_goals_against, home_points = home_performance
            if home_performance is None:
                home_strength_features.append({
                    'index': index,
                    'Home_Avg_Goals_For': 1.0,  # League average prior
                    'Home_Avg_Goals_Against': 1.0,
                    'Home_Avg_Points': 1.3,
                    'Home_Form': 1.3
                })

            else:
                home_strength_features.append({
                    'index': index,
                    'Home_Avg_Goals_For': np.mean(home_goals_for),
                    'Home_Avg_Goals_Against': np.mean(home_goals_against),
                    'Home_Avg_Points': np.mean(home_points),
                    'Home_Form': np.mean(home_points[-5:] if len(home_points) > 5 else home_points),
                })

            away_performance = self.calculate_historical_performance(away_history, away_team)
            away_goals_for, away_goals_against, away_points = away_performance

            if away_performance is None:
                away_strength_features.append({
                    'index': index,
                    'Away_Avg_Goals_For': 1.0,
                    'Away_Avg_Goals_Against': 1.0,
                    'Away_Avg_Points': 1.3,
                    'Away_Form': 1.3
                })
            else:
                away_strength_features.append({
                    'index': index,
                    'Away_Avg_Goals_For': np.mean(away_goals_for),
                    'Away_Avg_Goals_Against': np.mean(away_goals_against),
                    'Away_Avg_Points': np.mean(away_points),
                    'Away_Form': np.mean(away_points[-5:] if len(away_points) >= 5 else away_points),
                })

        home_df = pd.DataFrame(home_strength_features).set_index('index')
        away_df = pd.DataFrame(away_strength_features).set_index('index')

        result_df = df.join(home_df).join(away_df)

        result_df['Home_Attack_Strength'] = result_df['Home_Avg_Goals_For'] / result_df['Away_Avg_Goals_Against']
        result_df['Away_Attack_Strength'] = result_df['Away_Avg_Goals_For'] / result_df['Home_Avg_Goals_Against']
        result_df['Strength_Difference'] = result_df['Home_Avg_Points'] - result_df['Away_Avg_Points']

        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.fillna(1.0)

        return result_df

    def get_current_strength(self, team_name, clean_data, window=10):
        clean_data = clean_data.copy()
        clean_data['Date'] = pd.to_datetime(clean_data['Date'])

        latest_date = clean_data['Date'].max()
        recent_matches = self.game_history(team_name, latest_date, clean_data, window)
        performance = self.calculate_historical_performance(recent_matches, team_name)
        if performance is None:
            return {
                'Avg_Goals_For': 1.0,
                'Avg_Goals_Against': 1.0,
                'Avg_Points': 1.3,
                'Form': 1.3
            }

        goals_for, goals_against, points = performance

        return {
            'Avg_Goals_For': np.mean(goals_for),
            'Avg_Goals_Against': np.mean(goals_against),
            'Avg_Points': np.mean(points),
            'Form': np.mean(points[-5:] if len(points) >= 5 else points)
        }

    def create_quick_win_features(self, clean_data):
        df = clean_data.copy()

        df['Goals_Ratio'] = df['Home_Avg_Goals_For'] / (df['Away_Avg_Goals_For'] + 0.1)
        df['Form_Ratio'] = df['Home_Form'] / (df['Away_Form'] + 0.1)
        df['Home_Advantage_x_Form'] = df['Home_Form'] * 1.5
        df['Away_Disadvantage_x_Form'] = df['Away_Form'] * 0.8

        df['Points_Difference'] = df['Home_Avg_Points'] - df['Away_Avg_Points']
        df['Is_Strong_vs_Weak'] = (df['Points_Difference'] > 1.0).astype(int)
        df['Is_Even_Contest'] = ((df['Points_Difference'] >= -0.5) & (df['Points_Difference'] <= 0.5)).astype(int)

        df['Home_Goal_Difference'] = df['Home_Avg_Goals_For'] - df['Home_Avg_Goals_Against']
        df['Away_Goal_Difference'] = df['Away_Avg_Goals_For'] - df['Away_Avg_Goals_Against']
        df['Goal_Diff_Advantage'] = df['Home_Goal_Difference'] - df['Away_Goal_Difference']

        return df
