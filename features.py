import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PrepareFeatures:
    def __init__(self):
        self.label_encoder = LabelEncoder()


    def prepare_features(self, clean_data):

        feature_columns = sorted(['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home',
                                  'Home_Avg_Goals_For', 'Home_Avg_Goals_Against', 'Home_Avg_Points', 'Home_Form',
                                  'Away_Avg_Goals_For', 'Away_Avg_Goals_Against', 'Away_Avg_Points', 'Away_Form',
                                  'Home_Attack_Strength', 'Away_Attack_Strength', 'Strength_Difference'
                                  ])
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

        return x, y_encoded, feature_columns


    def game_history(self, team,current_date,df,window):
        team = df[
                (df['Date'] < current_date) &
                (df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(window)
        return team

    def calculate_historical_performance(self,history,team):
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

    def create_team_strength_features(self, clean_data,window = 10):
        df = clean_data.copy()
        df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y', errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)

        teams = pd.unique(df[['AwayTeam','HomeTeam']].values.ravel())

        home_strength_features = []
        away_strength_features = []

        for index, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            current_date = row['Date']

            home_history = self.game_history(home_team,current_date,df,window)
            away_history = self.game_history(away_team,current_date,df,window)

            home_performance = self.calculate_historical_performance(home_history,home_team)
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
                    'Home_Form':np.mean(home_points[-5:] if len(home_points)>5 else home_points),
                })

            away_performance = self.calculate_historical_performance(away_history,away_team)
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

        result_df['Home_Attack_Strength'] = result_df['Home_Avg_Goals_For']/result_df['Away_Avg_Goals_Against']
        result_df['Away_Attack_Strength'] = result_df['Away_Avg_Goals_For']/result_df['Home_Avg_Goals_Against']
        result_df['Strength_Difference'] = result_df['Home_Avg_Points'] - result_df['Away_Avg_Points']

        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.fillna(1.0)

        return result_df

    def get_current_strength(self,team_name,clean_data,window=10):
        latest_date =clean_data['Date'].max()
        recent_matches = self.game_history(team_name,latest_date,clean_data,window)
        performance = self.calculate_historical_performance(recent_matches,team_name)
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










