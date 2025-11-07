import os

import pandas as pd
import numpy as np



class DataCollector:
    def __init__(self):
        self.data = None
        self.clean_data = None



    def get_epl_dataset(self): #"8 seasons of EPL with odds"#

        if os.path.exists('epl_data.csv'):
            self.data = pd.read_csv('epl_data.csv')
            print("Data file exists")
            return self.data

        seasons = {
            '2015-16': '1516', '2016-17': '1617', '2017-18': '1718',
            '2018-19': '1819', '2019-20': '1920', '2020-21': '2021',
            '2021-22': '2122', '2022-23': '2223', '2023-24': '2324'
        }

        all_data = []

        for season_name,season_code in seasons.items():
            try:
                url = f'https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv'
                df = pd.read_csv(url)
                df['Season'] = season_name
                all_data.append(df)
                print(f"Downloaded {season_name}")
            except Exception as e:
                print(f"Failed {season_name}: {e}")

        self.data = pd.concat(all_data, ignore_index=True)
        self.data.to_csv('epl_data.csv', index=False)
        return self.data
    def clean_dataset(self):
        if os.path.exists('epl_clean_data.csv'):
            self.clean_data = pd.read_csv('epl_clean_data.csv')
            print("Cleaned data file exists")
            return self.clean_data
        clean_df = self.data.copy()
        columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
            'B365H', 'B365D', 'B365A', 'HS', 'AS', 'HST', 'AST',
            'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR', 'Season'
        ]

        clean_df=clean_df[columns]
        clean_df = clean_df.dropna(subset=['B365H', 'B365D', 'B365A','FTHG', 'FTAG','HS', 'AS'])

        conditions = [
            clean_df['FTHG'] > clean_df['FTAG'],
            clean_df['FTHG'] < clean_df['FTAG'],
            clean_df['FTHG'] == clean_df['FTAG'],
        ]

        choices = ['H','A','H']
        clean_df['Result'] = np.select(conditions, choices, default='D')

#calculate the market probabilities and remove the vig
        total_probs = (1/clean_df['B365H']) + (1/clean_df['B365D']) + (1/clean_df['B365A'])

        clean_df['Market_Prob_Home'] = (1/clean_df['B365H'])/total_probs
        clean_df['Market_Prob_Away'] = (1/clean_df['B365A'])/total_probs
        clean_df['Market_Prob_Draw'] = (1/clean_df['B365D'])/total_probs

        stat_col = ['HST', 'AST',
            'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']

        for col in stat_col:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].fillna(0)

        clean_df['Total_Goals'] = clean_df['FTHG'] + clean_df['FTAG']
        clean_df['Goal_Difference'] = clean_df['FTHG'] - clean_df['FTAG']
        clean_df['Total_Shots'] = clean_df['HS'] + clean_df['AS']
        clean_df['Shot_Ratio'] = clean_df['HS'] / (clean_df['HS'] + clean_df['AS'])

        print(f"✅ Cleaning complete! {len(clean_df)} matches with COMPLETE data ready")
        self.clean_data = clean_df
        self.clean_data.to_csv('epl_clean_data.csv', index=False)

        return clean_df




