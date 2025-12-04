import os

import pandas as pd


class DataCollector:
    def __init__(self):
        self.data = None
        self.clean_data = None

    def get_epl_dataset(self):

        if os.path.exists('csv/epl_data.csv'):
            self.data = pd.read_csv('csv/epl_data.csv')
            print("Data file exists")
            return self.data

        seasons = {
            '2012-13': '1213', '2013-14': '1314', '2014-15': '1415',
            '2015-16': '1516', '2016-17': '1617', '2017-18': '1718',
            '2018-19': '1819', '2019-20': '1920', '2020-21': '2021',
            '2021-22': '2122', '2022-23': '2223', '2023-24': '2324'
        }

        all_data = []

        for season_name, season_code in seasons.items():
            try:
                url = f'https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv'
                df = pd.read_csv(url)
                df['Season'] = season_name
                all_data.append(df)
                print(f"Downloaded {season_name}")
            except Exception as e:
                print(f"Failed {season_name}: {e}")

        self.data = pd.concat(all_data, ignore_index=True)
        self.data.to_csv('csv/epl_data.csv', index=False)
        return self.data

    def clean_raw_data(self):
        df = self.data.copy()
        columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
            'B365H', 'B365D', 'B365A', 'HS', 'AS', 'HST', 'AST',
            'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR', 'Season'
        ]

        df = df[columns]
        df = df.dropna(subset=['B365H', 'B365D', 'B365A', 'FTHG', 'FTAG', 'HS', 'AS'])

        stat_col = ['HST', 'AST',
                    'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']

        for col in stat_col:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df
