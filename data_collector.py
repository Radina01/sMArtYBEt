import os
import re

import pandas as pd
import numpy as np


class DataCollector:
    def __init__(self, team_strength_calculator=None, quick_win=None):
        self.data = None
        self.clean_data = None
        self.team_strength_calculator = team_strength_calculator
        self.quick_win = quick_win

    def get_epl_dataset(self):

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
        self.data.to_csv('epl_data.csv', index=False)
        return self.data

    def _parse_dates_robustly(self, df):

        date_series = df['Date'].copy()

        # Method 1: Try standard format (dd/mm/yy or dd/mm/yyyy)
        parsed_dates = pd.to_datetime(date_series, format='%d/%m/%Y', errors='coerce')

        # Method 2: Try alternative format (dd/mm/yy)
        if parsed_dates.isna().any():
            mask = parsed_dates.isna()
            alternative_parsed = pd.to_datetime(date_series[mask], format='%d/%m/%y', errors='coerce')
            parsed_dates[mask] = alternative_parsed

        # Method 3: Try any format that pandas can infer
        if parsed_dates.isna().any():
            mask = parsed_dates.isna()
            inferred_parsed = pd.to_datetime(date_series[mask], infer_datetime_format=True, errors='coerce')
            parsed_dates[mask] = inferred_parsed

        # Method 4: Manual parsing for common patterns
        if parsed_dates.isna().any():
            mask = parsed_dates.isna()
            remaining_dates = date_series[mask]

            def manual_parse(date_str):
                if pd.isna(date_str):
                    return pd.NaT

                date_str = str(date_str).strip()

                patterns = [
                    r'(\d{1,2})/(\d{1,2})/(\d{4})',
                    r'(\d{1,2})/(\d{1,2})/(\d{2})',
                    r'(\d{4})-(\d{1,2})-(\d{1,2})',
                ]

                for pattern in patterns:
                    match = re.search(pattern, date_str)
                    if match:
                        groups = match.groups()
                        if len(groups) == 3:
                            try:
                                if len(groups[2]) == 4:
                                    return pd.Timestamp(int(groups[2]), int(groups[1]), int(groups[0]))
                                else:
                                    year = int(groups[2])
                                    year = year + 2000 if year < 100 else year
                                    return pd.Timestamp(year, int(groups[1]), int(groups[0]))
                            except (ValueError, TypeError):
                                continue

                return pd.NaT

            manual_parsed = remaining_dates.apply(manual_parse)
            parsed_dates[mask] = manual_parsed

        failed_count = parsed_dates.isna().sum()
        success_count = len(parsed_dates) - failed_count

        print(f" Date parsing results:")
        print(f"   Successfully parsed: {success_count} dates")
        print(f"   Failed to parse: {failed_count} dates")

        if failed_count > 0:
            print(f"   Sample failed dates: {date_series[parsed_dates.isna()].head(5).tolist()}")

        if success_count > 0:
            valid_dates = parsed_dates[parsed_dates.notna()]
            print(
                f"   Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")

        return parsed_dates

    def clean_dataset(self):
        if os.path.exists('epl_clean_data.csv'):
            self.clean_data = pd.read_csv('epl_clean_data.csv')
            print("Cleaned data file exists")
            self.clean_data['Date'] = pd.to_datetime(self.clean_data['Date'])
            return self.clean_data
        clean_df = self.clean_raw_data()
        clean_df = self.add_features(clean_df)

        self.clean_data = clean_df
        self.clean_data.to_csv('epl_clean_data.csv', index=False)

        return clean_df

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

    def add_market_prob_features(self, df):
        total_probs = (1 / df['B365H']) + (1 / df['B365D']) + (1 / df['B365A'])

        df['Market_Prob_Home'] = (1 / df['B365H']) / total_probs
        df['Market_Prob_Away'] = (1 / df['B365A']) / total_probs
        df['Market_Prob_Draw'] = (1 / df['B365D']) / total_probs
        return df

    def add_features(self, df):
        df['Total_Goals'] = df['FTHG'] + df['FTAG']
        df['Goal_Difference'] = df['FTHG'] - df['FTAG']
        df['Total_Shots'] = df['HS'] + df['AS']
        df['Shot_Ratio'] = df['HS'] / (df['HS'] + df['AS'])

        df = self.add_market_prob_features(df)
        df['Date'] = self._parse_dates_robustly(df)

        conditions = [
            df['FTHG'] > df['FTAG'],
            df['FTHG'] < df['FTAG'],
            df['FTHG'] == df['FTAG'],
        ]

        choices = ['H', 'A', 'D']
        df['Result'] = np.select(conditions, choices, default='D')

        if self.team_strength_calculator:

            df = self.team_strength_calculator.create_team_strength_features(df, window=10)
            print(f"Added team strength features. Total columns: {len(df.columns)}")
        else:
            print("No team strength calculator provided - using basic features only")

        if self.quick_win:
            df = self.quick_win.create_quick_win_features(df)
            print(f"Added quick win features. Total columns: {len(df.columns)}")
        else:
            print("No quick win provided - using basic features only")

        return df
