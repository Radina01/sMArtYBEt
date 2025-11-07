from sklearn.datasets import clear_data_home

from data_collector import DataCollector

def main():
    print("Starting EPL collection")

    coll = DataCollector()

    data = coll.get_epl_dataset()
    print(f"Raw data: {len(data)} matches")

    clean_data = coll.clean_dataset()
    print(f"Clean data: {len(clean_data)} matches")

    print("\nClean Data Sample:")
    print(clean_data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result',
                      'Market_Prob_Home', 'Market_Prob_Draw', 'Market_Prob_Away']].head())

    print(f"   Target variable: 'Result' (H/D/A)")
    print(f"   Features: {len(clean_data.columns)} columns including odds and stats")



if __name__ == "__main__":
    main()