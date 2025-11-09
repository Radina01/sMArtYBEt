from pyexpat import features
from sklearn.datasets import clear_data_home

from data_collector import DataCollector
from model import ModelTrainer

def main():


    coll = DataCollector()

    clean_data = coll.clean_dataset()
    print(f"Clean data: {len(clean_data)} matches")

    result_counts = clean_data['Result'].value_counts()
    print(f"🔍 Result value counts:")
    for result, count in result_counts.items():
        print(f"   {result}: {count} games ({count / len(clean_data) * 100:.1f}%)")

    trainer = ModelTrainer()
    X,y_encoded,features = trainer.prepare_features(clean_data)

    trainer.create_baseline_model(X, y_encoded)
    trainer.train_models(X, y_encoded)

    trainer.compare_models()

if __name__ == "__main__":
    main()