
from sklearn.datasets import clear_data_home

from data_collector import DataCollector
from model import ModelTrainer
from analyses import DataAnalyses


def main():
    coll = DataCollector()
    clean_data = coll.clean_dataset()

    trainer = ModelTrainer()
    X,y_encoded,features = trainer.prepare_features(clean_data)

    trainer.create_baseline_model(X, y_encoded)
    trainer.train_models(X, y_encoded)

    best_model_name = trainer.compare_models()

    analyses = DataAnalyses(trainer.label_encoder)

    if best_model_name:
        best_model = trainer.models[best_model_name]

        analyses.analyze_market_performance(clean_data)

        analyses.analyze_probability_calibration(best_model,X,y_encoded,clean_data)

        value_bets = trainer.find_value_bets(best_model,X, clean_data,threshold = 0.03)

        trainer.check_for_leakage(clean_data, features)


            # Summary
        print("\n" + "=" * 60)
        print(" FINAL RESULTS")
        print("=" * 60)
        print(f"   Baseline (Market) Accuracy: {trainer.results['Baseline_Market']['accuracy']:.3f}")
        print(f"   Best Model Accuracy: {trainer.results[best_model_name]['accuracy']:.3f}")
        print(
            f"   Improvement: {(trainer.results[best_model_name]['accuracy'] / trainer.results['Baseline_Market']['accuracy'] - 1) * 100:.1f}%")
        print(f"   Value Bets Found: {len(value_bets)}")





if __name__ == "__main__":
    main()