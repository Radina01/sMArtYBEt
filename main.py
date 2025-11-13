from multiprocessing.spawn import prepare

from sklearn.datasets import clear_data_home

from data_collector import DataCollector
from model import ModelTrainer
from analyses import DataAnalyses
from features import PrepareFeatures


def main():
    prepare_features = PrepareFeatures()
    coll = DataCollector(team_strength_calculator=prepare_features)
    coll.get_epl_dataset()

    clean_data = coll.clean_dataset()

    trainer = ModelTrainer(prepare_features.label_encoder)


    X,y_encoded,features = prepare_features.prepare_features(clean_data)

    trainer.create_baseline_model(X, y_encoded)
    trainer.train_models(X, y_encoded)

    best_model_name = trainer.compare_models()

    analyses = DataAnalyses(prepare_features.label_encoder)

    if best_model_name:
        best_model = trainer.models[best_model_name]

        analyses.analyze_market_performance(clean_data)

        analyses.analyze_probability_calibration(best_model,X,y_encoded,clean_data)

        value_bets = analyses.find_value_bets(best_model,X, clean_data,threshold = 0.03)

        increase = analyses.diagnose_value_bet_increase(value_bets,best_model,X,clean_data)
        analyses.validate_value_bet_quality(value_bets, clean_data)
        analyses.debug_probability_mapping(best_model,X, clean_data,10)



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

        home_strength = prepare_features.get_current_strength("Liverpool",clean_data,window=10)
        away_strength = prepare_features.get_current_strength("Chelsea",clean_data,window=10)

        prediction = trainer.predict_match(model=best_model,
        home_team="Liverpool",
        away_team="Chelsea",
        home_odds=2.10,
        draw_odds=3.40,
        away_odds=3.50,
        home_strength=home_strength,
        away_strength=away_strength,
        feature_names=features)

        print(f"\n PREDICTION RESULT:")
        print(f"Match: {prediction['match']}")
        print(f"Predicted: {prediction['predicted_result']}")
        print(f"Confidence: {prediction['confidence']:.1%}")
        print(f"Probabilities - H: {prediction['probabilities']['Home']:.1%}, "
              f"D: {prediction['probabilities']['Draw']:.1%}, "
              f"A: {prediction['probabilities']['Away']:.1%}")

        home_strength = prepare_features.get_current_strength("Manchester United", clean_data, window=10)
        away_strength = prepare_features.get_current_strength("Everton", clean_data, window=10)

        prediction1 = trainer.predict_match(model=best_model,
                                           home_team="Manchester United",
                                           away_team="Everton",
                                           home_odds=1.94,
                                           draw_odds=3.58,
                                           away_odds=4.15,
                                           home_strength=home_strength,
                                           away_strength=away_strength,
                                           feature_names=features)

        print(f"Match: {prediction1['match']}")
        print(f"Predicted: {prediction1['predicted_result']}")
        print(f"Confidence: {prediction1['confidence']:.1%}")
        print(f"Probabilities - H: {prediction1['probabilities']['Home']:.1%}, "
              f"D: {prediction1['probabilities']['Draw']:.1%}, "
              f"A: {prediction1['probabilities']['Away']:.1%}")





if __name__ == "__main__":
    main()