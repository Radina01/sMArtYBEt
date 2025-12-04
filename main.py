import os.path

from data_features.data_collector import DataCollector
from data_features.features import PrepareFeatures
from models.model import ModelTrainer, load_pipelines, run_save_pipelines

CALIBRATED_MODEL_PATH = "models/calibrated_model.pkl"


def load_data():
    prepare_features = PrepareFeatures()
    trainer = ModelTrainer(prepare_features.label_encoder)

    coll = DataCollector()
    coll.get_epl_dataset()
    raw_data = coll.clean_raw_data()

    clean_data = prepare_features.build_dataset(raw_data)
    X, y, feature_columns = prepare_features.prepare_features(clean_data)
    clean_data['y'] = y
    return trainer, prepare_features, clean_data, X, y


def print_models_accuracies(model):
    print("Average Accuracy per model:")
    for model_name, acc in model.results.items():
        print(f"{model_name}: {acc:.3f}")

    best_model_name = max(model.results, key=model.results.get)
    print("Best model:", best_model_name)


def make_bet(model, prepare_features, calibrated_model, X_test_last, test_indexed):
    home_strength = prepare_features.get_current_strength("Man United", test_indexed)
    away_strength = prepare_features.get_current_strength("Everton", test_indexed)

    prediction = model.predict_match(
        prepare_features.label_encoder,
        calibrated_model,
        "Man United", "Everton",
        1.85, 3.80, 4.20,
        home_strength, away_strength,
        X_test_last,
    )

    print(f"\n PREDICTION RESULT:")
    print(f"Match: {prediction['match']}")
    print(f"Predicted: {prediction['predicted_result']}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print(f"Model Probabilities - H: {prediction['probabilities']['Home']:.1%}, "
          f"D: {prediction['probabilities']['Draw']:.1%}, "
          f"A: {prediction['probabilities']['Away']:.1%}")
    print(f"Market Probabilities - H: {prediction['market_probs']['Home']:.1%},"
          f"D: {prediction['market_probs']['Draw']:.1%},"
          f"A: {prediction['market_probs']['Away']:.1%} ")
    print(f"Edges (Model - Market)- H: {prediction['edges']['Home']:.1%},"
          f"D: {prediction['edges']['Draw']:.1%},"
          f"A: {prediction['edges']['Away']:.1%}")
    print(f"Edge: {prediction['best_edge']:.1%}")
    print(f"Best Value Bet: {prediction['best_value_bet']}")


def main():
    # model.create_baseline_model(X,y)
    # check_for_leakage(clean_data, feature_columns)
    if os.path.exists(CALIBRATED_MODEL_PATH):
        (model, calibrated_model, prepare_features,
         X_test_last, y_test_last,
         test_indexed) = load_pipelines()
    else:
        trainer, prepare_features, clean_data, X, y = load_data()
        (model, calibrated_model, prepare_features,
         X_test_last, y_test_last,
         test_indexed) = run_save_pipelines(trainer, prepare_features, clean_data, X, y)

    return model, calibrated_model, X_test_last, y_test_last, test_indexed, prepare_features


if __name__ == "__main__":
    main()
