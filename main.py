from multiprocessing.spawn import prepare

from sklearn.datasets import clear_data_home
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split

from data_collector import DataCollector
from model import ModelTrainer
from analyses import DataAnalyses
from features import PrepareFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


def main():
    prepare_features = PrepareFeatures()
    coll = DataCollector(team_strength_calculator=prepare_features,quick_win=prepare_features)
    coll.get_epl_dataset()
    analyses = DataAnalyses(prepare_features.label_encoder)

    clean_data = coll.clean_dataset()
    X, y, feature_columns = prepare_features.prepare_features(clean_data)
    clean_data['y'] = y

    model = ModelTrainer(prepare_features.label_encoder)
    train_data,test_data = model.time_series_train_test_split(clean_data,date_col='Date',test_size=0.2)

    core_features = ['Goals_Ratio', 'Form_Ratio', 'Home_Advantage_x_Form', 'Away_Disadvantage_x_Form',
                     'Points_Difference', 'Is_Strong_vs_Weak', 'Is_Even_Contest', 'Goal_Diff_Advantage',
                     'Home_Avg_Points', 'Away_Avg_Points', 'Home_Form', 'Away_Form']

    market_features = ['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home']

    features_with_market = core_features + market_features
    features_no_market = core_features.copy()

    X_train_with_market = train_data[features_with_market]
    X_test_with_market = test_data[features_with_market]
    X_train_no_market = train_data[features_no_market]
    X_test_no_market = test_data[features_no_market]

    y_train = train_data['y']
    y_test = test_data['y']

    pipe_with_market, pipe_no_market = model.create_models()

    pipe_with_market.fit(X_train_with_market, y_train)
    pipe_no_market.fit(X_train_no_market, y_train)

    # Calibrate both models (using held-out portion from train for calibration would be best; here we use a simple approach)
    # We'll reserve 20% of the trainset for calibration to avoid overfitting calibration step
    X_train_cal_mkt, X_cal_mkt, y_train_cal_mkt, y_cal_mkt = train_test_split(X_train_with_market, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=42)
    X_train_cal_nomkt, X_cal_nomkt, y_train_cal_nomkt, y_cal_nomkt = train_test_split(X_train_no_market, y_train, test_size=0.2, shuffle=True,stratify=y_train, random_state=42)


    # Calibrate both (isotonic recommended if you have enough samples)
    calibrated_with_market = model.calibrate_model(
        pipe_with_market,
        X_cal_mkt,
        y_cal_mkt,
        method='isotonic'
    )

    calibrated_no_market = model.calibrate_model(
        pipe_no_market,
        X_cal_nomkt,
        y_cal_nomkt,
        method='isotonic'
    )

    # Evaluate accuracy on test set
    probs_with_market = calibrated_with_market.predict_proba(X_test_with_market)
    preds_with_market = np.argmax(probs_with_market, axis=1)

    probs_no_market = calibrated_no_market.predict_proba(X_test_no_market.values)
    preds_no_market = np.argmax(probs_no_market, axis=1)

    acc_with_market = accuracy_score(y_test, preds_with_market)
    acc_no_market = accuracy_score(y_test, preds_no_market)

  #  brier_with_market = brier_score_loss(pd.get_dummies(y_test).values.argmax(axis=1), probs_with_market.argmax(axis=1))

    # Build value-bets using the NO-MARKET calibrated model (this is the right approach)
    # Use test set for realistic out-of-sample bets

    X_test_no_market_df = X_test_no_market.copy()
    # Ensure the columns match exactly what the pipeline was trained on
    X_test_no_market_df = X_test_no_market_df[features_no_market]
    test_indexed = test_data.copy()
    test_indexed.index = X_test_no_market_df.index


    value_bets_df = analyses.find_value_bets(calibrated_no_market,X_test_no_market_df, test_indexed, threshold=0.1189)
    # min_edge = 0.10  # only take bets where model edge > 10%
    # filtered_value_bets_df = value_bets_df[value_bets_df['edge'] >= min_edge].copy()

    # Recalculate ROI using filtered bets
    filtered_roi_results = analyses.calculate_roi(value_bets_df)

    print(f"Filtered value bets: {len(value_bets_df)}")
    print(f"Filtered ROI: {filtered_roi_results['roi']*100:.1f}%")
    print(f"Hit rate on filtered bets: {filtered_roi_results['hit_rate']:.3%}")
    print(f"acc_with_market: {acc_with_market*100:.1f}%\nacc_no_market: {acc_no_market*100:.1f}%\nnum_value_bets: {len(value_bets_df)}\n")

    analyses.diagnose_value_bet_increase(value_bets_df,calibrated_no_market,X_test_no_market_df,clean_data)

    # thresholds = np.arange(0.01, 0.21, 0.01)  # 0.01 to 0.20
    # roi_list = []
    # hit_rate_list = []
    # num_bets_list = []
    #
    # for th in thresholds:
    #     filtered_bets = value_bets_df[value_bets_df['edge'] >= th]
    #     if len(filtered_bets) == 0:
    #         roi_list.append(0)
    #         hit_rate_list.append(0)
    #         num_bets_list.append(0)
    #         continue
    #
    #     # Simulate ROI: profit = sum of (odds - 1) if correct, loss = 1 if incorrect
    #     profit = ((filtered_bets['odds'] - 1) * filtered_bets['correct']).sum()
    #     loss = ((~filtered_bets['correct']).astype(int)).sum()
    #     roi = profit / (profit + loss)
    #
    #     hit_rate = filtered_bets['correct'].mean()
    #
    #     roi_list.append(roi)
    #     hit_rate_list.append(hit_rate)
    #     num_bets_list.append(len(filtered_bets))
    #
    # # Plot ROI vs threshold
    # plt.figure(figsize=(10, 6))
    # plt.plot(thresholds, roi_list, marker='o', label='ROI')
    # plt.plot(thresholds, hit_rate_list, marker='x', label='Hit Rate')
    # plt.xlabel('Minimum Edge Threshold')
    # plt.ylabel('ROI / Hit Rate')
    # plt.title('ROI & Hit Rate vs Minimum Edge Threshold')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # # Optional: print optimal threshold
    # optimal_idx = np.argmax(roi_list)
    # print(
    #     f"Optimal edge threshold: {thresholds[optimal_idx]:.2f}, ROI: {roi_list[optimal_idx]:.3f}, Hit rate: {hit_rate_list[optimal_idx]:.3f}, Num bets: {num_bets_list[optimal_idx]}")


   #  trainer = ModelTrainer(prepare_features.label_encoder)
   #  X,y_encoded,features = prepare_features.prepare_features(clean_data)
   #
   #
   # # trainer.create_baseline_model(X, y_encoded)
   #  trainer.train_models(X, y_encoded)
   #
   #  best_model_name = trainer.compare_models()
   #
   #  analyses = DataAnalyses(prepare_features.label_encoder)
   #
   #  if best_model_name:
   #      best_model = trainer.models[best_model_name]
   #
   #     # analyses.analyze_market_performance(clean_data)
   #
   #      analyses.analyze_probability_calibration(best_model,X,y_encoded,clean_data)
   #
   #      value_bets = analyses.find_value_bets(best_model,X, clean_data,threshold = 0.05)
   #
   #      # increase = analyses.diagnose_value_bet_increase(value_bets,best_model,X,clean_data)
   #      # analyses.validate_value_bet_quality(value_bets, clean_data)
   #      # analyses.debug_probability_mapping(best_model,X, clean_data,10)
   #
   #
   #
   #      trainer.check_for_leakage(clean_data, features)


        #     # Summary
        # print("\n" + "=" * 60)
        # print(" FINAL RESULTS")
        # print("=" * 60)
        # print(f"   Baseline (Market) Accuracy: {trainer.results['Baseline_Market']['accuracy']:.3f}")
        # print(f"   Best Model Accuracy: {trainer.results[best_model_name]['accuracy']:.3f}")
        # print(
        #     f"   Improvement: {(trainer.results[best_model_name]['accuracy'] / trainer.results['Baseline_Market']['accuracy'] - 1) * 100:.1f}%")
        # print(f"   Value Bets Found: {len(value_bets)}")


        # home_strength = prepare_features.get_current_strength("Liverpool",clean_data,window=10)
        # away_strength = prepare_features.get_current_strength("Chelsea",clean_data,window=10)
        #
        # prediction = trainer.predict_match(model=best_model,
        # home_team="Liverpool",
        # away_team="Chelsea",
        # home_odds=2.10,
        # draw_odds=3.40,
        # away_odds=3.50,
        # home_strength=home_strength,
        # away_strength=away_strength,
        # feature_names=features)
        #
        # print(f"\n PREDICTION RESULT:")
        # print(f"Match: {prediction['match']}")
        # print(f"Predicted: {prediction['predicted_result']}")
        # print(f"Confidence: {prediction['confidence']:.1%}")
        # print(f"Probabilities - H: {prediction['probabilities']['Home']:.1%}, "
        #       f"D: {prediction['probabilities']['Draw']:.1%}, "
        #       f"A: {prediction['probabilities']['Away']:.1%}")
        #
        # home_strength = prepare_features.get_current_strength("Man United", clean_data, window=10)
        # away_strength = prepare_features.get_current_strength("Everton", clean_data, window=10)
        #
        # prediction1 = trainer.predict_match(model=best_model,
        #                                    home_team="Man United",
        #                                    away_team="Everton",
        #                                    home_odds=1.94,
        #                                    draw_odds=3.58,
        #                                    away_odds=4.15,
        #                                    home_strength=home_strength,
        #                                    away_strength=away_strength,
        #                                    feature_names=features)
        #
        # print(f"Match: {prediction1['match']}")
        # print(f"Predicted: {prediction1['predicted_result']}")
        # print(f"Confidence: {prediction1['confidence']:.1%}")
        # print(f"Probabilities - H: {prediction1['probabilities']['Home']:.1%}, "
        #       f"D: {prediction1['probabilities']['Draw']:.1%}, "
        #       f"A: {prediction1['probabilities']['Away']:.1%}")
        #
        #





if __name__ == "__main__":
    main()

