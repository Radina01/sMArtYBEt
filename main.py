from multiprocessing.spawn import prepare

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import clear_data_home
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split

from bets_analyses import ValueBetsAnalyses
from data_collector import DataCollector
from model import ModelTrainer
from prob_analyses import DataAnalyses
from features import PrepareFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


def print_simulation_report(bets_analysis, equity_fixed, equity_kelly, bets_with_stakes, roi_results):
    # fixed stake
    dd_amt_f, peak_i_f, trough_i_f, dd_pct_f = bets_analysis.max_drawdown(equity_fixed)
    shp_f = bets_analysis.sharpe_ratio(equity_fixed, periods_per_year=252)

    # fractional kelly
    dd_amt_k, peak_i_k, trough_i_k, dd_pct_k = bets_analysis.max_drawdown(equity_kelly)
    shp_k = bets_analysis.sharpe_ratio(equity_kelly, periods_per_year=252)

    mean_roi, ci_lo, ci_hi, _ = roi_results

    print("=" * 60)
    print("SPORTS BETTING SIMULATION SUMMARY".center(60))
    print("=" * 60)

    print("\nFIXED STAKE SIMULATION")
    print(f"{'Starting Bankroll:':25} ${equity_fixed[0]:,.2f}")
    print(f"{'Final Bankroll:':25} ${equity_fixed[-1]:,.2f}")
    print(f"{'Total Profit:':25} ${equity_fixed[-1] - equity_fixed[0]:,.2f}")
    print(f"{'Peak Bankroll:':25} ${max(equity_fixed):,.2f}")
    print(f"{'Max Drawdown:':25} ${dd_amt_f:,.2f} ({dd_pct_f * 100:.1f}%)")
    print(f"{'Sharpe Ratio:':25} {shp_f:.2f}")
    print(f"{'Total Bets:':25} {len(bets_with_stakes)}")

    print("\nFRACTIONAL KELLY SIMULATION")
    print(f"{'Starting Bankroll:':25} ${equity_kelly[0]:,.2f}")
    print(f"{'Final Bankroll:':25} ${equity_kelly[-1]:,.2f}")
    print(f"{'Total Profit:':25} ${equity_kelly[-1] - equity_kelly[0]:,.2f}")
    print(f"{'Peak Bankroll:':25} ${max(equity_kelly):,.2f}")
    print(f"{'Max Drawdown:':25} ${dd_amt_k:,.2f} ({dd_pct_k * 100:.1f}%)")
    print(f"{'Sharpe Ratio:':25} {shp_k:.2f}")
    print(f"{'Total Bets:':25} {len(bets_with_stakes)}")

    print("\nBOOTSTRAPPED ROI (FIXED STAKE)")
    print(f"{'Average ROI:':25} {mean_roi * 100:.2f}%")
    print(f"{'95% Confidence Interval:':25} [{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]")

    print("\nTOP 5 VALUE BETS BY EDGE")
    top_bets = bets_with_stakes.sort_values('edge', ascending=False).head(5)
    for _, b in top_bets.iterrows():
        print(f"Game {b['game_index']:3}: Bet {['Away', 'Draw', 'Home'][b['selection']]:5} | "
              f"Edge: {b['edge']:.2%} | Odds: {b['odds']:.2f} | Correct: {b['correct']} | Stake: ${b['stake']:.2f}")
    print("=" * 60)


# PUT PLOT IN ANOTHER FILE
def plot_bankroll_max_drawdown_chart():
    folds = [1, 3, 5]
    final_bankroll_fixed = [20850, 1275, 14648]
    final_bankroll_kelly = [14934, 25288, 22980]
    max_drawdown_fixed = [0.082, 0.061, 0.041]
    max_drawdown_kelly = [0.205, 0.143, 0.097]
    sharpe_fixed = [3.09, 2.22, 5.2]
    sharpe_kelly = [2.63, 6.09, 5.73]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    width = 0.25
    x = np.arange(len(folds))
    ax[0].bar(x - width / 2, final_bankroll_fixed, width, label='Fixed Stake')
    ax[0].bar(x + width / 2, final_bankroll_kelly, width, label='Kelly')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(folds)
    ax[0].set_xlabel("Number of Folds")
    ax[0].set_ylabel("Final Bankroll ($)")
    ax[0].set_title("Bankroll vs Folds")
    ax[0].legend()

    ax[1].plot(folds, np.array(max_drawdown_fixed) * 100, marker='o', label='Fixed Stake')
    ax[1].plot(folds, np.array(max_drawdown_kelly) * 100, marker='o', label='Kelly')
    ax[1].set_xlabel("Number of Folds")
    ax[1].set_ylabel("Max Drawdown (%)")
    ax[1].set_title("Drawdown vs Folds")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_bankroll_over_time(equity_fixed, equity_kelly):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_fixed, label="Fixed Stake")
    plt.plot(equity_kelly, label="Kelly Fractional")

    plt.xlabel("Bet Number")
    plt.ylabel("Bankroll")
    plt.title("Bankroll Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_edge_distribution(value_bets_df):
    edges = value_bets_df["edge"]

    plt.figure(figsize=(10, 5))
    plt.hist(edges, bins=30)

    plt.xlabel("Edge (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Value Bet Edges")
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(calibrated_model, X_cal, y_cal, label_names=None):
    probs = calibrated_model.predict_proba(X_cal)
    num_classes = probs.shape[1]

    if label_names is None:
        label_names = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(10, 5))

    for c in range(num_classes):
        true_y = (y_cal == c).astype(int)
        prob_true, prob_pred = calibration_curve(true_y, probs[:, c], n_bins=10)

        plt.plot(prob_pred, prob_true, marker='o', label=f"{label_names[c]}")

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Plot")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ev_vs_profit(bets_df):
    ev = bets_df['model_prob'] * (bets_df['odds'] - 1) * bets_df['stake']
    realized = bets_df['profit']

    plt.figure(figsize=(8, 6))
    colors = ['green' if c else 'red' for c in bets_df['correct']]
    plt.scatter(ev, realized, c=colors, alpha=0.6, s=bets_df['stake'] * 0.5)
    plt.plot([0, max(ev.max(), realized.max())],
             [0, max(ev.max(), realized.max())],
             linestyle='--', color='black', label='y = x')
    plt.xlabel("Expected Value ($)")
    plt.ylabel("Realized Profit ($)")
    plt.title("Expected Value vs Realized Profit")
    plt.legend(["y = x", "Wins (green) / Losses (red)"])
    plt.grid(True)
    plt.show()


def plot_bootstrap_roi_distribution(roi_results):
    avg_roi, ci_low, ci_high, roi_samples = roi_results

    plt.figure(figsize=(10, 6))
    plt.hist(roi_samples, bins=40, alpha=0.7)
    plt.axvline(avg_roi, linestyle="--", label=f"Average ROI = {avg_roi:.2%}")
    plt.axvline(ci_low, color="red", linestyle="--", label=f"Lower 95% CI = {ci_low:.2%}")
    plt.axvline(ci_high, color="green", linestyle="--", label=f"Upper 95% CI = {ci_high:.2%}")

    plt.title("Bootstrap ROI Distribution")
    plt.xlabel("ROI")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def main():
    prepare_features = PrepareFeatures()
    coll = DataCollector(team_strength_calculator=prepare_features, quick_win=prepare_features)
    coll.get_epl_dataset()
    clean_data = coll.clean_dataset()

    model = ModelTrainer(prepare_features.label_encoder)

    X, y, feature_columns = prepare_features.prepare_features(clean_data)
    clean_data['y'] = y

    market_features = ['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home']
    X_no_market = X.drop(columns=[f for f in market_features if f in X.columns])

    model.create_models()

    avg_accuracies = model.train_and_evaluate(X_no_market, y)
    print("Average Accuracy per model:", avg_accuracies)

    best_model_name = max(model.results, key=model.results.get)
    best_model = model.models[best_model_name]
    print("Best model:", best_model_name)

    folds = list(model.time_series_train_test_split(X_no_market, y))
    X_train_last, X_test_last, y_train_last, y_test_last = folds[-1]

    best_model.fit(X_train_last, y_train_last)
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_test_last, y_test_last)

    X_test_no_market_df = X_test_last.copy()

    test_indexed = clean_data.iloc[X_test_last.index[0]: X_test_last.index[-1] + 1].copy()
    test_indexed.index = X_test_no_market_df.index

    bets_analysis = ValueBetsAnalyses(calibrated_model, X_test_no_market_df, test_indexed)
    value_bets_df = bets_analysis.find_value_bets()

    initial_bankroll = 10000
    stake_fixed = initial_bankroll * 0.01

    equity_fixed, bets_with_stakes = bets_analysis.simulate_bankroll(
        bets=value_bets_df,
        initial_bankroll=initial_bankroll,
        stake_fixed=stake_fixed
    )
    equity_kelly, bets_kelly = bets_analysis.simulate_bankroll_kelly()
    roi_results = bets_analysis.calculate_roi(
        bets_with_stakes,
        n_boot=2000,
        seed=42
    )
    print_simulation_report(bets_analysis, equity_fixed, equity_kelly, bets_with_stakes, roi_results)

    home_strength = prepare_features.get_current_strength("Man United", test_indexed)
    away_strength = prepare_features.get_current_strength("Everton", test_indexed)

    prediction = model.predict_match(
        calibrated_model,
        "Man United", "Everton",
        1.85, 3.80, 4.20,
        home_strength, away_strength,
        X_test_no_market_df
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

    plot_ev_vs_profit(bets_with_stakes)
    plot_bankroll_over_time(equity_fixed, equity_kelly)
    plot_edge_distribution(value_bets_df)
    plot_calibration_curve(calibrated_model, X_test_last, y_test_last, ["Home", "Draw", "Away"])
    plot_bootstrap_roi_distribution(roi_results)

    # (0.5445614035087718) 5
    # (0.5364522417153996) 3
    # (0.52046783625731) 1


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

if __name__ == "__main__":
    main()
