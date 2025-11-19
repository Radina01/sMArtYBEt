import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve


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
