from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve


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


def plot_ev_vs_profit(bets_with_stakes):
    ev = bets_with_stakes['model_prob'] * (bets_with_stakes['odds'] - 1) * bets_with_stakes['stake']
    realized = bets_with_stakes['profit']

    plt.figure(figsize=(8, 6))
    colors = ['green' if c else 'red' for c in bets_with_stakes['correct']]
    plt.scatter(ev, realized, c=colors, alpha=0.6, s=bets_with_stakes['stake'] * 0.5)
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
