import numpy as np

from bets_evaluations.bets import find_value_bets
from bets_evaluations.risk_evaluation import bootstrap_roi, max_drawdown, sharpe_ratio


# fixed stake
def simulate_bankroll(bets, initial_bankroll=10000, stake_fixed=100):
    bankroll = initial_bankroll
    history = [bankroll]
    stakes = []
    profits = []
    for _, r in bets.iterrows():
        stake = stake_fixed
        stake = min(stake, bankroll * 0.5)
        stakes.append(stake)
        profit = stake * (r['odds'] - 1) if r['correct'] else -stake
        profits.append(profit)
        bankroll += profit
        history.append(bankroll)
    bets2 = bets.copy().reset_index(drop=True)
    bets2['stake'] = stakes
    bets2['profit'] = profits
    return np.array(history), bets2


def simulate_bankroll_kelly(value_bets,
                            confidence_min=0.55,
                            kelly_fraction=0.25,
                            max_fraction=0.05,
                            initial_bankroll=10000):
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    stakes = []

    # Filter by model confidence
    value_bets = value_bets[value_bets['model_prob'] >= confidence_min].reset_index(drop=True)

    for _, row in value_bets.iterrows():
        p = float(row['model_prob'])
        odds = float(row['odds'])

        # Full Kelly fraction
        b = odds - 1
        kelly_f = max((p * (b + 1) - 1) / b, 0) if b > 0 else 0

        # Fractional Kelly
        stake_fraction = min(kelly_f * kelly_fraction, max_fraction)
        stake = stake_fraction * bankroll
        stakes.append(stake)

        profit = stake * (odds - 1) if row['correct'] else -stake
        bankroll += profit
        bankroll_history.append(bankroll)

    bets_with_stakes = value_bets.copy()
    bets_with_stakes['stake'] = stakes

    return np.array(bankroll_history), bets_with_stakes


def bets_simulations(calibrated_model, X_test_last, test_indexed):
    value_bets_df = find_value_bets(calibrated_model, X_test_last, test_indexed)

    initial_bankroll = 10000
    stake_fixed = initial_bankroll * 0.01

    equity_fixed, bets_with_stakes = simulate_bankroll(
        bets=value_bets_df,
        initial_bankroll=initial_bankroll,
        stake_fixed=stake_fixed
    )
    equity_kelly, bets_kelly = simulate_bankroll_kelly(value_bets_df)
    roi_results = bootstrap_roi(
        bets_with_stakes,
        n_boot=2000,
        seed=42
    )
    return equity_fixed, equity_kelly, bets_with_stakes, roi_results, value_bets_df


def print_simulation_report(equity_fixed, equity_kelly, bets_with_stakes, roi_results):
    # Fixed stake
    dd_amt_f, peak_i_f, trough_i_f, dd_pct_f = max_drawdown(equity_fixed)
    shp_f = sharpe_ratio(equity_fixed, periods_per_year=252)

    # Fractional kelly
    dd_amt_k, peak_i_k, trough_i_k, dd_pct_k = max_drawdown(equity_kelly)
    shp_k = sharpe_ratio(equity_kelly, periods_per_year=252)

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
