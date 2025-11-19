import numpy as np
import pandas as pd

class ValueBetsAnalyses:
    def __init__(self, model_wrapper, X_test, clean_data, bankroll=10000):
        self.model = model_wrapper
        self.X_test = X_test
        self.clean_data = clean_data
        self.bankroll = bankroll

    def find_value_bets(self, threshold=0.10):
        model_probs = self.model.predict_proba(self.X_test)  # shape: (n_games, 3)
        market_probs = self.clean_data[['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home']].values
        edges = model_probs - market_probs

        max_edges = edges.max(axis=1)

        selections = edges.argmax(axis=1)

        value_bets = []
        for i, edge in enumerate(max_edges):
            if edge >= threshold:
                selection = selections[i]
                outcome = self.clean_data['y'].iloc[i]

                odds_col = ['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home'][selection]
                odds = 1 / self.clean_data[odds_col].iloc[i]

                model_prob = model_probs[i, selection]

                value_bets.append({
                    'game_index': i,
                    'selection': selection,
                    'edge': edge,
                    'odds': odds,
                    'model_prob': model_prob,
                    'actual_result': outcome,
                    'correct': selection == outcome
                })

        return pd.DataFrame(value_bets)

    def calculate_roi(self, bets, n_boot=1000, seed=42):
        rng = np.random.RandomState(seed)
        rois = []
        stakes = bets.get('stake', pd.Series(1.0, index=bets.index)).values
        for _ in range(n_boot):
            idx = rng.randint(0, len(bets), len(bets))
            sample = bets.iloc[idx]
            stakes_s = stakes[idx]
            profit = np.where(sample['correct'].values, stakes_s * (sample['odds'].values - 1), -stakes_s).sum()
            turnover = stakes_s.sum()
            roi = profit / turnover if turnover > 0 else 0.0
            rois.append(roi)
        arr = np.array(rois)
        return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5), arr

    def compute_equity_curve(self, bets, initial_bankroll=10000):
        bankroll = initial_bankroll
        history = [bankroll]
        for _, r in bets.iterrows():
            stake = float(r['stake'])
            profit = stake * (r['odds'] - 1) if r['correct'] else -stake
            bankroll += profit
            history.append(bankroll)
        return np.array(history)

    def max_drawdown(self, equity):

        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        trough_idx = np.argmax(drawdowns)
        peak_idx = np.argmax(equity[:trough_idx + 1])
        max_dd = drawdowns.max()
        pct = max_dd / running_max[peak_idx] if running_max[peak_idx] > 0 else np.nan
        return max_dd, peak_idx, trough_idx, pct

    def sharpe_ratio(self, equity, risk_free_rate=0.0, periods_per_year=252):

        returns = np.diff(equity) / equity[:-1]
        if len(returns) < 2:
            return np.nan
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r == 0:
            return np.nan
        return (mean_r - risk_free_rate / periods_per_year) / std_r * np.sqrt(periods_per_year)

    def kelly_fraction(self, p, odds):
        # f* = (b*p - q) / b where b = odds-1, q = 1-p

        b = odds - 1.0
        q = 1.0 - p
        if b <= 0:
            return 0.0
        f = (b * p - q) / b
        return max(f, 0.0)

    def simulate_bankroll_kelly(self, threshold=0.10,
                                confidence_min=0.55,
                                kelly_fraction=0.25,
                                max_fraction=0.05,
                                initial_bankroll=10000):

        value_bets = self.find_value_bets(threshold=threshold)
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

            # fractional Kelly and cap
            stake_fraction = min(kelly_f * kelly_fraction, max_fraction)
            stake = stake_fraction * bankroll
            stakes.append(stake)

            profit = stake * (odds - 1) if row['correct'] else -stake
            bankroll += profit
            bankroll_history.append(bankroll)

        bets_with_stakes = value_bets.copy()
        bets_with_stakes['stake'] = stakes

        return np.array(bankroll_history), bets_with_stakes

    def simulate_bankroll(self, bets, initial_bankroll=10000, stake_fixed=100):
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
