import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataAnalyses:
    def __init__(self,label_encoder):
        self.label_encoder = label_encoder

    def analyze_probability_calibration(self, model, X, y_encoded, clean_data):
        """Check if model probabilities are better calibrated than market"""
        print("\n" + "=" * 60)
        print("\nProbability Calibration Analysis:")
        print("\n" + "=" * 60)

        self.label_encoder.fit(clean_data['Result'])

        # Get model predictions
        model_proba = model.predict_proba(X)
        class_labels = self.label_encoder.classes_
        model_df = pd.DataFrame(model_proba, columns=class_labels, index=X.index)

        # Convert encoded y back to original labels
        y_original = self.label_encoder.inverse_transform(y_encoded)

        print(f"Label Mapping: {list(zip(range(len(class_labels)), class_labels))}")

        home_idx = np.where(class_labels == 'H')[0][0] if 'H' in class_labels else None
        away_idx = np.where(class_labels == 'A')[0][0] if 'A' in class_labels else None
        draw_idx = np.where(class_labels == 'D')[0][0] if 'D' in class_labels else None

        outcomes = [
            ('AWAY', 'A', 'Market_Prob_Away', away_idx),
            ('DRAW', 'D', 'Market_Prob_Draw', draw_idx),
            ('HOME', 'H', 'Market_Prob_Home', home_idx)
        ]
        for outcome_name,outcome_label,market_prob_col,idx in outcomes:
            print(f"\n {outcome_name} WIN Calibration:")
            print("   Prob Range  | Market Pred | Market Actual | Model Pred | Model Actual | Better")
            print("   " + "-" * 75)

            if outcome_name == 'AWAY' or outcome_name == 'HOME':
                bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            elif outcome_name == 'DRAW':
                bins = [0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 1.0]

            total_market_error = 0
            total_model_error = 0

            for i in range(len(bins)-1):
                low,high = bins[i], bins[i+1]

                market_mask = (clean_data[market_prob_col] >= low) & (clean_data[market_prob_col] <= high)
                model_col = class_labels[idx]
                model_mask = (model_df[model_col]>low) & (model_df[model_col]<high)

                if market_mask.sum() > 10 and model_mask.sum() > 10:
                    market_actual = (y_original[market_mask] == outcome_label).mean()
                    market_pred = clean_data.loc[market_mask, market_prob_col].mean()

                    model_actual = (y_original[model_mask] == outcome_label).mean()
                    model_pred_val = model_df.loc[model_mask, model_col].mean()

                    market_error = abs(market_pred - market_actual)
                    model_error = abs(model_pred_val - model_actual)

                    total_market_error += market_error
                    total_model_error += model_error

                    better = "  BETTER" if model_error < market_error else ""
                    if abs(model_error - market_error) < 0.001:
                        better = "  TIE"

                    print(f"   {low:.2f}-{high:.2f}   | {market_pred:.3f}     | {market_actual:.3f}       | {model_pred_val:.3f}     | {model_actual:.3f}      {better}")

    def analyze_market_performance(self, clean_data):
        print("\n"+"=" * 60)
        print("\n Market Performance Analysis:")
        print("\n" + "=" * 60)

        market_features = sorted(['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home'])

        print(f"   Feature order: {market_features}")
        print(f"   Label encoding: {list(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        market_probs= clean_data[market_features].values
        market_pred = market_probs.argmax(axis=1)

        market_pred_labels = self.label_encoder.inverse_transform(market_pred)

        actual_results = clean_data['Result']

        # Check distribution
        print(
            f"   Market predictions: H={sum(market_pred_labels == 'H')}, D={sum(market_pred_labels == 'D')}, A={sum(market_pred_labels == 'A')}")
        print(
            f"   Actual results: H={sum(actual_results == 'H')}, D={sum(actual_results == 'D')}, A={sum(actual_results == 'A')}")

        # Market accuracy by outcome
        for outcome in ['A', 'D', 'H']:
            mask = market_pred_labels == outcome
            if mask.sum() > 0:
                accuracy = (actual_results[mask] == outcome).mean()
                print(f"   Market accuracy when predicting {outcome}: {accuracy:.3f}")

        # Check first few games to see the mapping
        print("   First 5 games market probabilities [Away, Draw, Home]:")
        for i in range(min(5, len(market_probs))):
            probs = market_probs[i]
            pred_idx = probs.argmax()
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            actual_result = clean_data.iloc[i]['Result']

            print(
                f"   Game {i}: Probs[A={probs[0]:.3f}, D={probs[1]:.3f}, H={probs[2]:.3f}] -> Pred: {pred_label}, Actual: {actual_result}")
        self.analyze_draw_market_pred(market_probs,market_pred_labels)


    def analyze_draw_market_pred(self,market_probs, market_pred_labels):
        print(f"\n🎯 Draw Prediction Analysis:")

        total_games = len(market_pred_labels)
        draw_predictions = (market_pred_labels == 'D').sum()
        print(f"   Draw predictions: {draw_predictions}/{total_games} ({draw_predictions / total_games:.1%})")

        # Check if draw probabilities are generally low
        draw_probs = market_probs[:, 1]  # Draw probabilities are at index 1
        avg_draw_prob = draw_probs.mean()
        max_draw_prob = draw_probs.max()

        print(f"   Average draw probability: {avg_draw_prob:.3f}")
        print(f"   Maximum draw probability: {max_draw_prob:.3f}")
        print(f"   Games with draw prob > 0.4: {(draw_probs > 0.4).sum()}")
        print(f"   Games with draw prob > 0.5: {(draw_probs > 0.5).sum()}")




