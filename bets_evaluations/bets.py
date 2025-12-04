import numpy as np
import pandas as pd


def find_value_bets(calibrated_model, X_test, clean_data, threshold=0.10):
    # Model probabilities
    model_probs = calibrated_model.predict_proba(X_test)

    # Market probabilities
    market_probs = clean_data[['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home']].values

    edges = model_probs - market_probs
    max_edges = edges.max(axis=1)

    selections = edges.argmax(axis=1)

    value_bets = []
    for i, edge in enumerate(max_edges):
        if edge >= threshold:
            selection = selections[i]
            outcome = clean_data['y'].iloc[i]

            odds_col = ['Market_Prob_Away', 'Market_Prob_Draw', 'Market_Prob_Home'][selection]
            odds = 1 / clean_data[odds_col].iloc[i]

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


def predict_match(label_encoder, clf, home_team, away_team, home_odds, draw_odds, away_odds, home_strength,
                  away_strength,
                  feature_names):
    all_features = {

        'Home_Avg_Goals_For': home_strength['Avg_Goals_For'],
        'Home_Avg_Goals_Against': home_strength['Avg_Goals_Against'],
        'Home_Avg_Points': home_strength['Avg_Points'],
        'Home_Form': home_strength['Form'],
        'Away_Avg_Goals_For': away_strength['Avg_Goals_For'],
        'Away_Avg_Goals_Against': away_strength['Avg_Goals_Against'],
        'Away_Avg_Points': away_strength['Avg_Points'],
        'Away_Form': away_strength['Form'],

        'Home_Attack_Strength': home_strength['Avg_Goals_For'] / (away_strength['Avg_Goals_Against'] + 0.1),

        'Away_Attack_Strength': away_strength['Avg_Goals_For'] / (home_strength['Avg_Goals_Against'] + 0.1),

        'Strength_Difference': home_strength['Avg_Points'] - away_strength['Avg_Points'],
        'Goals_Ratio': home_strength['Avg_Goals_For'] / (away_strength['Avg_Goals_For'] + 0.1),
        'Form_Ratio': home_strength['Form'] / (away_strength['Form'] + 0.1),
        'Home_Advantage_x_Form': home_strength['Form'] * 1.5,
        'Away_Disadvantage_x_Form': away_strength['Form'] * 0.8,
        'Points_Difference': home_strength['Avg_Points'] - away_strength['Avg_Points'],
        'Is_Strong_vs_Weak': 1.0 if (home_strength['Avg_Points'] - away_strength['Avg_Points']) > 1.0 else 0.0,
        'Is_Even_Contest': 1.0 if abs(home_strength['Avg_Points'] - away_strength['Avg_Points']) <= 0.5 else 0.0,
        'Home_Goal_Difference': home_strength['Avg_Goals_For'] - home_strength['Avg_Goals_Against'],
        'Away_Goal_Difference': away_strength['Avg_Goals_For'] - away_strength['Avg_Goals_Against'],
        'Goal_Diff_Advantage': (home_strength['Avg_Goals_For'] - home_strength['Avg_Goals_Against']) -
                               (away_strength['Avg_Goals_For'] - away_strength['Avg_Goals_Against'])

    }
    feature_values = [all_features.get(f, 1.0) for f in feature_names]  # To ensure the right order
    X_match = np.array([feature_values])

    # Predict probabilities and class
    probs = clf.predict_proba(X_match)[0]
    predicted_class = clf.predict(X_match)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    model_probs = {
        'Home': probs[2],
        'Draw': probs[1],
        'Away': probs[0]
    }

    total = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)

    # Removing the vig
    market_probs = {
        'Home': (1 / home_odds) / total,
        'Draw': (1 / draw_odds) / total,
        'Away': (1 / away_odds) / total
    }

    edge_home = model_probs['Home'] - market_probs['Home']
    edge_draw = model_probs['Draw'] - market_probs['Draw']
    edge_away = model_probs['Away'] - market_probs['Away']

    edges = {
        'Home': edge_home,
        'Draw': edge_draw,
        'Away': edge_away
    }

    best_outcome = max(edges, key=edges.get)
    best_edge = edges[best_outcome]

    is_value = best_edge > 0

    return {
        "match": f"{home_team} vs {away_team}",
        "predicted_result": predicted_label,
        "probabilities": model_probs,
        "confidence": max(model_probs.values()),
        "market_probs": market_probs,
        "edges": edges,
        "best_value_bet": best_outcome if is_value else None,
        "best_edge": best_edge,
        "is_value_bet": is_value
    }
