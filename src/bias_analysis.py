"""
Bias analysis module for election prediction analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from .config import PARTIES


def calculate_prediction_bias(predictions_df: pd.DataFrame,
                              results_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Calculate prediction bias for each party and respondent.

    Args:
        predictions_df: DataFrame with predictions
        results_df: DataFrame with actual results

    Returns:
        Tuple of (bias_df, party_bias_stats, respondent_bias_stats)
        - bias_df: DataFrame with signed errors (positive = overestimated, negative = underestimated)
        - party_bias_stats: Overall bias statistics by party
        - respondent_bias_stats: Individual bias patterns by respondent
    """
    # Get actual results
    actual_results = results_df.iloc[0] if len(results_df) == 1 else results_df

    # Calculate signed errors (predicted - actual)
    bias_data = []

    for idx, row in predictions_df.iterrows():
        bias_row = {'respondent': row['Eget navn']}

        for party in PARTIES:
            if party in actual_results:
                predicted = row[party]
                actual = actual_results[party]
                if pd.isna(predicted):
                    bias = np.nan
                else:
                    bias = predicted - actual  # Positive = overestimated, Negative = underestimated
                bias_row[party] = bias
            else:
                bias_row[party] = np.nan

        bias_data.append(bias_row)

    bias_df = pd.DataFrame(bias_data)

    # Calculate party bias statistics
    party_bias_stats = {}
    for party in PARTIES:
        if party in bias_df.columns:
            biases = bias_df[party].dropna()
            if len(biases) > 0:
                party_bias_stats[party] = {
                    'mean_bias': biases.mean(),
                    'median_bias': biases.median(),
                    'std_bias': biases.std(),
                    'overestimated_count': (biases > 0).sum(),
                    'underestimated_count': (biases < 0).sum(),
                    'perfect_count': (biases == 0).sum(),
                    'actual_result': actual_results[party] if party in actual_results else np.nan,
                    'n_predictions': len(biases)
                }

    # Calculate respondent bias patterns
    respondent_bias_stats = {}
    for idx, row in bias_df.iterrows():
        respondent = row['respondent']
        biases = row[PARTIES].dropna()

        if len(biases) > 0:
            respondent_bias_stats[respondent] = {
                'mean_bias': biases.mean(),
                'positive_bias_count': (biases > 0).sum(),
                'negative_bias_count': (biases < 0).sum(),
                'total_predictions': len(biases),
                'optimism_ratio': (biases > 0).sum() / len(biases) if len(biases) > 0 else 0
            }

    return bias_df, party_bias_stats, respondent_bias_stats


def get_bias_insights(bias_df: pd.DataFrame,
                      party_bias_stats: Dict[str, Dict[str, Any]],
                      respondent_bias_stats: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate detailed bias analysis insights.

    Args:
        bias_df: DataFrame with bias data
        party_bias_stats: Party bias statistics
        respondent_bias_stats: Respondent bias statistics

    Returns:
        String with bias insights
    """
    insights = "🎯 BIAS DETECTION INSIGHTS:\n\n"

    # Overall bias trends
    overall_bias = bias_df[PARTIES].mean().mean()

    if abs(overall_bias) < 0.1:
        insights += f"📊 OVERALL: Group predictions were remarkably unbiased (avg: {overall_bias:+.2f})\n"
    elif overall_bias > 0:
        insights += f"📈 OVERALL: Group tends to be optimistic/overestimate parties (avg: {overall_bias:+.1f})\n"
    else:
        insights += f"📉 OVERALL: Group tends to be pessimistic/underestimate parties (avg: {overall_bias:+.1f})\n"
    # Find most biased predictions
    most_overestimated = max(party_bias_stats.items(),
                             key=lambda x: x[1]['mean_bias'])
    most_underestimated = min(party_bias_stats.items(),
                              key=lambda x: x[1]['mean_bias'])

    insights += f"\n🔴 BIGGEST OVERESTIMATE: {most_overestimated[0]} "
    insights += f"({most_overestimated[1]['mean_bias']:+.1f} on average)\n"
    insights += f"   → {most_overestimated[1]['overestimated_count']}/{len(bias_df)} people overestimated it\n"

    insights += f"\n🔵 BIGGEST UNDERESTIMATE: {most_underestimated[0]} "
    insights += f"({most_underestimated[1]['mean_bias']:+.1f} on average)\n"
    insights += f"   → {most_underestimated[1]['underestimated_count']}/{len(bias_df)} people underestimated it\n"

    # Individual patterns
    most_optimistic = max(respondent_bias_stats.items(),
                          key=lambda x: x[1]['mean_bias'])
    most_pessimistic = min(respondent_bias_stats.items(),
                           key=lambda x: x[1]['mean_bias'])

    insights += f"\n😊 MOST OPTIMISTIC PERSON: {most_optimistic[0]} "
    insights += f"({most_optimistic[1]['mean_bias']:+.1f} avg bias)\n"

    insights += f"😔 MOST PESSIMISTIC PERSON: {most_pessimistic[0]} "
    insights += f"({most_pessimistic[1]['mean_bias']:+.1f} avg bias)\n"
    return insights
