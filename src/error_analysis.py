"""
Error analysis module for election prediction analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from .config import PARTIES


def calculate_total_errors(predictions_df: pd.DataFrame,
                           results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total absolute error for each respondent.

    Args:
        predictions_df: DataFrame with columns ['Eget navn', party1, party2, ...]
        results_df: DataFrame with party names as columns and actual results as values

    Returns:
        DataFrame with respondent and total_error columns
    """
    # Get actual results
    actual_results = results_df.iloc[0] if len(results_df) == 1 else results_df

    # Initialize results
    error_results = []

    for idx, row in predictions_df.iterrows():
        respondent = row['Eget navn']
        total_error = 0

        for party in PARTIES:
            if party in actual_results:
                predicted = row[party]
                actual = actual_results[party]
                # Skip NaN predictions
                if not pd.isna(predicted):
                    error = abs(predicted - actual)
                    total_error += error

        error_results.append({
            'respondent': respondent,
            'total_error': total_error
        })

    error_df = pd.DataFrame(error_results)
    error_df = error_df.set_index('respondent')
    return error_df


def calculate_party_errors(predictions_df: pd.DataFrame,
                           results_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Calculate individual party errors for each respondent.

    Args:
        predictions_df: DataFrame with predictions
        results_df: DataFrame with actual results

    Returns:
        Tuple of (party_errors_df, party_stats)
        - party_errors_df: DataFrame with columns ['respondent'] + party columns (containing absolute errors)
        - party_stats: Summary statistics for each party
    """
    # Get actual results
    actual_results = results_df.iloc[0] if len(results_df) == 1 else results_df

    # Initialize results
    party_errors = []

    for idx, row in predictions_df.iterrows():
        error_row = {'respondent': row['Eget navn']}

        for party in PARTIES:
            if party in actual_results:
                predicted = row[party]
                actual = actual_results[party]
                if pd.isna(predicted):
                    error = np.nan
                else:
                    error = abs(predicted - actual)
                error_row[party] = error
            else:
                error_row[party] = np.nan

        party_errors.append(error_row)

    party_errors_df = pd.DataFrame(party_errors)

    # Calculate summary statistics for each party
    party_stats = {}
    for party in PARTIES:
        if party in party_errors_df.columns:
            errors = party_errors_df[party].dropna()
            if len(errors) > 0:
                party_stats[party] = {
                    'mean_error': errors.mean(),
                    'median_error': errors.median(),
                    'std_error': errors.std(),
                    'min_error': errors.min(),
                    'max_error': errors.max(),
                    'actual_result': actual_results[party] if party in actual_results else np.nan,
                    'n_predictions': len(errors)
                }

    return party_errors_df, party_stats


def get_error_summary(error_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for total errors.

    Args:
        error_df: DataFrame with total errors

    Returns:
        Dictionary with summary statistics
    """
    errors = error_df['total_error']

    return {
        'count': len(errors),
        'mean': errors.mean(),
        'median': errors.median(),
        'std': errors.std(),
        'min': errors.min(),
        'max': errors.max(),
        'range': errors.max() - errors.min(),
        'best_performer': error_df.loc[error_df['total_error'].idxmin()].name,
        'worst_performer': error_df.loc[error_df['total_error'].idxmax()].name
    }


def get_party_error_insights(party_errors_df: pd.DataFrame,
                             party_stats: Dict[str, Dict[str, float]]) -> str:
    """
    Generate text insights about party prediction patterns.

    Args:
        party_errors_df: DataFrame with party errors
        party_stats: Party statistics

    Returns:
        String with insights
    """
    # Find hardest and easiest parties
    party_difficulties = [(party, stats['mean_error'])
                          for party, stats in party_stats.items()]
    party_difficulties.sort(key=lambda x: x[1])

    easiest_party = party_difficulties[0]
    hardest_party = party_difficulties[-1]

    # Find most consistent/inconsistent parties
    party_consistency = [(party, stats['std_error'])
                         for party, stats in party_stats.items()]
    party_consistency.sort(key=lambda x: x[1])

    most_consistent = party_consistency[0]
    least_consistent = party_consistency[-1]

    # Find who was best/worst at hardest party
    hardest_party_name = hardest_party[0]
    hardest_party_errors = party_errors_df[[
        'respondent', hardest_party_name]].copy()
    best_at_hardest = hardest_party_errors.loc[hardest_party_errors[hardest_party_name].idxmin(
    )]
    worst_at_hardest = hardest_party_errors.loc[hardest_party_errors[hardest_party_name].idxmax(
    )]

    insights = ".1f"".1f"".1f"".1f"".1f"".1f"f"""
PARTY PREDICTION INSIGHTS:

🎯 EASIEST TO PREDICT: {easiest_party[0]} (avg error: {easiest_party[1]:.1f})
😰 HARDEST TO PREDICT: {hardest_party[0]} (avg error: {hardest_party[1]:.1f})

📊 MOST CONSISTENT PREDICTIONS: {most_consistent[0]} (std: {most_consistent[1]:.1f})
📈 MOST INCONSISTENT PREDICTIONS: {least_consistent[0]} (std: {least_consistent[1]:.1f})

🏆 BEST AT HARDEST PARTY ({hardest_party_name}): {best_at_hardest['respondent']} ({best_at_hardest[hardest_party_name]:.1f} error)
🤦 WORST AT HARDEST PARTY ({hardest_party_name}): {worst_at_hardest['respondent']} ({worst_at_hardest[hardest_party_name]:.1f} error)

DIFFICULTY RANKING (hardest first):
"""

    party_difficulties.reverse()  # Hardest first
    for i, (party, avg_error) in enumerate(party_difficulties, 1):
        actual_pct = party_stats[party]['actual_result']
        insights += f"{i}. {party}: {avg_error:.1f} avg error (actual: {actual_pct:.1f}%)\n"

    return insights
