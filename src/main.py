"""
Main module for election prediction analysis.
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt

from .data_loader import load_and_prepare_data
from .error_analysis import calculate_total_errors, calculate_party_errors, get_error_summary, get_party_error_insights
from .bias_analysis import calculate_prediction_bias, get_bias_insights
from .clustering import (prepare_clustering_data, perform_clustering, calculate_similarity_network,
                         analyze_cluster_characteristics, get_clustering_insights, find_political_archetypes)
from .visualization import (create_error_distribution_plots, create_party_analysis_plots,
                            create_bias_detection_plots, create_political_clustering_plots)
from .config import RESULTS_FILE, ERROR_RESULTS_FILE, DEFAULT_N_CLUSTERS, PARTIES, DATA_FILE

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_error_analysis(predictions_df, results_df, show_plots: bool = True):
    """Run comprehensive error analysis."""
    logger.info("Running error analysis...")

    # Calculate total errors
    error_df = calculate_total_errors(predictions_df, results_df)
    logger.info(f"Calculated total errors for {len(error_df)} respondents")

    # Calculate party-specific errors
    party_errors_df, party_stats = calculate_party_errors(
        predictions_df, results_df)
    logger.info(f"Calculated party errors for {len(party_stats)} parties")

    # Get insights
    error_summary = get_error_summary(error_df)
    party_insights = get_party_error_insights(party_errors_df, party_stats)

    # Print results
    print("\n" + "="*50)
    print("ERROR ANALYSIS RESULTS")
    print("="*50)
    print(f"Total respondents: {error_summary['count']}")
    print(f"Mean error: {error_summary['mean']:.2f}")
    print(
        f"Best performer: {error_summary['best_performer']} ({error_summary['min']:.2f})")
    print(
        f"Worst performer: {error_summary['worst_performer']} ({error_summary['max']:.2f})")

    print("\nParty Insights:")
    print(party_insights)

    # Create plots
    if show_plots:
        logger.info("Creating error distribution plots...")
        fig = create_error_distribution_plots(error_df)
        fig.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info("Creating party analysis plots...")
        fig, _ = create_party_analysis_plots(party_errors_df, party_stats)
        fig.savefig('party_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    return error_df, party_errors_df, party_stats


def run_bias_analysis(predictions_df, results_df, show_plots: bool = True):
    """Run comprehensive bias analysis."""
    logger.info("Running bias analysis...")

    # Calculate bias
    bias_df, party_bias_stats, respondent_bias_stats = calculate_prediction_bias(
        predictions_df, results_df)

    # Get insights
    bias_insights = get_bias_insights(
        bias_df, party_bias_stats, respondent_bias_stats)

    # Print results
    print("\n" + "="*50)
    print("BIAS ANALYSIS RESULTS")
    print("="*50)
    print(bias_insights)

    # Create plots
    if show_plots:
        logger.info("Creating bias detection plots...")
        fig = create_bias_detection_plots(bias_df, party_bias_stats, respondent_bias_stats,
                                          predictions_df, results_df)
        fig.savefig('bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    return bias_df, party_bias_stats, respondent_bias_stats


def run_clustering_analysis(predictions_df, results_df, n_clusters: int = DEFAULT_N_CLUSTERS, show_plots: bool = True):
    """Run comprehensive clustering analysis."""
    logger.info(f"Running clustering analysis with {n_clusters} clusters...")

    # Perform clustering
    fig, clustering_results, network, similarity_matrix, correlation_matrix = create_political_clustering_plots(
        predictions_df, n_clusters)

    # Analyze clusters
    cluster_analysis = analyze_cluster_characteristics(
        predictions_df, clustering_results, PARTIES)

    # Get insights
    respondent_names = predictions_df['Eget navn'].values
    clustering_insights = get_clustering_insights(predictions_df, clustering_results,
                                                  cluster_analysis, correlation_matrix, respondent_names)

    # Find archetypes
    archetypes = find_political_archetypes(cluster_analysis, results_df)

    # Print results
    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS RESULTS")
    print("="*50)
    print(clustering_insights)

    print("\nPolitical Archetypes:")
    for cluster_id, archetype in archetypes.items():
        members = cluster_analysis[cluster_id]['members']
        print(f"Cluster {cluster_id+1}: {archetype} - {', '.join(members)}")

    # Save plots
    if show_plots:
        logger.info("Creating clustering plots...")
        fig.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    return clustering_results, cluster_analysis, network, similarity_matrix, correlation_matrix


def save_results(error_df, results_df):
    """Save analysis results to CSV files."""
    logger.info("Saving results to CSV files...")

    # Save results
    results_df.to_csv(RESULTS_FILE, index=False)
    error_df.to_csv(ERROR_RESULTS_FILE, index=True)

    logger.info(f"Results saved to {RESULTS_FILE} and {ERROR_RESULTS_FILE}")


def main(data_file: str = DATA_FILE,
         results: Dict[str, float] = None,
         n_clusters: int = DEFAULT_N_CLUSTERS,
         show_plots: bool = True,
         save_csv: bool = True):
    """
    Main function to run the complete election prediction analysis.

    Args:
        data_file: Path to predictions CSV file
        results: Dictionary with actual results (if None, uses synthetic)
        n_clusters: Number of clusters for analysis
        show_plots: Whether to create and save plots
        save_csv: Whether to save results to CSV
    """
    try:
        logger.info("Starting election prediction analysis...")

        # Load and prepare data
        predictions_df, results_df = load_and_prepare_data(data_file, results)

        # Run error analysis
        error_df, party_errors_df, party_stats = run_error_analysis(
            predictions_df, results_df, show_plots)

        # Run bias analysis
        bias_df, party_bias_stats, respondent_bias_stats = run_bias_analysis(
            predictions_df, results_df, show_plots)

        # Run clustering analysis
        clustering_results, cluster_analysis, network, similarity_matrix, correlation_matrix = run_clustering_analysis(
            predictions_df, results_df, n_clusters, show_plots)

        # Save results
        if save_csv:
            save_results(error_df, results_df)

        logger.info("Analysis complete!")

        return {
            'predictions': predictions_df,
            'results': results_df,
            'error_df': error_df,
            'party_errors': party_errors_df,
            'party_stats': party_stats,
            'bias_df': bias_df,
            'party_bias_stats': party_bias_stats,
            'respondent_bias_stats': respondent_bias_stats,
            'clustering_results': clustering_results,
            'cluster_analysis': cluster_analysis,
            'similarity_matrix': similarity_matrix,
            'correlation_matrix': correlation_matrix
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Election Prediction Analysis")
    parser.add_argument("--data-file", type=str,
                        help="Path to predictions CSV file")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS,
                        help="Number of clusters for analysis")
    parser.add_argument("--no-plots", action="store_true",
                        help="Don't create plots")
    parser.add_argument("--no-csv", action="store_true",
                        help="Don't save CSV results")

    args = parser.parse_args()

    main(
        data_file=args.data_file,
        n_clusters=args.n_clusters,
        show_plots=not args.no_plots,
        save_csv=not args.no_csv
    )
