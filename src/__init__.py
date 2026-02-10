"""
Election Prediction Analysis Package

A comprehensive toolkit for analyzing election prediction data,
including error analysis, bias detection, and clustering of prediction patterns.
"""

__version__ = "1.0.0"
__author__ = "Roar Election Analysis Team"
__description__ = "Analysis toolkit for election predictions"

from .config import PARTIES, SYNTHETIC_RESULTS
from .data_loader import load_and_prepare_data
from .error_analysis import calculate_total_errors, calculate_party_errors
from .bias_analysis import calculate_prediction_bias
from .clustering import perform_clustering, analyze_cluster_characteristics
from .visualization import create_error_distribution_plots, create_party_analysis_plots
from .main import main

__all__ = [
    # Configuration
    'PARTIES',
    'SYNTHETIC_RESULTS',

    # Data loading
    'load_and_prepare_data',

    # Error analysis
    'calculate_total_errors',
    'calculate_party_errors',

    # Bias analysis
    'calculate_prediction_bias',

    # Clustering
    'perform_clustering',
    'analyze_cluster_characteristics',

    # Visualization
    'create_error_distribution_plots',
    'create_party_analysis_plots',

    # Main function
    'main'
]
