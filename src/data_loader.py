"""
Data loading and preprocessing module for election prediction analysis.
"""

import pandas as pd
import logging
from typing import Tuple, Dict, Any

from .config import PARTIES, SYNTHETIC_RESULTS, DATA_FILE, SKIP_ROWS, TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)


def replace_str_to_float(df: pd.DataFrame, party: str) -> pd.Series:
    """
    Convert string percentage values to float.

    Args:
        df: DataFrame containing the data
        party: Party column name

    Returns:
        Series with float values
    """
    return df[party].str.replace(',', '.').astype('float')


def load_prediction_data(file_path: str = DATA_FILE) -> pd.DataFrame:
    """
    Load prediction data from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with prediction data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(file_path, skiprows=SKIP_ROWS, header='infer')
        logger.info(f"Loaded {len(df)} predictions from {file_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

    # Validate required columns
    required_cols = ['Tidsmerke', 'Eget navn'] + PARTIES
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert timestamp
    try:
        df['Tidsmerke'] = pd.to_datetime(
            df['Tidsmerke'], format=TIMESTAMP_FORMAT)
    except Exception as e:
        logger.warning(f"Could not parse timestamps: {e}")

    # Select only the columns we need (timestamp, name, and parties)
    columns_to_keep = ['Tidsmerke', 'Eget navn'] + PARTIES
    df = df[columns_to_keep]

    # Convert party predictions to float
    for party in PARTIES:
        df[party] = replace_str_to_float(df, party)

    return df


def create_results_dataframe(results: Dict[str, float] = None, validate_all_parties: bool = True) -> pd.DataFrame:
    """
    Create a DataFrame from election results.

    Args:
        results: Dictionary of party results. If None, uses synthetic results.
        validate_all_parties: Whether to validate that all configured parties are present.

    Returns:
        DataFrame with results
    """
    if results is None:
        results = SYNTHETIC_RESULTS

    # Validate that all parties have results (skip for testing with partial data)
    if validate_all_parties:
        missing_results = [party for party in PARTIES if party not in results]
        if missing_results:
            raise ValueError(f"Missing results for parties: {missing_results}")

    return pd.DataFrame([results])


def validate_data(predictions_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """
    Validate that prediction and results data are compatible.

    Args:
        predictions_df: DataFrame with predictions
        results_df: DataFrame with actual results

    Raises:
        ValueError: If data is incompatible
    """
    # Check that all parties in predictions have results
    pred_parties = set(predictions_df.columns) - \
        {'Eget navn', 'Tidsmerke', 'E-postadresse'}
    result_parties = set(results_df.columns)

    if not pred_parties.issubset(result_parties):
        missing = pred_parties - result_parties
        raise ValueError(f"Results missing for parties: {missing}")

    # Check for missing predictions
    for party in pred_parties:
        if predictions_df[party].isnull().any():
            n_missing = predictions_df[party].isnull().sum()
            logger.warning(f"{n_missing} missing predictions for {party}")

    logger.info("Data validation passed")


def load_and_prepare_data(data_file: str = DATA_FILE,
                          results: Dict[str, float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare both prediction and results data.

    Args:
        data_file: Path to predictions CSV
        results: Actual results dictionary

    Returns:
        Tuple of (predictions_df, results_df)
    """
    logger.info("Loading prediction data...")
    predictions_df = load_prediction_data(data_file)

    logger.info("Creating results data...")
    results_df = create_results_dataframe(results)

    logger.info("Validating data...")
    validate_data(predictions_df, results_df)

    return predictions_df, results_df
