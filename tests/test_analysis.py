"""
Unit tests for election prediction analysis package.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.config import PARTIES, SYNTHETIC_RESULTS
from src.data_loader import replace_str_to_float, create_results_dataframe, validate_data
from src.error_analysis import calculate_total_errors, calculate_party_errors


class TestDataLoader:
    """Test data loading functionality."""

    def test_replace_str_to_float(self):
        """Test string to float conversion."""
        df = pd.DataFrame({'party': ['28,5', '30,2', '25,8']})
        result = replace_str_to_float(df, 'party')
        expected = pd.Series([28.5, 30.2, 25.8], name='party')
        pd.testing.assert_series_equal(result, expected)

    def test_create_results_dataframe(self):
        """Test results DataFrame creation."""
        results = {'Party A': 30.0, 'Party B': 25.0}
        df = create_results_dataframe(results, validate_all_parties=False)
        assert len(df) == 1
        assert df.loc[0, 'Party A'] == 30.0
        assert df.loc[0, 'Party B'] == 25.0

    def test_create_results_dataframe_synthetic(self):
        """Test synthetic results DataFrame creation."""
        df = create_results_dataframe()
        assert len(df) == 1
        for party in PARTIES:
            assert party in df.columns
            assert df.loc[0, party] == SYNTHETIC_RESULTS[party]


class TestErrorAnalysis:
    """Test error analysis functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample prediction and results data."""
        predictions = pd.DataFrame({
            'Eget navn': ['Person A', 'Person B', 'Person C'],
            'Arbeiderpartiet': [25.0, 30.0, 28.0],
            'Høyre': [20.0, 15.0, 18.0]
        })

        results = pd.DataFrame({
            'Arbeiderpartiet': [27.0],
            'Høyre': [17.0]
        })

        return predictions, results

    def test_calculate_total_errors(self, sample_data):
        """Test total error calculation."""
        predictions, results = sample_data
        error_df = calculate_total_errors(predictions, results)

        assert len(error_df) == 3
        assert 'total_error' in error_df.columns
        # |25-27| + |20-17| = 2 + 3 = 5
        assert error_df.loc['Person A', 'total_error'] == 5.0
        # |30-27| + |15-17| = 3 + 2 = 5
        assert error_df.loc['Person B', 'total_error'] == 5.0
        # |28-27| + |18-17| = 1 + 1 = 2
        assert error_df.loc['Person C', 'total_error'] == 2.0

    def test_calculate_party_errors(self, sample_data):
        """Test party-specific error calculation."""
        predictions, results = sample_data
        party_errors_df, party_stats = calculate_party_errors(
            predictions, results)

        assert len(party_errors_df) == 3
        assert 'Arbeiderpartiet' in party_errors_df.columns
        assert 'Høyre' in party_errors_df.columns

        # Check specific errors
        assert party_errors_df.loc[0, 'Arbeiderpartiet'] == 2.0  # |25-27|
        assert party_errors_df.loc[1, 'Høyre'] == 2.0  # |15-17|

        # Check statistics
        assert 'Arbeiderpartiet' in party_stats
        assert 'Høyre' in party_stats
        assert party_stats['Arbeiderpartiet']['mean_error'] == 2.0  # (2+3+1)/3


class TestValidation:
    """Test data validation."""

    def test_validate_data_success(self):
        """Test successful data validation."""
        predictions = pd.DataFrame({
            'Eget navn': ['A', 'B'],
            'Arbeiderpartiet': [25.0, 30.0],
            'Høyre': [20.0, 15.0]
        })

        results = pd.DataFrame({
            'Arbeiderpartiet': [27.0],
            'Høyre': [17.0]
        })

        # Should not raise exception
        validate_data(predictions, results)

    def test_validate_data_missing_party(self):
        """Test validation with missing party in results."""
        predictions = pd.DataFrame({
            'Eget navn': ['A', 'B'],
            'Arbeiderpartiet': [25.0, 30.0],
            'Høyre': [20.0, 15.0]
        })

        results = pd.DataFrame({
            'Arbeiderpartiet': [27.0]
            # Missing Høyre
        })

        with pytest.raises(ValueError, match="Results missing for parties"):
            validate_data(predictions, results)


if __name__ == "__main__":
    pytest.main([__file__])
