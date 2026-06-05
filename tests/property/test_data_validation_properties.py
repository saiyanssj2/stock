"""
Property-based tests for data validation error reporting.

**Validates: Requirements 1.2, 1.4, 1.5**

Properties tested:
- Property 8: Data validation error reporting
  - Missing/malformed files correctly identified in error response
  - For any non-existent file path, DataError is raised with FILE_NOT_FOUND
  - For any CSV missing required columns, DataError is raised with MISSING_COLUMNS
    and error details include which columns are missing
  - For any CSV with insufficient rows, DataError is raised with INSUFFICIENT_DATA
  - Valid CSVs pass validation without error
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.config import DataError, EngineConfig
from engine.decision_engine import DecisionEngine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
MIN_DATA_ROWS = 5  # From EngineConfig default


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def nonexistent_file_path_strategy(draw):
    """
    Generate file paths that do not exist on the filesystem.

    Produces random path-like strings under a non-existent directory
    to guarantee the file does not exist.
    """
    # Use random alphanumeric segments for directory and filename
    dir_segment = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=3, max_size=20,
    ))
    filename = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=3, max_size=20,
    ))
    extension = draw(st.sampled_from([".csv", ".CSV", ".dat", ".txt"]))

    # Construct a path under a definitely non-existent root
    path = Path(tempfile.gettempdir()) / f"__nonexistent_{dir_segment}" / f"{filename}{extension}"
    # Ensure it truly doesn't exist
    assume(not path.exists())
    return path


@st.composite
def missing_columns_dataframe_strategy(draw):
    """
    Generate a DataFrame that is missing at least one required column.

    The resulting DataFrame will have some but not all of the required
    columns (time, open, high, low, close, volume). Always includes at
    least one column so the CSV file is readable by pandas.
    """
    # Decide how many required columns to include (at least 1 included, at least 1 missing)
    num_to_include = draw(st.integers(min_value=1, max_value=len(REQUIRED_COLUMNS) - 1))
    included_columns = draw(
        st.lists(
            st.sampled_from(REQUIRED_COLUMNS),
            min_size=num_to_include,
            max_size=num_to_include,
            unique=True,
        )
    )

    # Ensure at least one required column is missing
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in included_columns]
    assume(len(missing_columns) > 0)

    num_rows = draw(st.integers(min_value=MIN_DATA_ROWS, max_value=50))

    # Build DataFrame with only included columns
    data = {}
    for col in included_columns:
        if col == "time":
            data[col] = pd.bdate_range(end="2024-06-30", periods=num_rows)
        elif col == "volume":
            data[col] = np.random.uniform(100000, 5000000, num_rows)
        else:
            data[col] = np.random.uniform(10, 200, num_rows)

    # Optionally add some extra non-required columns
    num_extra = draw(st.integers(min_value=0, max_value=3))
    for i in range(num_extra):
        extra_name = f"extra_col_{i}"
        data[extra_name] = np.random.uniform(0, 100, num_rows)

    df = pd.DataFrame(data)
    return df, missing_columns


@st.composite
def insufficient_rows_dataframe_strategy(draw):
    """
    Generate a valid-column DataFrame with fewer rows than min_data_rows.

    The DataFrame has all required columns but insufficient rows (1 to 4).
    """
    num_rows = draw(st.integers(min_value=1, max_value=MIN_DATA_ROWS - 1))

    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    closes = base_price * np.cumprod(
        1.0 + np.random.uniform(-0.05, 0.05, num_rows)
    )
    highs = closes * 1.01
    lows = closes * 0.99
    opens = (highs + lows) / 2
    volumes = np.random.uniform(100000, 5000000, num_rows)
    dates = pd.bdate_range(end="2024-06-30", periods=num_rows)

    df = pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


@st.composite
def valid_csv_dataframe_strategy(draw):
    """
    Generate a valid DataFrame with all required columns and sufficient rows.

    This DataFrame should pass validation without errors.
    """
    num_rows = draw(st.integers(min_value=MIN_DATA_ROWS, max_value=100))

    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    returns = np.random.uniform(-0.05, 0.05, num_rows)
    returns[0] = 0.0
    closes = base_price * np.cumprod(1.0 + returns)
    highs = closes * 1.01
    lows = closes * 0.99
    opens = (highs + lows) / 2
    volumes = np.random.uniform(100000, 5000000, num_rows)
    dates = pd.bdate_range(end="2024-06-30", periods=num_rows)

    df = pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


# ---------------------------------------------------------------------------
# Property 8: Data validation error reporting
# ---------------------------------------------------------------------------


class TestDataValidationErrorReporting:
    """
    Property 8: Data validation error reporting.

    For any set of required CSV file paths where some files are missing or
    contain malformed data (missing required columns or insufficient rows),
    the Decision_Engine SHALL return an error that identifies the specific
    problematic files and their issues, and the set of reported files SHALL
    exactly match the set of actual problematic files.

    **Validates: Requirements 1.2, 1.4, 1.5**
    """

    @given(file_path=nonexistent_file_path_strategy())
    @settings(max_examples=100)
    def test_nonexistent_file_raises_file_not_found(self, file_path):
        """
        For any non-existent file path, DataError is raised with FILE_NOT_FOUND.

        The error details must include the path that was attempted.

        **Validates: Requirements 1.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DecisionEngine(base_dir=tmp_dir)

            with pytest.raises(DataError) as exc_info:
                engine.validate_csv(file_path)

            error = exc_info.value
            assert error.error_code == "FILE_NOT_FOUND"
            assert str(file_path) in error.details["path"]

    @given(data=missing_columns_dataframe_strategy())
    @settings(max_examples=100)
    def test_missing_columns_raises_missing_columns_error(self, data):
        """
        For any CSV missing required columns, DataError is raised with
        MISSING_COLUMNS and the error details include which columns are missing.

        **Validates: Requirements 1.5**
        """
        df, expected_missing = data

        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DecisionEngine(base_dir=tmp_dir)

            # Write the DataFrame to a CSV file
            csv_path = Path(tmp_dir) / "test_data.csv"
            df.to_csv(csv_path, index=False)

            with pytest.raises(DataError) as exc_info:
                engine.validate_csv(csv_path)

            error = exc_info.value
            assert error.error_code == "MISSING_COLUMNS"

            # Verify the missing columns are correctly reported
            reported_missing = error.details["missing_columns"]
            assert set(reported_missing) == set(expected_missing), (
                f"Expected missing columns {set(expected_missing)}, "
                f"but got {set(reported_missing)}"
            )

    @given(df=insufficient_rows_dataframe_strategy())
    @settings(max_examples=100)
    def test_insufficient_rows_raises_insufficient_data_error(self, df):
        """
        For any CSV with insufficient rows (< min_data_rows), DataError is
        raised with INSUFFICIENT_DATA.

        The error details must include the actual row count and the minimum
        required.

        **Validates: Requirements 1.2**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DecisionEngine(base_dir=tmp_dir)

            # Write the DataFrame to a CSV file
            csv_path = Path(tmp_dir) / "test_data.csv"
            df.to_csv(csv_path, index=False)

            with pytest.raises(DataError) as exc_info:
                engine.validate_csv(csv_path)

            error = exc_info.value
            assert error.error_code == "INSUFFICIENT_DATA"
            assert error.details["rows"] == len(df)
            assert error.details["min_required"] == MIN_DATA_ROWS

    @given(df=valid_csv_dataframe_strategy())
    @settings(max_examples=100)
    def test_valid_csv_passes_validation(self, df):
        """
        Valid CSVs (all required columns, sufficient rows) pass validation
        without raising any error.

        **Validates: Requirements 1.2**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DecisionEngine(base_dir=tmp_dir)

            # Write the DataFrame to a CSV file
            csv_path = Path(tmp_dir) / "test_data.csv"
            df.to_csv(csv_path, index=False)

            # Should not raise any exception
            engine.validate_csv(csv_path)

    @given(data=st.data())
    @settings(max_examples=50)
    def test_validate_multiple_csvs_reports_all_problematic_files(self, data):
        """
        When validating multiple CSV files, the set of reported problematic
        files exactly matches the set of actual problematic files.

        **Validates: Requirements 1.4, 1.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DecisionEngine(base_dir=tmp_dir)

            # Create a mix of valid and invalid files
            num_valid = data.draw(st.integers(min_value=1, max_value=3))
            num_invalid = data.draw(st.integers(min_value=1, max_value=3))

            file_paths = []
            expected_invalid_paths = set()

            # Create valid files
            for i in range(num_valid):
                num_rows = data.draw(st.integers(min_value=MIN_DATA_ROWS, max_value=20))
                closes = np.random.uniform(10, 200, num_rows)
                df = pd.DataFrame({
                    "time": pd.bdate_range(end="2024-06-30", periods=num_rows),
                    "open": closes * 0.99,
                    "high": closes * 1.01,
                    "low": closes * 0.98,
                    "close": closes,
                    "volume": np.random.uniform(100000, 5000000, num_rows),
                })
                path = Path(tmp_dir) / f"valid_{i}.csv"
                df.to_csv(path, index=False)
                file_paths.append(path)

            # Create invalid files (mix of missing and nonexistent)
            for i in range(num_invalid):
                invalid_type = data.draw(st.sampled_from(["nonexistent", "missing_cols", "insufficient_rows"]))

                if invalid_type == "nonexistent":
                    path = Path(tmp_dir) / f"nonexistent_{i}.csv"
                    # Don't create the file
                elif invalid_type == "missing_cols":
                    # Create file with missing columns
                    num_rows = data.draw(st.integers(min_value=MIN_DATA_ROWS, max_value=20))
                    df = pd.DataFrame({
                        "time": pd.bdate_range(end="2024-06-30", periods=num_rows),
                        "open": np.random.uniform(10, 200, num_rows),
                        # Missing: high, low, close, volume
                    })
                    path = Path(tmp_dir) / f"malformed_{i}.csv"
                    df.to_csv(path, index=False)
                else:  # insufficient_rows
                    num_rows = data.draw(st.integers(min_value=1, max_value=MIN_DATA_ROWS - 1))
                    closes = np.random.uniform(10, 200, num_rows)
                    df = pd.DataFrame({
                        "time": pd.bdate_range(end="2024-06-30", periods=num_rows),
                        "open": closes * 0.99,
                        "high": closes * 1.01,
                        "low": closes * 0.98,
                        "close": closes,
                        "volume": np.random.uniform(100000, 5000000, num_rows),
                    })
                    path = Path(tmp_dir) / f"insufficient_{i}.csv"
                    df.to_csv(path, index=False)

                file_paths.append(path)
                expected_invalid_paths.add(str(path))

            # Validate all files
            errors = engine.validate_multiple_csvs(file_paths)

            # The set of reported problematic files should exactly match expected
            reported_paths = set(errors.keys())
            assert reported_paths == expected_invalid_paths, (
                f"Expected problematic files: {expected_invalid_paths}, "
                f"but got: {reported_paths}"
            )

    @given(file_path=nonexistent_file_path_strategy())
    @settings(max_examples=50)
    def test_file_not_found_error_includes_suggestion(self, file_path):
        """
        FILE_NOT_FOUND error details include a suggestion for the user.

        **Validates: Requirements 1.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DecisionEngine(base_dir=tmp_dir)

            with pytest.raises(DataError) as exc_info:
                engine.validate_csv(file_path)

            error = exc_info.value
            assert error.error_code == "FILE_NOT_FOUND"
            # Error should include a suggestion to help the user
            assert "suggestion" in error.details

    @given(data=missing_columns_dataframe_strategy())
    @settings(max_examples=50)
    def test_missing_columns_error_reports_available_columns(self, data):
        """
        MISSING_COLUMNS error details include what columns ARE available.

        **Validates: Requirements 1.5**
        """
        df, _ = data

        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DecisionEngine(base_dir=tmp_dir)

            csv_path = Path(tmp_dir) / "test_data.csv"
            df.to_csv(csv_path, index=False)

            with pytest.raises(DataError) as exc_info:
                engine.validate_csv(csv_path)

            error = exc_info.value
            assert error.error_code == "MISSING_COLUMNS"
            # Should report what columns are available
            assert "available_columns" in error.details
            assert set(error.details["available_columns"]) == set(df.columns)
