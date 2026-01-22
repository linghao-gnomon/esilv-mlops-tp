import os
from unittest.mock import patch

import pandas as pd
import pytest

from tests import _PATH_DATA
from src.data.load import load_local_data


DATA_PATH = os.path.join(_PATH_DATA, "titanic.csv")


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Training data file is missing",
)
def test_train_data_loading():
    """Ensure that the raw Titanic CSV file exists and has the expected columns."""
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0, "Training dataset is empty"
    assert "Survived" in df.columns, "Target column 'Survived' is missing"
    assert "Age" in df.columns
    assert "Fare" in df.columns


def test_load_local_data_uses_pandas_read_csv(tmp_path):
    """Test that load_local_data uses pandas.read_csv correctly."""
    fake_df = pd.DataFrame({"A": [1, 2, 3]})
    fake_path = tmp_path / "dummy.csv"

    with patch("src.data.load.pd.read_csv", return_value=fake_df) as mock_read_csv:
        df = load_local_data(fake_path)

    mock_read_csv.assert_called_once_with(fake_path)
    assert df.equals(fake_df)
