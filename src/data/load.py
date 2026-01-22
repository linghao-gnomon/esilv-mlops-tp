from pathlib import Path
from typing import Union

import pandas as pd


def load_local_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file (string or Path object).

    Returns:
        DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)
