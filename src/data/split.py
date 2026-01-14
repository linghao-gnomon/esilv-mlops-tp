from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        data: Input DataFrame.
        target_column: Name of the target column.
        test_size: Proportion of dataset to include in the test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing (X_train, X_test, y_train, y_test).
    """
    y = data[target_column]
    X = data.drop(target_column, axis="columns")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
