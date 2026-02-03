import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def titanic_sample():
    """
    Small toy Titanic dataset used in multiple tests.
    
    Includes extra columns to ensure tests remain robust when feature engineering
    is added that might require additional columns from the raw dataset.
    """
    X = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Pclass": [3, 1, 3, 1],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques Heath",
            ],
            "Sex": ["male", "female", "female", "female"],
            "Age": [22, 38, 26, 35],
            "SibSp": [1, 1, 0, 1],
            "Parch": [0, 0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803"],
            "Fare": [7.25, 71.28, 7.92, 53.1],
            "Cabin": [np.nan, "C85", np.nan, "C123"],
            "Embarked": ["S", "C", "S", "S"],
        }
    )
    y = np.array([0, 1, 1, 1])
    return X, y
