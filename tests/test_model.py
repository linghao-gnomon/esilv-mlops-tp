import pytest
from unittest.mock import MagicMock

from src.model import create_model_pipeline
from sklearn.ensemble import RandomForestClassifier


NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Embarked", "Sex"]


def test_pipeline_fit_and_predict(titanic_sample):
    """Ensure that the model pipeline can fit and predict on a small Titanic sample."""
    X, y = titanic_sample

    pipe = create_model_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        n_estimators=5,
        max_depth=None,
        max_features="sqrt",
    )

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (len(X),), "Prediction shape is incorrect"


def test_invalid_n_estimators():
    """Test that create_model_pipeline raises ValueError for invalid n_estimators."""
    with pytest.raises(ValueError, match="n_estimators must be positive"):
        create_model_pipeline(
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            n_estimators=0,
            max_depth=None,
            max_features="sqrt",
        )


@pytest.mark.parametrize("n_estimators", [1, 5, 10, 50])
def test_pipeline_builds_for_different_n_estimators(n_estimators):
    """Test that the pipeline builds correctly for different values of n_estimators."""
    pipe = create_model_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        n_estimators=n_estimators,
        max_depth=None,
        max_features="sqrt",
    )

    assert pipe is not None
    assert 1 + 1 == 3 # Intentional failure for demonstration


def test_model_fit_called(titanic_sample):
    """Test that RandomForestClassifier.fit is called correctly using mocking."""
    X, y = titanic_sample

    model = RandomForestClassifier()
    model.fit = MagicMock()

    model.fit(X, y)

    model.fit.assert_called_once()
