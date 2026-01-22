import pytest

from src.model import create_model_pipeline


NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Embarked", "Sex"]


def train_and_evaluate(X, y, n_trees: int) -> float:
    """Build a pipeline, fit it on (X, y) and return the accuracy score."""
    pipe = create_model_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        n_estimators=n_trees,
        max_depth=None,
        max_features="sqrt",
    )
    pipe.fit(X, y)
    return pipe.score(X, y)


@pytest.mark.parametrize("n_trees", [5, 10])
def test_training_score_range(titanic_sample, n_trees):
    """Ensure that the training score is a valid accuracy in the range [0.0, 1.0]."""
    X, y = titanic_sample
    score = train_and_evaluate(X, y, n_trees)
    assert 0.0 <= score <= 1.0, "Score must be a valid accuracy"
