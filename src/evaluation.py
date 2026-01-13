import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None
) -> dict:
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained pipeline model.
        X_test: Test features.
        y_test: Test target variable.
        X_train: Optional training features for training score.
        y_train: Optional training target for training score.
        
    Returns:
        Dictionary containing evaluation metrics including test score,
        confusion matrix, and optionally training score.
    """
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        "test_score": test_score,
        "confusion_matrix": cm,
    }
    
    if X_train is not None and y_train is not None:
        train_score = model.score(X_train, y_train)
        results["train_score"] = train_score
    
    return results