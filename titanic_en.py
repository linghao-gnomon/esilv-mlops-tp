import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def create_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numeric and categorical features.
    
    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        
    Returns:
        ColumnTransformer that handles imputation and encoding for both feature types.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            ("Preprocessing categorical", categorical_transformer, categorical_features),
        ]
    )
    
    return preprocessor


def create_model_pipeline(
    numeric_features: list,
    categorical_features: list,
    n_estimators: int = 20,
    max_depth: int = None,
    max_features: str = 'sqrt'
) -> Pipeline:
    """
    Create a complete pipeline including preprocessing and classification.
    
    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        n_estimators: Number of trees in the random forest.
        max_depth: Maximum depth of the trees.
        max_features: Number of features to consider when looking for the best split.
        
    Returns:
        Pipeline that includes preprocessing and RandomForestClassifier.
    """
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features
        )),
    ])
    
    return pipe


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Train a machine learning model.
    
    Args:
        model: The pipeline model to train.
        X_train: Training features.
        y_train: Training target variable.
        
    Returns:
        Trained model pipeline.
    """
    model.fit(X_train, y_train)
    return model


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


def split_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.1,
    random_state: int = None
) -> tuple:
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
    X = data.drop(target_column, axis='columns')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to run the complete machine learning pipeline.
    """
    print("Loading data...")
    training_data = load_data('titanic.csv')
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(
        training_data,
        target_column='Survived',
        test_size=0.1,
        random_state=42
    )
    
    print("Creating model pipeline...")
    numeric_features = ["Age", "Fare"]
    categorical_features = ["Embarked", "Sex"]
    
    model = create_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_estimators=20,
        max_depth=None,
        max_features='sqrt'
    )
    
    print("Training model...")
    trained_model = train_model(model, X_train, y_train)
    
    print("Evaluating model...")
    results = evaluate_model(
        trained_model,
        X_test,
        y_test,
        X_train,
        y_train
    )
    
    print("\n" + "="*50)
    print("Model Evaluation Results")
    print("="*50)
    print(f"Test Score: {results['test_score']:.2%}")
    if 'train_score' in results:
        print(f"Train Score: {results['train_score']:.2%}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("="*50)


if __name__ == "__main__":
    main()
