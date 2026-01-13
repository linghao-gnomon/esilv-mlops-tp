import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix



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


