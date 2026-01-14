import argparse

from constants import RAW_DATA_PATH
from data.load import load_local_data
from data.split import split_data
from evaluation import evaluate_model
from model import create_model_pipeline, train_model


def main():
    """
    Main function to run the complete machine learning pipeline.
    """
    parser = argparse.ArgumentParser(description="Titanic Survival Prediction")
    parser.add_argument(
        "--n_trees", type=int, default=20, help="Number of trees in the model"
    )
    args = parser.parse_args()

    print("Loading data...")
    training_data = load_local_data(RAW_DATA_PATH)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(
        training_data, target_column="Survived", test_size=0.1, random_state=42
    )

    print("Creating model pipeline...")
    numeric_features = ["Age", "Fare"]
    categorical_features = ["Embarked", "Sex"]

    model = create_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_estimators=args.n_trees,
        max_depth=None,
        max_features="sqrt",
    )

    print("Training model...")
    trained_model = train_model(model, X_train, y_train)

    print("Evaluating model...")
    results = evaluate_model(trained_model, X_test, y_test, X_train, y_train)
    print("Number of trees used:", args.n_trees)
    print("\n" + "=" * 50)
    print("Model Evaluation Results")
    print("=" * 50)
    print(f"Test Score: {results['test_score']:.2%}")
    if "train_score" in results:
        print(f"Train Score: {results['train_score']:.2%}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("=" * 50)


if __name__ == "__main__":
    main()
