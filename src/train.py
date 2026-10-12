from pathlib import Path

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from sklearn.model_selection import train_test_split

from data_loader import load_data
from validate_data import validate_data
from model_pipeline import create_model_pipeline
from evaluate import evaluate_model


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

EXPERIMENT_NAME = "churn-prediction"
REGISTERED_MODEL_NAME = "churn_classifier"
CHAMPION_ALIAS = "champion"


def register_if_better(model_uri, run_id, candidate_f1):

    client = MlflowClient()

    # Check whether registered model already exists
    try:
        champion = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS
        )

        # Get F1 score of current champion
        champion_run = client.get_run(champion.run_id)

        champion_f1 = champion_run.data.metrics.get("f1")

        if champion_f1 is None:
            print("\nCurrent champion does not have an F1 metric.")
            print("Cannot safely compare candidate against champion.")
            return

        print(f"\nCurrent champion F1: {champion_f1:.4f}")
        print(f"Candidate model F1: {candidate_f1:.4f}")

        # Candidate is not better
        if candidate_f1 <= champion_f1:

            print(
                "\nCandidate model is NOT better "
                "than the current champion."
            )

            print("Model will NOT be registered.")

            return

        print(
            "\nCandidate model is BETTER "
            "than the current champion!"
        )

    except MlflowException:

        # No champion exists yet
        print(
            "\nNo champion model exists."
        )

        print(
            "This will be the first registered model."
        )

    # Create registered model if it doesn't exist
    try:
        client.create_registered_model(
            REGISTERED_MODEL_NAME
        )

        print(
            f"Created registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

    except MlflowException:
        # Model already exists
        pass

    # Create new model version
    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_uri,
        run_id=run_id,
    )

    print(
        f"\nRegistered model version: "
        f"{model_version.version}"
    )

    # Move champion alias to this version
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        CHAMPION_ALIAS,
        model_version.version,
    )

    print(
        f"Champion alias now points to "
        f"version {model_version.version}"
    )


def main():

    # Connect to MLflow server
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    # Select experiment
    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # Start MLflow run
    with mlflow.start_run() as run:

        run_id = run.info.run_id

        print(f"MLflow Run ID: {run_id}")

        # -------------------------
        # Load data
        # -------------------------

        df = load_data()

        print(
            f"Dataset shape: {df.shape}"
        )

        # -------------------------
        # Validate data
        # -------------------------

        validate_data(df)

        # -------------------------
        # Convert TotalCharges
        # -------------------------

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        # -------------------------
        # Features and target
        # -------------------------

        X = df.drop(
            columns=[
                "customerID",
                "Churn"
            ]
        )

        y = df["Churn"].map(
            {
                "No": 0,
                "Yes": 1
            }
        )

        # -------------------------
        # Train/test split
        # -------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        print(
            f"Training samples: {len(X_train)}"
        )

        print(
            f"Testing samples: {len(X_test)}"
        )

        # -------------------------
        # Model parameters
        # -------------------------

        n_estimators = 350
        max_depth = 12
        min_samples_split = 5
        min_samples_leaf = 2
        class_weight = "balanced"

        # -------------------------
        # Create model
        # -------------------------

        pipeline = create_model_pipeline(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )

        # -------------------------
        # Log parameters
        # -------------------------

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": class_weight,
                "test_size": 0.2,
                "random_state": 42,
            }
        )

        # -------------------------
        # Train
        # -------------------------

        print("\nTraining model...")

        pipeline.fit(
            X_train,
            y_train
        )

        print(
            "Training completed!"
        )

        # -------------------------
        # Evaluate
        # -------------------------

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        print("\nModel Metrics:")

        for name, value in metrics.items():

            print(
                f"{name}: {value:.4f}"
            )

            mlflow.log_metric(
                name,
                value
            )

        # -------------------------
        # Log model
        # -------------------------

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="churn_model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        print(
            "\nModel logged to MLflow."
        )

        # -------------------------
        # Save local model
        # -------------------------

        joblib.dump(
            pipeline,
            MODEL_PATH
        )

        print(
            f"Model saved to: {MODEL_PATH}"
        )

        # -------------------------
        # Register only if better
        # -------------------------

        register_if_better(
            model_uri=model_info.model_uri,
            run_id=run_id,
            candidate_f1=metrics["f1"],
        )


if __name__ == "__main__":
    main()