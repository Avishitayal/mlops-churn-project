import os
import mlflow

from app.config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODEL_ALIAS,
)


def load_model():

    model_source = os.getenv(
        "MODEL_SOURCE",
        "mlflow"
    )

    if model_source == "local":

        model_path = os.getenv(
            "MODEL_PATH",
            "models/champion"
        )

        print(
            f"Loading local model from: {model_path}"
        )

        model = mlflow.sklearn.load_model(
            model_path
        )

        print("Local champion model loaded successfully!")

        return model

    # Default: MLflow Registry
    print(
        f"Loading model from MLflow: "
        f"{MODEL_NAME}@{MODEL_ALIAS}"
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    print("MLflow champion model loaded successfully!")

    return model