
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from app.model_loader import load_model
from app.schemas import CustomerFeatures


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once when the API starts."""
    print("Loading MLflow champion model...")

    try:
        app.state.model = load_model()
        print("Champion model loaded successfully!")
    except Exception as exc:
        app.state.model = None
        print(f"Model loading failed: {exc}")

    yield

    # Clean up application state during shutdown
    app.state.model = None
    print("Application shut down.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using the MLflow champion model.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model(request: Request):
    """Provide the loaded model to API endpoints."""
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Check MLflow and model registration.",
        )

    return model


@app.get("/health")
def health_check():
    """Check whether the API process is running."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check(model=Depends(get_model)):
    """Check whether the model is ready to serve predictions."""
    return {"status": "ready", "model": "churn_classifier@champion"}


@app.post("/predict")
def predict(
    customer: CustomerFeatures,
    model=Depends(get_model),
):
    """Predict churn for one customer."""
    customer_data = customer.model_dump()
    input_data = pd.DataFrame([customer_data])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API logs.",
        ) from exc

    return {
        "churn_prediction": "Yes" if int(prediction) == 1 else "No",
        "churn_probability": round(float(probability), 4),
    }
