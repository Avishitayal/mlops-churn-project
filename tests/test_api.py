import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class MockModel:
    def predict(self, data):
        return [0]

    def predict_proba(self, data):
        return [[0.7, 0.3]]


@pytest.fixture
def client():
    mock_model = MockModel()

    with patch("app.main.load_model", return_value=mock_model):
        with TestClient(app) as client:
            yield client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_endpoint(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_invalid_prediction_request(client):
    response = client.post(
        "/predict",
        json={"gender": "Female"}
    )

    assert response.status_code == 422


def test_successful_prediction(client):
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 60.5,
        "TotalCharges": 302.5
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= result["churn_probability"] <= 1.0