from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _sample_transaction():
    transaction = {f"v{i}": 0.0 for i in range(1, 29)}
    transaction.update({"time": 40000.0, "amount": 149.62})
    return transaction


@patch("api.main.predict_single")
def test_predict_valid_input(mock_predict_single):
    mock_predict_single.return_value = {
        "predicted_class": 0,
        "fraud_probability": 0.0123,
        "latency_ms": 5.4,
    }

    response = client.post("/predict", json=_sample_transaction())

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == 0
    assert body["fraud_probability"] == 0.0123
    assert "latency_ms" in body
    mock_predict_single.assert_called_once()


@patch("api.main.predict_single")
def test_predict_negative_amount(mock_predict_single):
    transaction = _sample_transaction()
    transaction["amount"] = -10.0

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422
    mock_predict_single.assert_not_called()


@patch("api.main.predict_single")
def test_predict_pipeline_exception(mock_predict_single):
    mock_predict_single.side_effect = RuntimeError("Model file not found")

    response = client.post("/predict", json=_sample_transaction())

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"
    mock_predict_single.assert_called_once()
