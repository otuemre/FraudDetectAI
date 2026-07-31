from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.logger import get_error_logger, log_exception
from src.pipelines.predict_pipeline import predict_single

error_logger = get_error_logger()

app = FastAPI(title="FraudDetectAI", version="1.0.0")


class Transaction(BaseModel):
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predicted_class: int
    fraud_probability: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        result = predict_single(transaction.model_dump())
        return result
    except Exception as e:
        log_exception(error_logger, e)
        raise HTTPException(status_code=500, detail="Prediction failed") from e
