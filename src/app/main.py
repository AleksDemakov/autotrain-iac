from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_URI = "models:/kaggle-champion@champion"  

app = FastAPI(title="MLOps pipeline API")
Instrumentator().instrument(app).expose(app)     

model = None
@app.on_event("startup")
def load_champion():
    global model
    model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "loading"}

class PredictRequest(BaseModel):
    features: list[float]   

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "model not loaded")
    return {"prediction": int(model.predict([req.features])[0])}
