from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.inference import predict_spans
import os
app = FastAPI(title="NER Service")

class PredictIn(BaseModel):
    input: str

@app.post("/api/predict")
async def predict(body: PredictIn) -> List[Dict[str, Any]]:
    if not isinstance(body.input, str):
        raise HTTPException(status_code=400, detail="Field 'input' must be a string")
    return predict_spans(body.input)
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    return {"model_dir": os.getenv("MODEL_DIR", "models/serve")}
