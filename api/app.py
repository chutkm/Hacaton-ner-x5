from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.inference import predict_spans
from api.log_middleware import LogIO
import os
import sys, logging
app = FastAPI(title="NER Service")
app.add_middleware(LogIO)

reqlog = logging.getLogger("app.requests")
if not reqlog.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    reqlog.addHandler(h)
reqlog.setLevel(logging.INFO)

class PredictIn(BaseModel):
    input: str

class Span(BaseModel):
    start_index: int
    end_index: int
    label: str

@app.post("/api/predict")
async def predict(body: PredictIn) -> List[Span]:
    if not isinstance(body.input, str):
        raise HTTPException(status_code=400, detail="Field 'input' must be a string")
    spans = predict_spans(body.input)
    return [Span(start_index=start, end_index=end, label=label) for start, end, label in spans]

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    return {"model_dir": os.getenv("MODEL_DIR", "models/serve")}
