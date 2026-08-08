import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Pastikan aic_qc_artifact/ bisa di-import terlepas dari CWD/WORKDIR Docker
sys.path.insert(0, str(Path(__file__).resolve().parent / "aic_qc_artifact"))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import inference
from schemas import AnalyzeRequest

app = FastAPI(title="AIC QC Analyzer")

# TODO: ganti allow_origins ke domain frontend spesifik sebelum submission final
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format ulang error default FastAPI biar sesuai API_CONTRACT_v1.md."""
    first_error = exc.errors()[0]
    field = first_error["loc"][-1] if first_error["loc"] else None
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "INVALID_INPUT", "message": f"Field {field}: {first_error['msg']}", "field": field}},
    )

def to_jsonable(obj):
    """Bikin output inference.predict() (numpy/pandas types) aman buat JSON encoder."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item) and hasattr(obj, "dtype"):
        return obj.item()      # numpy scalar -> python native
    if hasattr(obj, "tolist") and callable(obj.tolist) and hasattr(obj, "dtype"):
        return obj.tolist()    # numpy ndarray -> list
    return obj

@app.post("/api/v1/analyze")
async def analyze(payload: AnalyzeRequest):
    record = payload.model_dump()

    # [OPEN item di kontrak] — sementara di-generate server kalau client gak kirim.
    # Perlu difinalkan bareng tim, ini asumsi kerja aja dulu.
    if not record.get("batch_id"):
        record["batch_id"] = f"AUTO-{uuid.uuid4().hex[:8]}"
    record["timestamp"] = (record.get("timestamp") or datetime.now(timezone.utc)).isoformat()

    result = inference.predict(record)
    return to_jsonable(result)


@app.get("/health")
async def health():
    return {"status": "ok"}