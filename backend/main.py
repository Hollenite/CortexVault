# backend/main.py
import os
import shutil
import tempfile
import time
from typing import List, Optional, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your existing backend module (ensure it's in the same folder or PYTHONPATH)
# organizer_backend.py was uploaded by you — it contains analyze_path, save_item, ingest_json_batch, load_models, analyze_image, ORGANIZED_ROOT
from organizer_backend import analyze_path, save_item, ingest_json_batch, load_models, analyze_image, ORGANIZED_ROOT

app = FastAPI(title="CortexVault Backend")

# Allow CORS from React dev server (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = os.path.join(tempfile.gettempdir(), "cortexvault_tmp_fastapi")
os.makedirs(TMP_DIR, exist_ok=True)

# Load ML models once (best-effort). This mirrors your load_models usage.
yolo_model, clf_model, preprocess, labels = None, None, None, None
try:
    yolo_model, clf_model, preprocess, labels = load_models()
except Exception:
    yolo_model, clf_model, preprocess, labels = None, None, None, None

def save_upload_to_tmp(upload: UploadFile) -> str:
    """Save UploadFile to temp path and return path."""
    ts = int(time.time() * 1000)
    out_dir = os.path.join(TMP_DIR, f"u_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    dest_path = os.path.join(out_dir, upload.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest_path

@app.get("/health")
async def health():
    return {"status": "ok", "organized_root": ORGANIZED_ROOT}

@app.post("/analyze_file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Upload a single file and return the backend analyze_path result.
    Does NOT persist to final output; file is stored temporarily and path is included for further save calls.
    """
    try:
        tmp_path = save_upload_to_tmp(file)
        info = analyze_path(tmp_path)
        # Return also a reference path so frontend can call /save_file with tmp_path
        return {"ok": True, "tmp_path": tmp_path, "info": info, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_file")
async def save_file(tmp_path: str = Form(...), custom_folder: Optional[str] = Form(None)):
    """
    Save a previously uploaded temp file into organized output using save_item.
    Accepts tmp_path returned from /analyze_file.
    """
    if not os.path.exists(tmp_path):
        raise HTTPException(status_code=400, detail="tmp_path not found on server.")
    try:
        res = save_item(tmp_path, custom_folder)
        return {"ok": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_json")
async def analyze_json(payload: Any):
    """
    Analyze pasted JSON (body) — run the backend ingestion decisioning but do not persist.
    Body must be application/json.
    """
    try:
        info = analyze_path(None, batch_json=payload)
        return {"ok": True, "analysis": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_json")
async def ingest_json(payload: Any):
    """
    Ingest JSON into SQL/NoSQL output (persist) using ingest_json_batch.
    Returns artifacts info (db path, saved file list, etc.)
    """
    try:
        res = ingest_json_batch(payload)
        return {"ok": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_image_file")
async def analyze_image_file(file: UploadFile = File(...)):
    """
    Specialized endpoint: upload image and run analyze_image using loaded models. Returns detected & mapped categories.
    """
    try:
        tmp_path = save_upload_to_tmp(file)
        detected, mapped, annotated = analyze_image(tmp_path, yolo_model, clf_model, preprocess, labels)
        return {"ok": True, "tmp_path": tmp_path, "detected": detected, "mapped": mapped}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_artifact")
async def download_artifact(path: str):
    """
    Return a saved artifact by path. Use with caution: ensure path is within ORGANIZED_ROOT for security.
    """
    # Security check: prevent path traversal
    abs_root = os.path.abspath(ORGANIZED_ROOT)
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(abs_root):
        raise HTTPException(status_code=403, detail="Access to path denied.")
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(abs_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
