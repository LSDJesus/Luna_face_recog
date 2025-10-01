"""FastAPI microservice exposing core face recognition endpoints.

Run with:
  uvicorn luna_face_recog.api_server:app --host 0.0.0.0 --port 8000

Endpoints:
  GET /health -> {status}
  POST /verify {img1: <b64>, img2: <b64>} -> distance / verified
  POST /represent {img: <b64>} -> embedding (list[float])
  POST /detect {img: <b64>} -> detections

Images are base64-encoded JPEG/PNG bytes. Returns JSON.
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2

from .luna_face import LunaFace


app = FastAPI(title="Luna Face Recognition API", version="1.0.0")


class VerifyRequest(BaseModel):
    img1: str
    img2: str
    model: Optional[str] = "arcface"
    device: Optional[str] = "auto"
    pack: Optional[str] = "buffalo_l"
    pack_root: Optional[str] = None


class RepresentRequest(BaseModel):
    img: str
    model: Optional[str] = "arcface"
    device: Optional[str] = "auto"
    pack: Optional[str] = "buffalo_l"
    pack_root: Optional[str] = None


class DetectRequest(BaseModel):
    img: str
    model: Optional[str] = "arcface"  # for parity, though detection independent
    device: Optional[str] = "auto"
    pack: Optional[str] = "buffalo_l"
    pack_root: Optional[str] = None


def _decode_image(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 decode returned None")
        return img
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


@app.get("/health")
def health():  # pragma: no cover - trivial
    return {"status": "ok"}


@app.post("/verify")
def verify(req: VerifyRequest):
    lf = LunaFace(model_name=req.model or "arcface", device=req.device or "auto", detector_pack=req.pack or "buffalo_l", detector_root=req.pack_root)
    img1 = _decode_image(req.img1)
    img2 = _decode_image(req.img2)
    res = lf.verify(img1, img2)
    return res


@app.post("/represent")
def represent(req: RepresentRequest):
    lf = LunaFace(model_name=req.model or "arcface", device=req.device or "auto", detector_pack=req.pack or "buffalo_l", detector_root=req.pack_root)
    img = _decode_image(req.img)
    emb = lf.represent(img)
    return {"embedding": emb.tolist(), "length": len(emb)}


@app.post("/detect")
def detect(req: DetectRequest):
    lf = LunaFace(model_name=req.model or "arcface", device=req.device or "auto", detector_pack=req.pack or "buffalo_l", detector_root=req.pack_root)
    img = _decode_image(req.img)
    dets = lf.detect(img)
    return dets
