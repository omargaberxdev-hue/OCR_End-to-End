import json
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .config import settings
from .engine import OCREngine

logging.basicConfig(level=logging.INFO)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = OCREngine(settings)   # load once at startup
    yield
    state.clear()


app = FastAPI(title="RepViT + ExCTC OCR", version="1.0.0", lifespan=lifespan)


class WordResult(BaseModel):
    text: str


class LinesResult(BaseModel):
    lines: list[str]
    text: str


async def _decode(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@app.get("/health")
def health():
    eng: OCREngine = state["engine"]
    return {"status": "ok", "model": eng.info, "device": str(eng.device)}


@app.post("/ocr/word", response_model=WordResult)
async def ocr_word(file: UploadFile = File(..., description="a single cropped word image")):
    img = await _decode(file)
    return {"text": state["engine"].read(img, spaces=False)}


@app.post("/ocr/lines", response_model=LinesResult)
async def ocr_lines(
    file: UploadFile = File(..., description="a line image, or a page image together with `boxes`"),
    boxes: str | None = Form(None, description='JSON [[x1,y1,x2,y2], ...], one box per line. '
                                                'Omit to treat the whole image as ONE line.'),
):
    """The 1D CTC branch reads one line at a time, so a page needs line boxes (no line detector here)."""
    img = await _decode(file)
    H, W = img.shape[:2]
    if boxes is None:
        rects = [(0, 0, W, H)]
    else:
        try:
            rects = [tuple(int(round(float(v))) for v in b) for b in json.loads(boxes)]
            assert rects and all(len(r) == 4 for r in rects)
        except Exception:
            raise HTTPException(400, "boxes must be JSON like [[x1,y1,x2,y2], ...]")
    if len(rects) > settings.max_lines:
        raise HTTPException(400, f"at most {settings.max_lines} lines per request")

    eng: OCREngine = state["engine"]
    lines = []
    for x1, y1, x2, y2 in rects:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            raise HTTPException(400, f"empty or out-of-image box {[x1, y1, x2, y2]}")
        lines.append(eng.read(img[y1:y2, x1:x2], spaces=True))
    return {"lines": lines, "text": "\n".join(lines)}
