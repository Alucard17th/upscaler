import os
from typing import Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer


MODEL_NAME = "RealESRGAN_x4plus"
NETSCALE = 4
OUTSCALE = 2.490234375
TILE = 512
TILE_PAD = 10
PRE_PAD = 0
FP32 = True
RESULTS_DIR = "results"


app = FastAPI(title="Real-ESRGAN Upscaler API")

_upsampler: Optional[RealESRGANer] = None


def _get_model_path() -> str:
    model_path = os.path.join("weights", f"{MODEL_NAME}.pth")
    if os.path.isfile(model_path):
        return model_path

    root_dir = os.path.dirname(os.path.abspath(__file__))
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    return load_file_from_url(url=url, model_dir=os.path.join(root_dir, "weights"), progress=True, file_name=None)


@app.on_event("startup")
def _startup() -> None:
    global _upsampler

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=NETSCALE)
    model_path = _get_model_path()

    _upsampler = RealESRGANer(
        scale=NETSCALE,
        model_path=model_path,
        model=model,
        tile=TILE,
        tile_pad=TILE_PAD,
        pre_pad=PRE_PAD,
        half=not FP32,
        gpu_id=None,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "outscale": OUTSCALE,
        "tile": TILE,
        "fp32": FP32,
    }


def _save_to_results(output: np.ndarray, original_name: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_name or "image"))[0]
    save_path = os.path.join(RESULTS_DIR, f"{base_name}_out.png")
    if os.path.exists(save_path):
        i = 1
        while True:
            candidate = os.path.join(RESULTS_DIR, f"{base_name}_out_{i}.png")
            if not os.path.exists(candidate):
                save_path = candidate
                break
            i += 1
    cv2.imwrite(save_path, output)
    return save_path


@app.post("/upscale", responses={200: {"content": {"image/png": {}}}})
async def upscale(file: UploadFile = File(...)) -> Response:
    if _upsampler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image")

    try:
        output, _ = _upsampler.enhance(img, outscale=OUTSCALE)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    _save_to_results(output, file.filename or "image")

    ok, encoded = cv2.imencode(".png", output)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output")

    return Response(content=encoded.tobytes(), media_type="image/png")


@app.post("/upscale_binary", responses={200: {"content": {"image/png": {}}}})
async def upscale_binary(
    body: bytes = Body(...),
    x_filename: Optional[str] = Header(default=None),
) -> Response:
    if _upsampler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    arr = np.frombuffer(body, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image")

    try:
        output, _ = _upsampler.enhance(img, outscale=OUTSCALE)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    _save_to_results(output, x_filename or "image")

    ok, encoded = cv2.imencode(".png", output)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output")

    return Response(content=encoded.tobytes(), media_type="image/png")
