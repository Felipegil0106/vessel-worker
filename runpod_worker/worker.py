#!/usr/bin/env python3
"""
Vessel Render Worker — Se ejecuta automáticamente dentro del pod RunPod.

Características de seguridad y robustez:
  - Firma cada callback con HMAC SHA256
  - Envía heartbeats cada 30s para que el watchdog sepa que sigue vivo
  - En caso de error, notifica explícitamente al backend (que terminará el pod)
  - Self-terminate: si el callback final no pudo entregarse, el pod se mata solo
"""
import os
import sys
import json
import time
import hmac
import hashlib
import threading
import subprocess
import traceback
from pathlib import Path

import requests
import boto3
from botocore.client import Config

# --- Configuración desde env vars ---

TOUR_ID = os.environ["TOUR_ID"]
VIDEO_URL = os.environ["VIDEO_URL"]
RESULT_KEY = os.environ["RESULT_KEY"]
CALLBACK_URL = os.environ.get("CALLBACK_URL", "")
CALLBACK_SECRET = os.environ.get("CALLBACK_SECRET", "")

S3_CONFIG = {
    "endpoint_url": os.environ.get("S3_ENDPOINT") or None,
    "aws_access_key_id": os.environ["S3_KEY"],
    "aws_secret_access_key": os.environ["S3_SECRET"],
    "region_name": os.environ.get("S3_REGION", "auto"),
    "config": Config(signature_version="s3v4"),
}
S3_BUCKET = os.environ["S3_BUCKET"]

GS_ITERATIONS = int(os.environ.get("GS_ITERATIONS", "30000"))
USE_CLAHE = os.environ.get("GS_PREPROCESS_CLAHE", "1") == "1"

WORK = Path("/workspace/job")
WORK.mkdir(parents=True, exist_ok=True)
INPUT_VIDEO = WORK / "input.mp4"
FRAMES_DIR = WORK / "frames"
COLMAP_DIR = WORK / "colmap"
OUTPUT_PLY = WORK / "scene.ply"
OUTPUT_SPLAT = WORK / "scene.splat"

# --- Heartbeat global ---
_current_progress = 0.0
_current_message = "Iniciando..."
_keep_heartbeat = True


def _clean_url(url: str) -> str:
    """Limpia URLs mal pegadas tipo http:https://..."""
    if not url:
        return ""
    url = url.strip()
    for _ in range(5):
        changed = False
        for bad in ("http:https://", "https:https://", "http:http://", "https:http://"):
            if url.startswith(bad):
                url = url[len(bad) - len("https://"):]
                changed = True
                break
        if not changed:
            break
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


CALLBACK_URL = _clean_url(CALLBACK_URL)
print(f"[worker] CALLBACK_URL normalizada: {CALLBACK_URL}", flush=True)


def s3():
    return boto3.client("s3", **S3_CONFIG)


def sign_payload(body: bytes) -> str:
    return hmac.new(CALLBACK_SECRET.encode(), body, hashlib.sha256).hexdigest()


def callback(payload: dict):
    if not CALLBACK_URL:
        print("[no-callback]", json.dumps(payload), flush=True)
        return False
    body = json.dumps(payload).encode()
    sig = sign_payload(body)
    try:
        r = requests.post(
            CALLBACK_URL, data=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": sig,
                "X-Callback-Secret": CALLBACK_SECRET,
            },
            timeout=15,
        )
        if r.status_code != 200:
            print(f"callback non-200: {r.status_code} {r.text[:200]}", flush=True)
        return r.status_code == 200
    except Exception as e:
        print(f"callback failed: {e}", flush=True)
        return False


def report(progress: float, message: str):
    global _current_progress, _current_message
    _current_progress = progress
    _current_message = message
    print(f"[{progress*100:.1f}%] {message}", flush=True)
    callback({"type": "progress", "progress": progress, "message": message})


def heartbeat_loop():
    """Cada 30s manda un heartbeat para que el watchdog del backend sepa que estamos vivos."""
    while _keep_heartbeat:
        try:
            callback({"type": "progress", "progress": _current_progress, "message": _current_message})
        except Exception:
            pass
        time.sleep(30)


def run(cmd: list, **kw):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


# =============================================================================
# ETAPAS DEL PIPELINE
# =============================================================================

def download_video():
    report(0.05, "Descargando video desde storage...")
    r = requests.get(VIDEO_URL, stream=True, timeout=600)
    r.raise_for_status()
    with INPUT_VIDEO.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    size_mb = INPUT_VIDEO.stat().st_size / 1e6
    print(f"Video descargado: {size_mb:.1f} MB", flush=True)


def extract_frames():
    import cv2
    report(0.10, "Extrayendo frames con FFmpeg...")
    FRAMES_DIR.mkdir(exist_ok=True)
    run([
        "ffmpeg", "-y", "-i", str(INPUT_VIDEO),
        "-vf", "fps=2,scale=1920:-2",
        "-q:v", "2",
        str(FRAMES_DIR / "frame_%05d.jpg"),
    ])
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    report(0.14, f"Filtrando {len(frames)} frames borrosos...")
    kept = 0
    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None or cv2.Laplacian(img, cv2.CV_64F).var() < 40:
            fp.unlink(missing_ok=True)
        else:
            kept += 1
    print(f"Frames tras filtro: {kept}", flush=True)

    if USE_CLAHE:
        report(0.18, "Pre-procesando paredes blancas con CLAHE...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for fp in FRAMES_DIR.glob("*.jpg"):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            cv2.imwrite(
                str(fp),
                cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )


def run_colmap():
    report(0.22, "COLMAP: extrayendo features SIFT...")
    COLMAP_DIR.mkdir(exist_ok=True)
    db = COLMAP_DIR / "database.db"
    sparse = COLMAP_DIR / "sparse"
    sparse.mkdir(exist_ok=True)
    # Importante: usamos CPU para SIFT porque COLMAP necesita OpenGL para usar GPU,
    # y los pods de RunPod son headless (sin display X). El extra de 5-7 min en CPU
    # es aceptable vs. tener que compilar COLMAP con EGL.
    # El 3D Gaussian Splatting que viene después SÍ usa GPU al 100%.
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(FRAMES_DIR),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "0",  # CPU: evita problema de OpenGL en headless
        "--SiftExtraction.max_num_features", "8192",
        "--SiftExtraction.peak_threshold", "0.004",
        "--SiftExtraction.num_threads", "-1",  # usar todos los cores
    ])
    report(0.32, "COLMAP: matching secuencial (optimizado para video)...")
    # IMPORTANTE: usamos sequential_matcher en vez de exhaustive_matcher.
    # Para video con N frames, exhaustive compara N*(N-1)/2 pares (122 frames = 7,381 pares = horas en CPU).
    # Sequential compara cada frame con los siguientes 'overlap' frames.
    # Esto es 10-50x más rápido y de hecho da MEJOR calidad para video porque no hay
    # falsos matches entre frames temporalmente lejanos.
    # NOTA: loop_detection deshabilitado porque requiere un vocab_tree pre-entrenado
    # que no viene con COLMAP en Ubuntu. Para compensar, subimos el overlap a 15.
    run([
        "colmap", "sequential_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "0",  # CPU: mismo motivo que extraction
        "--SiftMatching.num_threads", "-1",
        "--SequentialMatching.overlap", "15",  # cada frame matchea con los próximos 15
        "--SequentialMatching.quadratic_overlap", "1",  # también con 2, 4, 8, 16, 32... (mejor cobertura)
    ])
    report(0.40, "COLMAP: Structure-from-Motion (poses de cámara)...")
    run([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(FRAMES_DIR),
        "--output_path", str(sparse),
    ])
    if not (sparse / "0").exists():
        raise RuntimeError(
            "COLMAP no logró reconstruir poses. "
            "El video probablemente no tiene suficiente cobertura o solapamiento."
        )


def train_gaussian_splatting():
    report(0.50, f"Entrenando 3D Gaussian Splatting ({GS_ITERATIONS} iters)...")
    gs_dir = Path("/workspace/gaussian-splatting")
    if not gs_dir.exists():
        run([
            "git", "clone", "--recursive",
            "https://github.com/graphdeco-inria/gaussian-splatting",
            str(gs_dir),
        ])
        run([
            "pip", "install", "-q",
            str(gs_dir / "submodules" / "diff-gaussian-rasterization"),
            str(gs_dir / "submodules" / "simple-knn"),
            "plyfile",
        ])
    run([
        "python3", str(gs_dir / "train.py"),
        "-s", str(COLMAP_DIR),
        "-m", str(WORK / "gs_output"),
        "--iterations", str(GS_ITERATIONS),
        "--densify_grad_threshold", "0.0002",
    ])
    final_ply = WORK / "gs_output" / "point_cloud" / f"iteration_{GS_ITERATIONS}" / "point_cloud.ply"
    if not final_ply.exists():
        candidates = list((WORK / "gs_output").rglob("*.ply"))
        if not candidates:
            raise RuntimeError("Training no produjo archivo .ply")
        final_ply = candidates[-1]
    import shutil
    shutil.copy(final_ply, OUTPUT_PLY)


def convert_to_splat():
    report(0.93, "Convirtiendo .ply a formato web .splat...")
    import numpy as np
    from plyfile import PlyData

    plydata = PlyData.read(str(OUTPUT_PLY))
    v = plydata["vertex"]
    n = len(v)
    print(f"Convirtiendo {n} splats", flush=True)

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).astype(np.float32)
    C0 = 0.28209479177387814
    r = ((0.5 + C0 * v["f_dc_0"]) * 255).clip(0, 255).astype(np.uint8)
    g = ((0.5 + C0 * v["f_dc_1"]) * 255).clip(0, 255).astype(np.uint8)
    b = ((0.5 + C0 * v["f_dc_2"]) * 255).clip(0, 255).astype(np.uint8)
    a = ((1 / (1 + np.exp(-v["opacity"]))) * 255).clip(0, 255).astype(np.uint8)
    color = np.stack([r, g, b, a], axis=1)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    rot = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9)
    rot_q = ((rot * 128) + 128).clip(0, 255).astype(np.uint8)

    buf = bytearray()
    for i in range(n):
        buf += xyz[i].tobytes()
        buf += scales[i].tobytes()
        buf += color[i].tobytes()
        buf += rot_q[i].tobytes()

    OUTPUT_SPLAT.write_bytes(bytes(buf))
    print(f".splat: {OUTPUT_SPLAT.stat().st_size / 1e6:.1f} MB", flush=True)


def upload_result():
    report(0.97, "Subiendo resultado al storage...")
    s3().upload_file(
        str(OUTPUT_SPLAT), S3_BUCKET, RESULT_KEY,
        ExtraArgs={
            "ContentType": "application/octet-stream",
            "CacheControl": "public, max-age=31536000",
        },
    )


def main():
    global _keep_heartbeat
    # PRIMER HEARTBEAT: lo mandamos antes de hacer NADA, para que el watchdog
    # sepa que estamos vivos. Si esto falla, no podemos comunicarnos con el backend,
    # así que abortamos rápido en vez de gastar GPU procesando.
    print("[worker] Iniciando, enviando primer heartbeat...", flush=True)
    for attempt in range(3):
        if callback({"type": "progress", "progress": 0.0, "message": "Worker arrancado, preparando ambiente..."}):
            print("[worker] Backend respondió OK al primer heartbeat", flush=True)
            break
        print(f"[worker] Primer heartbeat falló (intento {attempt+1}/3), reintentando en 3s...", flush=True)
        time.sleep(3)
    else:
        print("[worker] CRITICAL: No se pudo contactar al backend tras 3 intentos. Abortando.", flush=True)
        sys.exit(1)

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()
    try:
        download_video()
        extract_frames()
        run_colmap()
        train_gaussian_splatting()
        convert_to_splat()
        upload_result()
        _keep_heartbeat = False

        # Notificar éxito. Intentamos hasta 5 veces para asegurar que el backend
        # se entera y mata el pod en lugar de dejarnos corriendo.
        success_payload = {"type": "completed", "result_key": RESULT_KEY}
        for attempt in range(5):
            if callback(success_payload):
                print("[worker] Backend notificado del éxito", flush=True)
                break
            time.sleep(5)
        print("[worker] Pipeline completado", flush=True)
    except subprocess.CalledProcessError as e:
        _keep_heartbeat = False
        traceback.print_exc()
        for attempt in range(3):
            if callback({
                "type": "error",
                "error_code": "subprocess_failed",
                "error_message": f"Comando falló: {' '.join(e.cmd[:2]) if e.cmd else 'unknown'}",
            }):
                break
            time.sleep(5)
        sys.exit(1)
    except Exception as e:
        _keep_heartbeat = False
        traceback.print_exc()
        for attempt in range(3):
            if callback({
                "type": "error",
                "error_code": "pipeline_exception",
                "error_message": str(e)[:300],
            }):
                break
            time.sleep(5)
        sys.exit(1)


if __name__ == "__main__":
    main()
