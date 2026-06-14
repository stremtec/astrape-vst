from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from astrape.streaming_pipeline import StreamingVoiceConverter
from astrape.voicebank import VoiceBank
from astrape.wave_decoder import load_wave_decoder


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "webui" / "static"
VOICEBANK_DIR = ROOT / "voicebanks"
MEDIA_DIR = VOICEBANK_DIR / "media"
CHECKPOINT_DIR = ROOT / "checkpoints"
STATE_DIR = ROOT / ".webui"
SETTINGS_PATH = STATE_DIR / "settings.json"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_CONTENT_CHECKPOINT = CHECKPOINT_DIR / "content_student_768x10_fsq.best.pt"

VOICEBANK_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BuildJob:
    id: str
    name: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting"
    output: str = ""
    created_at: float = 0.0
    finished_at: float | None = None


class RuntimeSettings(BaseModel):
    voicebank: str = ""
    input_device: str = ""
    output_device: str = ""
    compute_device: str = "mps"
    chunk_ms: float = Field(default=5.0, ge=2.5, le=20.0)
    pitch_semitones: float = Field(default=0.0, ge=-24.0, le=24.0)
    formant_semitones: float = Field(default=0.0, ge=-12.0, le=12.0)
    input_gain_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    output_gain_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    noise_gate_db: float = Field(default=-55.0, ge=-80.0, le=-20.0)
    wet: float = Field(default=1.0, ge=0.0, le=1.0)
    f0_engine: str = "fcpe"
    f0_threshold: float = Field(default=0.006, ge=0.0, le=1.0)
    f0_min: float = Field(default=50.0, ge=20.0, le=1000.0)
    f0_max: float = Field(default=1100.0, ge=50.0, le=3000.0)
    f0_smoothing: float = Field(default=0.15, ge=0.0, le=1.0)
    protect_unvoiced: bool = True


app = FastAPI(title="Astrape VC Console", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_jobs: dict[str, BuildJob] = {}
_jobs_lock = threading.Lock()
_fcpe_model: Any = None
_fcpe_lock = threading.Lock()
_runtime_lock = asyncio.Lock()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._")
    if not slug:
        raise HTTPException(400, "Profile name is empty")
    return slug[:64]


def _mio_python() -> str:
    configured = os.environ.get("MIO_PYTHON")
    candidates = [
        configured,
        sys.executable if importlib.util.find_spec("miocodec") else None,
        "/Users/asill/btrvrc0/.venv/bin/python",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("MioCodec Python runtime was not found")


def _wave_checkpoint() -> Path | None:
    candidates = sorted(
        CHECKPOINT_DIR.glob("direct_wave_decoder*.best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _training_status() -> dict[str, Any]:
    latest = ROOT / "logs" / "content_curriculum.latest"
    if not latest.exists():
        return {"running": False, "line": "No active curriculum"}
    log_path = Path(latest.read_text().strip())
    lines = log_path.read_text(errors="replace").splitlines() if log_path.exists() else []
    line = lines[-1] if lines else "Waiting for first log line"
    match = re.search(
        r"E(\d+) (\w+)(?: step=(\d+)/(\d+))?.*?(?:frame_cos=([0-9.-]+))?",
        line,
    )
    pid_path = ROOT / "logs" / "content_curriculum.pid"
    pid = int(pid_path.read_text()) if pid_path.exists() else None
    running = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            running = True
        except OSError:
            pass
    result: dict[str, Any] = {
        "running": running,
        "pid": pid,
        "line": line,
        "log": str(log_path),
    }
    if match:
        result.update(
            {
                "epoch": int(match.group(1)),
                "phase": match.group(2),
                "step": int(match.group(3)) if match.group(3) else None,
                "steps": int(match.group(4)) if match.group(4) else None,
                "frame_cosine": (
                    float(match.group(5)) if match.group(5) else None
                ),
            }
        )
    return result


def _voicebank_payload(path: Path) -> dict[str, Any]:
    bank = VoiceBank.load(path)
    source = Path(bank.source_path)
    return {
        "id": path.stem,
        "file": path.name,
        "duration_seconds": bank.duration_seconds,
        "source_sample_rate": bank.source_sample_rate,
        "embedding_model": bank.embedding_model,
        "embedding_norm": float(bank.global_embedding.norm()),
        "created_utc": bank.created_utc,
        "peak_amplitude": bank.peak_amplitude,
        "rms_dbfs": bank.rms_dbfs,
        "clipping_fraction": bank.clipping_fraction,
        "active_speech_ratio": bank.active_speech_ratio,
        "dc_offset": bank.dc_offset,
        "quality_warnings": list(bank.quality_warnings),
        "has_source": source.is_file(),
        "preview_url": f"/api/voicebanks/{path.stem}/source" if source.is_file() else None,
    }


def _list_voicebanks() -> list[dict[str, Any]]:
    profiles = []
    for path in sorted(
        VOICEBANK_DIR.glob("*.npz"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        try:
            profiles.append(_voicebank_payload(path))
        except Exception as error:
            profiles.append(
                {
                    "id": path.stem,
                    "file": path.name,
                    "error": str(error),
                    "quality_warnings": ["profile_unreadable"],
                }
            )
    return profiles


def _decoder_capabilities() -> dict[str, Any]:
    checkpoint = _wave_checkpoint()
    if checkpoint is None:
        return {
            "ready": False,
            "checkpoint": None,
            "reason": "Direct waveform decoder checkpoint is not trained yet",
            "supports_f0_conditioning": False,
            "supports_formant_conditioning": False,
            "f0_model": "",
        }
    try:
        model = load_wave_decoder(checkpoint)
        config = model.config
        return {
            "ready": True,
            "checkpoint": str(checkpoint),
            "reason": "",
            "supports_f0_conditioning": config.supports_f0_conditioning,
            "supports_formant_conditioning": config.supports_formant_conditioning,
            "f0_model": config.f0_model,
            "sample_rate": config.sample_rate,
            "content_rate": config.content_rate,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    except Exception as error:
        return {
            "ready": False,
            "checkpoint": str(checkpoint),
            "reason": str(error),
            "supports_f0_conditioning": False,
            "supports_formant_conditioning": False,
            "f0_model": "",
        }


def _run_voicebank_job(
    job_id: str,
    source_path: Path,
    output_path: Path,
    device: str,
) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = "running"
        job.progress = 0.2
        job.message = "Loading Mio global encoder"
    command = [
        _mio_python(),
        str(ROOT / "build_voicebank.py"),
        "--reference",
        str(source_path),
        "--output",
        str(output_path),
        "--device",
        device,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        with _jobs_lock:
            job.status = "complete"
            job.progress = 1.0
            job.message = "VoiceBank ready"
            job.output = output_path.name
            job.finished_at = time.time()
    except Exception as error:
        with _jobs_lock:
            job.status = "failed"
            job.progress = 1.0
            job.message = str(error)
            job.finished_at = time.time()


def _get_fcpe(device: str = "cpu"):
    global _fcpe_model
    with _fcpe_lock:
        if _fcpe_model is None:
            import torchfcpe

            _fcpe_model = torchfcpe.spawn_bundled_infer_model(device=device)
        return _fcpe_model


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), sample_rate


def _f0_summary(
    path: Path,
    threshold: float,
    f0_min: float,
    f0_max: float,
) -> dict[str, Any]:
    audio, sample_rate = _load_mono(path)
    model = _get_fcpe("cpu")
    with torch.inference_mode():
        f0 = model.infer(
            torch.from_numpy(audio).unsqueeze(0),
            sample_rate,
            threshold=threshold,
            f0_min=f0_min,
            f0_max=f0_max,
        ).squeeze().cpu().numpy()
    voiced = f0 >= f0_min
    voiced_f0 = f0[voiced]
    if voiced_f0.size:
        statistics = {
            "median_hz": float(np.median(voiced_f0)),
            "mean_hz": float(np.mean(voiced_f0)),
            "p05_hz": float(np.percentile(voiced_f0, 5)),
            "p95_hz": float(np.percentile(voiced_f0, 95)),
            "min_hz": float(np.min(voiced_f0)),
            "max_hz": float(np.max(voiced_f0)),
        }
    else:
        statistics = {
            key: None
            for key in ("median_hz", "mean_hz", "p05_hz", "p95_hz", "min_hz", "max_hz")
        }
    target_points = min(360, max(1, f0.size))
    positions = np.linspace(0, max(0, f0.size - 1), target_points).astype(int)
    hop_ms = float(model.get_hop_size_ms())
    return {
        "engine": "FCPE",
        "hop_ms": hop_ms,
        "model_sample_rate": int(model.get_model_sr()),
        "model_range": model.get_model_f0_range(),
        "voiced_ratio": float(np.mean(voiced)) if f0.size else 0.0,
        "statistics": statistics,
        "curve": [
            {
                "time": float(index * hop_ms / 1000.0),
                "hz": float(f0[index]) if voiced[index] else 0.0,
            }
            for index in positions
        ],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    capabilities = _decoder_capabilities()
    return {
        "training": _training_status(),
        "decoder": capabilities,
        "content_checkpoint": (
            str(DEFAULT_CONTENT_CHECKPOINT)
            if DEFAULT_CONTENT_CHECKPOINT.exists()
            else None
        ),
        "voicebank_count": len(list(VOICEBANK_DIR.glob("*.npz"))),
        "f0": {
            "engine": "FCPE",
            "installed": importlib.util.find_spec("torchfcpe") is not None,
            "live_conditioning": (
                capabilities["ready"]
                and capabilities["supports_f0_conditioning"]
            ),
        },
    }


@app.get("/api/voicebanks")
def voicebanks() -> list[dict[str, Any]]:
    return _list_voicebanks()


@app.post("/api/voicebanks")
async def create_voicebank(
    file: UploadFile = File(...),
    name: str = Form(...),
    device: str = Form("cpu"),
) -> dict[str, Any]:
    profile_id = _slug(name)
    suffix = Path(file.filename or "reference.wav").suffix.lower() or ".wav"
    source_path = MEDIA_DIR / f"{profile_id}{suffix}"
    output_path = VOICEBANK_DIR / f"{profile_id}.npz"
    total = 0
    with source_path.open("wb") as destination:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                destination.close()
                source_path.unlink(missing_ok=True)
                raise HTTPException(413, "Reference file exceeds 100 MB")
            destination.write(chunk)
    if total == 0:
        source_path.unlink(missing_ok=True)
        raise HTTPException(400, "Reference file is empty")
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = BuildJob(
            id=job_id,
            name=profile_id,
            created_at=time.time(),
        )
    threading.Thread(
        target=_run_voicebank_job,
        args=(job_id, source_path, output_path, device),
        daemon=True,
    ).start()
    return {"job_id": job_id, "profile_id": profile_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        current = _jobs.get(job_id)
        if current is None:
            raise HTTPException(404, "Job not found")
        return asdict(current)


@app.get("/api/voicebanks/{profile_id}/source")
def voicebank_source(profile_id: str) -> FileResponse:
    path = VOICEBANK_DIR / f"{_slug(profile_id)}.npz"
    if not path.exists():
        raise HTTPException(404, "VoiceBank not found")
    source = Path(VoiceBank.load(path).source_path)
    if not source.is_file():
        raise HTTPException(404, "Reference source is unavailable")
    return FileResponse(source)


@app.get("/api/voicebanks/{profile_id}/f0")
def voicebank_f0(
    profile_id: str,
    threshold: float = 0.006,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
) -> dict[str, Any]:
    path = VOICEBANK_DIR / f"{_slug(profile_id)}.npz"
    if not path.exists():
        raise HTTPException(404, "VoiceBank not found")
    source = Path(VoiceBank.load(path).source_path)
    if not source.is_file():
        raise HTTPException(404, "Reference source is unavailable")
    if not 0 <= threshold <= 1 or not 20 <= f0_min < f0_max <= 3000:
        raise HTTPException(400, "Invalid FCPE range or threshold")
    return _f0_summary(source, threshold, f0_min, f0_max)


@app.delete("/api/voicebanks/{profile_id}")
def delete_voicebank(profile_id: str) -> dict[str, bool]:
    profile_id = _slug(profile_id)
    path = VOICEBANK_DIR / f"{profile_id}.npz"
    if not path.exists():
        raise HTTPException(404, "VoiceBank not found")
    bank = VoiceBank.load(path)
    source = Path(bank.source_path)
    path.unlink()
    try:
        source.relative_to(MEDIA_DIR).is_relative_to(Path("."))
        source.unlink(missing_ok=True)
    except ValueError:
        pass
    return {"deleted": True}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return RuntimeSettings().model_dump()
    try:
        return RuntimeSettings(**json.loads(SETTINGS_PATH.read_text())).model_dump()
    except Exception:
        return RuntimeSettings().model_dump()


@app.put("/api/settings")
def put_settings(settings: RuntimeSettings) -> dict[str, Any]:
    SETTINGS_PATH.write_text(
        json.dumps(settings.model_dump(), indent=2),
        encoding="utf-8",
    )
    return settings.model_dump()


@app.websocket("/api/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        configuration = await websocket.receive_json()
        profile_id = _slug(str(configuration.get("voicebank", "")))
        voicebank_path = VOICEBANK_DIR / f"{profile_id}.npz"
        wave_checkpoint = _wave_checkpoint()
        if not voicebank_path.exists():
            await websocket.send_json({"type": "error", "message": "VoiceBank not found"})
            await websocket.close(code=1008)
            return
        if wave_checkpoint is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Direct waveform decoder checkpoint is not ready",
                }
            )
            await websocket.close(code=1013)
            return
        wave_model = load_wave_decoder(wave_checkpoint)
        pitch = float(configuration.get("pitch_semitones", 0.0))
        formant = float(configuration.get("formant_semitones", 0.0))
        if pitch and not wave_model.config.supports_f0_conditioning:
            await websocket.send_json(
                {"type": "error", "message": "This checkpoint has no F0 conditioning"}
            )
            await websocket.close(code=1008)
            return
        if formant and not wave_model.config.supports_formant_conditioning:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "This checkpoint has no formant conditioning",
                }
            )
            await websocket.close(code=1008)
            return
        device = str(configuration.get("compute_device", "mps"))
        async with _runtime_lock:
            converter = StreamingVoiceConverter.from_checkpoints(
                DEFAULT_CONTENT_CHECKPOINT,
                wave_checkpoint,
                voicebank_path,
                device=device,
            )
            converter.warmup(
                max(1, round(converter.input_sample_rate * 0.005))
            )
            await websocket.send_json(
                {
                    "type": "ready",
                    "input_sample_rate": converter.input_sample_rate,
                    "output_sample_rate": converter.output_sample_rate,
                }
            )
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    audio = np.frombuffer(message["bytes"], dtype=np.float32).copy()
                    chunk = converter.process(torch.from_numpy(audio))
                    if chunk.output_samples:
                        await websocket.send_bytes(
                            chunk.audio.squeeze(0).numpy().astype(np.float32).tobytes()
                        )
                elif message.get("text"):
                    control = json.loads(message["text"])
                    if control.get("type") == "flush":
                        chunk = converter.flush()
                        if chunk.output_samples:
                            await websocket.send_bytes(
                                chunk.audio.squeeze(0).numpy().astype(np.float32).tobytes()
                            )
                        await websocket.send_json(
                            {
                                "type": "flushed",
                                "counters": asdict(converter.counters),
                            }
                        )
                        return
    except WebSocketDisconnect:
        return
    except Exception as error:
        try:
            await websocket.send_json({"type": "error", "message": str(error)})
        except Exception:
            pass


def main() -> None:
    import uvicorn

    uvicorn.run(
        "webui.server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
