#!/usr/bin/env python3
"""Gradio app for evaluating VAD showcase videos and rating their performance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
GRADIO_TMP_DIR = REPO_ROOT / "tmp" / "gradio_temp"
GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TMP_DIR))


def _resolve_runtime_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


RUNS_DIR = _resolve_runtime_path(os.environ.get("MUVAD_GRADIO_RUNS_DIR", REPO_ROOT / "tmp" / "gradio_runs"))
DATA_DIR = _resolve_runtime_path(os.environ.get("MUVAD_DATA_DIR", REPO_ROOT / "data"))
EVALUATIONS_FILE = _resolve_runtime_path(
    os.environ.get("MUVAD_EVALUATIONS_FILE", REPO_ROOT / "tmp" / "vad_evaluations.json")
)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
MAX_RUNTIME_STEM_LENGTH = 80

try:
    import gradio as gr
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Gradio is not installed. Install it with 'pip install gradio'."
    ) from exc

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# For plotting
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demos.vad_showcase import ANOMALY_PROMPT


# ------------------------------
# Translations
# ------------------------------

TRANSLATIONS = {
    "en": {
        "title": "## VAD Performance Evaluation",
        "subtitle": "Process new videos or review existing analysis results.",
        "tab_process": "Process Videos",
        "tab_evaluate": "Evaluate Runs",
        "process_header": "### Process Unprocessed Videos",
        "select_video": "Select video to process",
        "unprocessed": "unprocessed",
        "refresh": "Refresh",
        "refresh_short": "Refresh",
        "llama_url": "Llama Server URL",
        "llama_model": "Llama Model",
        "start_processing": "Start Processing",
        "processing_logs": "Processing Logs",
        "evaluate_header": "### Review and Rate Analysis Results",
        "select_run": "Select a run to evaluate",
        "status": "Status",
        "select_run_begin": "Select a run to begin.",
        "visualization_header": "### Analysis Visualization",
        "video_player": "Original Video",
        "window_frames": "Analysis Window",
        "no_windows": "No windows analyzed.",
        "select_run_details": "Select a run to see current window details",
        "timeline_header": "### Anomaly Score Timeline",
        "timeline_hint": "Click on the plot to navigate to a specific window",
        "anomaly_plot": "Anomaly Score Plot",
        "window_nav": "Window Navigation",
        "window_nav_info": "Drag to navigate between analysis windows",
        "rate_header": "### Rate the Analysis Quality",
        "rate_scale": "1 = Very Poor | 2 = Poor | 3 = Average | 4 = Good | 5 = Excellent",
        "score_consistency": "Anomaly Score Consistency",
        "score_consistency_info": "Do the scores make sense and are consistent?",
        "description_quality": "Description Quality",
        "description_quality_info": "Are the descriptions accurate?",
        "detection_accuracy": "Detection Accuracy",
        "detection_accuracy_info": "Does it correctly identify anomalies?",
        "overall_quality": "Overall Quality",
        "overall_quality_info": "Your overall impression.",
        "notes": "Notes (optional)",
        "notes_placeholder": "Any additional observations...",
        "submit": "Submit Evaluation",
        "result": "Result",
        "view_all": "View All Evaluations",
        "load_summary": "Load Summary",
        "eval_file": "Evaluations file",
        "export_csv": "Export CSV",
        "csv_status": "Export Status",
        "csv_path": "CSV file path",
        "language": "Language",
        "upload_video": "Upload video",
        "upload_video_info": "Drag & drop",
        "video_uploaded": "Video uploaded: {name}",
        "current_window": "Current Window",
        "score": "Score",
        "no_anomalies": "The model did not detect any anomalies above threshold.",
        "anomaly_detected": "The model detected an anomaly in the video from {start}s to {end}s. Reason: {reason}",
        "no_description": "No description available.",
        "error_no_video": "Error: Please select a video first.",
        "error_no_run": "Error: No run selected.",
        "eval_saved": "Evaluation saved for {run}!",
        "processing_complete": "Processing complete!",
        "processing_failed": "Processing failed with exit code {code}",
        "no_evals": "No evaluations found.",
        "no_evals_export": "No evaluations to export.",
        "exported": "Exported {count} evaluations to CSV.",
        "autostart_label": "Auto-start llama-server",
        "autostart_info": "Automatically start the server if not running",
    },
    "it": {
        "title": "## Valutazione Prestazioni VAD",
        "subtitle": "Elabora nuovi video o rivedi i risultati delle analisi esistenti.",
        "tab_process": "Elabora Video",
        "tab_evaluate": "Valuta Run",
        "process_header": "### Elabora Video Non Processati",
        "select_video": "Seleziona video da elaborare",
        "unprocessed": "non elaborati",
        "refresh": "Aggiorna",
        "refresh_short": "Agg.",
        "llama_url": "URL Server Llama",
        "llama_model": "Modello Llama",
        "start_processing": "Avvia Elaborazione",
        "processing_logs": "Log Elaborazione",
        "evaluate_header": "### Rivedi e Valuta i Risultati",
        "select_run": "Seleziona una run da valutare",
        "status": "Stato",
        "select_run_begin": "Seleziona una run per iniziare.",
        "visualization_header": "### Visualizzazione Analisi",
        "video_player": "Video Originale",
        "window_frames": "Finestra di Analisi",
        "no_windows": "Nessuna finestra analizzata.",
        "select_run_details": "Seleziona una run per vedere i dettagli",
        "timeline_header": "### Timeline Score Anomalia",
        "timeline_hint": "Clicca sul grafico per navigare a una finestra specifica",
        "anomaly_plot": "Grafico Score Anomalia",
        "window_nav": "Navigazione Finestre",
        "window_nav_info": "Trascina per navigare tra le finestre di analisi",
        "rate_header": "### Valuta la Qualità dell'Analisi",
        "rate_scale": "1 = Pessimo | 2 = Scarso | 3 = Medio | 4 = Buono | 5 = Eccellente",
        "score_consistency": "Coerenza Score Anomalia",
        "score_consistency_info": "Gli score hanno senso e sono coerenti?",
        "description_quality": "Qualità Descrizioni",
        "description_quality_info": "Le descrizioni sono accurate?",
        "detection_accuracy": "Accuratezza Rilevamento",
        "detection_accuracy_info": "Identifica correttamente le anomalie?",
        "overall_quality": "Qualità Complessiva",
        "overall_quality_info": "La tua impressione generale.",
        "notes": "Note (opzionale)",
        "notes_placeholder": "Eventuali osservazioni aggiuntive...",
        "submit": "Invia Valutazione",
        "result": "Risultato",
        "view_all": "Tutte le Valutazioni",
        "load_summary": "Carica Riepilogo",
        "eval_file": "File valutazioni",
        "export_csv": "Esporta CSV",
        "csv_status": "Stato Esportazione",
        "csv_path": "Percorso file CSV",
        "language": "Lingua",
        "upload_video": "Carica video",
        "upload_video_info": "Trascina qui",
        "video_uploaded": "Video caricato: {name}",
        "autostart_label": "Avvia llama-server automaticamente",
        "autostart_info": "Avvia il server automaticamente se non in esecuzione",
        "current_window": "Finestra Corrente",
        "score": "Score",
        "no_anomalies": "Il modello non ha rilevato anomalie sopra soglia.",
        "anomaly_detected": "Il modello ha rilevato un'anomalia nel video da {start}s a {end}s. Motivo: {reason}",
        "no_description": "Nessuna descrizione disponibile.",
        "error_no_video": "Errore: Seleziona prima un video.",
        "error_no_run": "Errore: Nessuna run selezionata.",
        "eval_saved": "Valutazione salvata per {run}!",
        "processing_complete": "Elaborazione completata!",
        "processing_failed": "Elaborazione fallita con codice {code}",
        "no_evals": "Nessuna valutazione trovata.",
        "no_evals_export": "Nessuna valutazione da esportare.",
        "exported": "Esportate {count} valutazioni in CSV.",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """Get translation for key in specified language."""
    text = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# ------------------------------
# LLM summarization
# ------------------------------

# Default LLM config (same as processing)
DEFAULT_LLM_URL = "http://localhost:8080"
DEFAULT_LLM_MODEL = "unsloth/InternVL3-2B-GGUF:UD-Q4_K_XL"
DEFAULT_LLAMA_CTX_LEN = 8192
DEFAULT_LLAMA_BATCH = 1024
DEFAULT_LLAMA_UBATCH = 256
DEFAULT_LLAMA_PARALLEL = 2
DEFAULT_ANALYSIS_FPS = 2.0
DEFAULT_CHUNK_SECONDS = 600
DEFAULT_CAPTION_WINDOW_SIZE = 6
DEFAULT_CAPTION_PARALLEL_WINDOWS = 2
DEFAULT_CHUNK_WORKERS = 1
DEFAULT_FRAME_SAVE_EXT = "jpg"
DEFAULT_IMAGE_QUALITY = 85

# Cache for summaries (run_name -> summary)
_summary_cache: Dict[str, str] = {}

# Global variable to track if server was started by us
_llama_server_process: Optional[subprocess.Popen] = None
_server_started: bool = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower().lstrip(".")
    return value if value in choices else default


def _is_server_running(url: str = DEFAULT_LLM_URL) -> bool:
    """Check if the LLM server is running."""
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except:
        try:
            # Some servers don't have /health, try /v1/models
            r = requests.get(f"{url}/v1/models", timeout=2)
            return r.status_code == 200
        except:
            return False


def _start_llama_server(
    model_name: str = DEFAULT_LLM_MODEL,
    port: int = 8080,
    ngl: int = 999,
    ctx_len: int = DEFAULT_LLAMA_CTX_LEN,
    server_bin: Optional[str] = None,
) -> bool:
    """Start llama-server if not already running. Uses -hf flag to download from HuggingFace."""
    global _llama_server_process, _server_started
    
    if _is_server_running(f"http://localhost:{port}"):
        print(f"[LLM Server] Server already running on port {port}")
        return True
    
    print(f"[LLM Server] Starting llama-server with model: {model_name}")
    server_bin = server_bin or os.environ.get("MUVAD_LLAMA_SERVER_CMD", "llama-server")
    ctx_len = _env_int("MUVAD_LLAMA_CTX_LEN", int(ctx_len))
    parallel_slots = _env_int("MUVAD_LLAMA_NP", DEFAULT_LLAMA_PARALLEL)
    batch_size = _env_int("MUVAD_LLAMA_BATCH", DEFAULT_LLAMA_BATCH)
    ubatch_size = _env_int("MUVAD_LLAMA_UBATCH", DEFAULT_LLAMA_UBATCH)
    cont_batching = _env_bool("MUVAD_LLAMA_CONT_BATCHING", True)
    flash_attn = _env_bool("MUVAD_LLAMA_FLASH_ATTN", True)
    
    # Create log file
    log_path = REPO_ROOT / "tmp" / "llama_server_summary.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = [
            server_bin,
            "-hf", model_name,  # Use -hf flag to download from HuggingFace
            "--port", str(port),
            "-ngl", str(ngl),
            "--ctx-size", str(ctx_len),
            "-np", str(parallel_slots),
            "-b", str(batch_size),
            "-ub", str(ubatch_size),
            "--host", "0.0.0.0",
        ]
        if cont_batching:
            cmd.append("--cont-batching")
        if flash_attn:
            cmd += ["--flash-attn", "on"]
        
        print(f"[LLM Server] Command: {' '.join(cmd)}")
        print(f"[LLM Server] Log file: {log_path}")
        
        log_f = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
        _llama_server_process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        
        # Wait for server to start (longer timeout for first download)
        print("[LLM Server] Waiting for server to start (may take longer if downloading model)...")
        for i in range(120):  # Wait up to 2 minutes
            time.sleep(1)
            if _is_server_running(f"http://localhost:{port}"):
                print(f"[LLM Server] Server started successfully on port {port}")
                _server_started = True
                return True
            if i % 10 == 0 and i > 0:
                print(f"[LLM Server] Still waiting... ({i}s)")
        
        print("[LLM Server] Server failed to start within timeout. Check log file.")
        return False
        
    except FileNotFoundError:
        print("[LLM Server] llama-server not found. Please install llama.cpp or start server manually.")
        print("[LLM Server] You can install it with: pip install llama-cpp-python[server]")
        return False
    except Exception as e:
        print(f"[LLM Server] Error starting server: {e}")
        return False


def _ensure_server_running(model_name: str = DEFAULT_LLM_MODEL) -> bool:
    """Ensure the LLM server is running, start if needed."""
    if _is_server_running():
        return True
    return _start_llama_server(model_name=model_name)


def text_chat(
    prompt: str,
    base_url: str = DEFAULT_LLM_URL,
    model: str = DEFAULT_LLM_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.3,
    timeout: int = 60,
    auto_start: bool = True,
) -> Optional[str]:
    """Call LLM for text-only chat (no images)."""
    # Try to start server if not running
    if auto_start and not _is_server_running(base_url):
        if not _ensure_server_running():
            print("[LLM Error] Server not available and could not be started")
            return None
    
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = requests.post(
            f"{base_url}/v1/chat/completions", 
            json=payload, 
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"[LLM Error] {e}")
        return None


def translate_text(
    text: str,
    target_lang: str = "it",
    base_url: str = DEFAULT_LLM_URL,
    model: str = DEFAULT_LLM_MODEL,
) -> str:
    """Translate text to the target language using LLM."""
    if not text or target_lang == "en":
        return text
    if not _is_server_running(base_url):
        return text
    
    lang_name = "Italian" if target_lang == "it" else target_lang
    prompt = f"""Translate the following text to {lang_name}. 
Only output the translation, nothing else.

Text to translate:
{text}"""

    translated = text_chat(prompt, base_url=base_url, model=model, max_tokens=500, auto_start=False)
    
    if translated:
        return translated.strip()
    return text


def summarize_anomaly_descriptions(
    descriptions: List[str],
    start_ts: float,
    end_ts: float,
    base_url: str = DEFAULT_LLM_URL,
    model: str = DEFAULT_LLM_MODEL,
    lang: str = "en",
) -> str:
    """Use LLM to summarize multiple anomaly descriptions into one coherent summary."""
    if not descriptions:
        return "Nessuna descrizione dell'anomalia disponibile." if lang == "it" else "No anomaly descriptions available."
    
    # Language instruction
    lang_instruction = "Scrivi in italiano." if lang == "it" else "Write in English."
    
    # If only one description, translate if needed
    if len(descriptions) == 1:
        if lang == "it":
            return translate_text(descriptions[0], "it", base_url, model)
        return descriptions[0]

    if not _is_server_running(base_url):
        return " ".join(descriptions)
    
    # Build prompt for summarization
    all_descs = "\n".join(f"- {d}" for d in descriptions)
    prompt = f"""You are analyzing a video surveillance system's anomaly detection results.
The system detected anomalies between {start_ts:.1f}s and {end_ts:.1f}s in the video.

Here are the individual frame-by-frame descriptions from the detection system:
{all_descs}

Please provide a single, coherent summary (2-3 sentences max) describing what anomaly was detected in the video. Be concise and focus on the main event. {lang_instruction}"""

    summary = text_chat(prompt, base_url=base_url, model=model, auto_start=False)
    
    if summary:
        return summary.strip()
    else:
        # Fallback: just return concatenated descriptions (translated if needed)
        fallback = " ".join(descriptions)
        if lang == "it":
            return translate_text(fallback, "it", base_url, model)
        return fallback


# ------------------------------
# Video processing functions
# ------------------------------


def _get_all_data_videos() -> List[Path]:
    """Get all video files from the data directory."""
    if not DATA_DIR.exists():
        return []

    videos: List[Path] = []

    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    return sorted(videos)


def _list_video_files(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _get_processed_video_names() -> set:
    """Get names of videos that have already been processed."""
    processed = set()
    if not RUNS_DIR.exists():
        return processed

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        output_dir = run_dir / "output"
        if not output_dir.exists():
            continue
        # Check for inference data JSON
        for json_file in output_dir.glob("*_inference_data.json"):
            # Extract video name from JSON filename
            # e.g., "Abuse028_x264_inference_data.json" -> "Abuse028_x264"
            stem = json_file.stem.replace("_inference_data", "")
            processed.add(stem)

    return processed


def _get_unprocessed_videos() -> List[Tuple[str, Path]]:
    """Get list of videos that haven't been processed yet."""
    all_videos = _get_all_data_videos()
    processed = _get_processed_video_names()

    unprocessed = []
    for video_path in all_videos:
        video_stem = video_path.stem
        if video_stem not in processed:
            # Create a display name with relative path
            try:
                rel_path = video_path.relative_to(DATA_DIR)
                display_name = str(rel_path)
            except ValueError:
                display_name = video_path.name
            unprocessed.append((display_name, video_path))

    return unprocessed


def _safe_runtime_stem(value: str, max_length: int = MAX_RUNTIME_STEM_LENGTH) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    stem = re.sub(r"_+", "_", stem).strip("._-")
    if not stem:
        stem = "video"
    if len(stem) <= max_length:
        return stem
    digest = hashlib.sha1(str(value).encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{stem[: max_length - 11].rstrip('._-')}_{digest}"


def _safe_runtime_filename(path: Path) -> str:
    suffix = path.suffix if path.suffix.lower() in VIDEO_EXTENSIONS else ".mp4"
    return f"{_safe_runtime_stem(path.stem)}{suffix}"


def _next_run_dir(video_stem: str) -> Path:
    """Allocate a run directory name without collisions."""
    safe_stem = _safe_runtime_stem(video_stem)
    run_dir = RUNS_DIR / safe_stem
    if run_dir.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{safe_stem}_{timestamp}"
    return run_dir


def _normalize_llama_host(url: str) -> str:
    return (url or DEFAULT_LLM_URL).strip().rstrip("/")


def _read_video_metadata(video_path: Path) -> Tuple[float, float, int]:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return 30.0, 0.0, 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0, 0.0, 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = float(frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        return fps, duration, frame_count
    finally:
        cap.release()


def _selected_frame_timestamp(path: Path) -> float:
    raw = path.stem.removeprefix("frame_")
    try:
        return float(raw)
    except ValueError:
        match = re.search(r"(\d+(?:\.\d+)?)", raw)
        return float(match.group(1)) if match else 0.0


def _parse_json_object(text: str) -> Dict[str, Any]:
    for raw in re.findall(r"\{[^\<\>]*?\}", text, flags=re.DOTALL):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _captioner_blocks(captioner_text: str) -> List[str]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in captioner_text.splitlines():
        if line.startswith("- frames: ") and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(block).strip() for block in blocks if any(line.strip() for line in block)]


def _window_frame_names(block: str, selected_frames: List[Path]) -> List[str]:
    frame_line = next((line for line in block.splitlines() if line.startswith("- frames: ")), "")
    raw_timestamps = frame_line.removeprefix("- frames: ").split(",")
    timestamps: List[float] = []
    for raw_ts in raw_timestamps:
        raw_ts = raw_ts.strip()
        if not raw_ts:
            continue
        try:
            timestamps.append(float(raw_ts))
        except ValueError:
            continue

    by_ts = {round(_selected_frame_timestamp(path), 3): path.name for path in selected_frames}
    ordered_ts = sorted(by_ts)
    names: List[str] = []
    for ts in timestamps:
        key = round(ts, 3)
        name = by_ts.get(key)
        if name is None and ordered_ts:
            nearest = min(ordered_ts, key=lambda candidate: abs(candidate - key))
            if abs(nearest - key) <= 0.002:
                name = by_ts[nearest]
        if name:
            names.append(name)
    return names


def _compute_segments_from_records(records: List[Dict[str, Any]], threshold: float) -> List[List[int]]:
    segments: List[List[int]] = []
    start: Optional[int] = None
    for idx, record in enumerate(records):
        is_anomalous = bool(record.get("is_anomalous")) or float(record.get("anomaly_score", 0.0)) >= threshold
        if is_anomalous and start is None:
            start = idx
        is_last = idx == len(records) - 1
        if start is not None and ((not is_anomalous) or is_last):
            end = idx if is_anomalous and is_last else idx - 1
            segments.append([
                int(records[start].get("frame_index_center", start)),
                int(records[end].get("frame_index_center", end)),
            ])
            start = None
    return segments


def _save_async_inference_data(
    *,
    run_dir: Path,
    video_name: str,
    prompt: str,
    threshold: float,
    window_step: int,
    video_total_s: float,
) -> Path:
    output_dir = run_dir / "output"
    selected_dir = run_dir / "frames_selected"
    captioner_path = output_dir / "captioner.txt"
    selected_frames = sorted(selected_dir.glob("frame_*.*"), key=_selected_frame_timestamp)
    selected_index = {path.name: idx for idx, path in enumerate(selected_frames)}

    captioner_text = captioner_path.read_text(encoding="utf-8", errors="ignore") if captioner_path.exists() else ""
    records: List[Dict[str, Any]] = []
    for block in _captioner_blocks(captioner_text):
        frame_names = _window_frame_names(block, selected_frames)
        if not frame_names:
            continue

        parsed = _parse_json_object(block)
        try:
            score = float(parsed.get("anomaly_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        description = str(parsed.get("description", "") or "").strip()

        start_name = frame_names[0]
        center_name = frame_names[len(frame_names) // 2]
        end_name = frame_names[-1]
        start_idx = int(selected_index.get(start_name, 0))
        center_idx = int(selected_index.get(center_name, start_idx))
        end_idx = int(selected_index.get(end_name, center_idx))
        start_ts = _selected_frame_timestamp(selected_dir / start_name)
        center_ts = _selected_frame_timestamp(selected_dir / center_name)
        end_ts = _selected_frame_timestamp(selected_dir / end_name)

        records.append({
            "frame_names": frame_names,
            "frame_name_start": start_name,
            "frame_name_center": center_name,
            "frame_name_end": end_name,
            "frame_index_start": start_idx,
            "frame_index_center": center_idx,
            "frame_index_end": end_idx,
            "timestamp_start": start_ts,
            "timestamp_center": center_ts,
            "timestamp_end": end_ts,
            "anomaly_score": score,
            "description": description,
            "is_anomalous": score >= float(threshold),
            "raw": block,
        })

    data = {
        "video_name": video_name,
        "video_total_s": video_total_s,
        "threshold": float(threshold),
        "window_step": int(window_step),
        "prompt": prompt,
        "segments": _compute_segments_from_records(records, float(threshold)),
        "selected_frames": [
            {"name": path.name, "timestamp": _selected_frame_timestamp(path)}
            for path in selected_frames
        ],
        "windows": records,
        "pipeline": "async_main_workflow",
    }

    json_path = output_dir / f"{video_name}_inference_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return json_path


def _build_async_pipeline_config(
    *,
    video_path: Path,
    run_dir: Path,
    llama_url: str,
    llama_model: str,
    prompt: str,
    autostart_server: bool,
    video_fps: float,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    llama_parallel_slots: Optional[int] = None,
    caption_parallel_windows: Optional[int] = None,
    llama_ctx_len: Optional[int] = None,
    llama_batch_size: Optional[int] = None,
    llama_ubatch_size: Optional[int] = None,
    caption_window_size: Optional[int] = None,
    llama_flash_attn: Optional[bool] = None,
    llama_cont_batching: Optional[bool] = None,
) -> Dict[str, Any]:
    from scripts.prediction.workflow import read_config

    try:
        config = read_config("config.yml")
    except Exception:
        config = {}

    output_dir = run_dir / "output"
    select_fps = _env_float("MUVAD_ANALYSIS_FPS", DEFAULT_ANALYSIS_FPS)
    selector_stride = max(1, int(round(float(video_fps or 30.0) / select_fps)))
    normalized_llama_url = _normalize_llama_host(llama_url)
    server_already_running = _is_server_running(normalized_llama_url)

    captioner_params = dict(config.get("captioner", {}).get("parameters", {}) or {})
    ctx_len = int(llama_ctx_len) if llama_ctx_len else _env_int("MUVAD_LLAMA_CTX_LEN", DEFAULT_LLAMA_CTX_LEN)
    batch_size = int(llama_batch_size) if llama_batch_size else _env_int("MUVAD_LLAMA_BATCH", DEFAULT_LLAMA_BATCH)
    ubatch_size = int(llama_ubatch_size) if llama_ubatch_size else _env_int("MUVAD_LLAMA_UBATCH", DEFAULT_LLAMA_UBATCH)
    parallel_slots = int(llama_parallel_slots) if llama_parallel_slots else _env_int("MUVAD_LLAMA_NP", DEFAULT_LLAMA_PARALLEL)
    caption_window_size = int(caption_window_size) if caption_window_size else _env_int("MUVAD_CAPTION_WINDOW_SIZE", DEFAULT_CAPTION_WINDOW_SIZE)
    parallel_windows = int(caption_parallel_windows) if caption_parallel_windows else _env_int("MUVAD_CAPTION_PARALLEL_WINDOWS", DEFAULT_CAPTION_PARALLEL_WINDOWS)
    flash_attn = _env_bool("MUVAD_LLAMA_FLASH_ATTN", True) if llama_flash_attn is None else bool(llama_flash_attn)
    cont_batching = _env_bool("MUVAD_LLAMA_CONT_BATCHING", True) if llama_cont_batching is None else bool(llama_cont_batching)
    ctx_len = max(1024, ctx_len)
    batch_size = max(1, batch_size)
    ubatch_size = max(1, ubatch_size)
    parallel_slots = max(1, parallel_slots)
    caption_window_size = max(1, caption_window_size)
    parallel_windows = max(1, parallel_windows)
    frame_save_ext = _env_choice("MUVAD_FRAME_SAVE_EXT", DEFAULT_FRAME_SAVE_EXT, {"png", "jpg", "jpeg", "webp"})
    image_quality = _env_int("MUVAD_IMAGE_QUALITY", DEFAULT_IMAGE_QUALITY)

    # The base config is tuned for large offline runs (np=10, ctx_len=81920).
    # The Gradio demo runs one request stream and should not inherit those VRAM-heavy defaults.
    captioner_params.update({
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 0.0,
        "cont_batching": cont_batching,
        "flash_attn": flash_attn,
        "ngl": captioner_params.get("ngl", 999),
        "ctx_len": ctx_len,
        "batch": batch_size,
        "ubatch": ubatch_size,
        "np": parallel_slots,
        "image_format": frame_save_ext.upper().replace("JPG", "JPEG"),
        "image_quality": image_quality,
        "autostart": bool(autostart_server) and not server_already_running,
        "ready_timeout": 300.0 if autostart_server else 20.0,
        "log_file": str(output_dir / "llama_server.log"),
    })

    return {
        "evaluate": config.get("evaluate", {}),
        "extractor": {
            **config.get("extractor", {}),
            "video_url": str(video_path),
            "timeout": 0.0,
            "resize": [448, 448],
            "frame_stride": selector_stride,
            "start_time": float(start_time),
            "end_time": float(end_time) if end_time is not None else None,
            "save_dir": "",
            "log": "INFO",
        },
        "selector": {
            **config.get("selector", {}),
            "batch_size": 1,
            "save_dir": str(run_dir / "frames_selected"),
            "save_ext": frame_save_ext,
            "jpeg_quality": image_quality,
            "log": "INFO",
        },
        "captioner": {
            **config.get("captioner", {}),
            "model_name": llama_model,
            "prompt": prompt,
            "batch_size": caption_window_size * parallel_windows,
            "warmup_timeout": 20,
            "random_seed": 1337,
            "aggregate": True,
            "aggregate_window_size": caption_window_size,
            "aggregate_max_workers": parallel_windows,
            "aggregate_frames_tag": "FramesCount",
            "aggregate_timestamp_joiner": ", ",
            "parameters": captioner_params,
            "backend": "llamacpp",
            "host": normalized_llama_url,
            "save_file": str(output_dir / "captioner.txt"),
            "log": "INFO",
        },
        "detector": {
            **config.get("detector", {}),
            "model_name": None,
            "prompt": "",
            "batch_size": 1,
            "parameters": {},
            "save_file": str(output_dir / "detector.txt"),
            "log": "INFO",
        },
        "notifier": {
            **config.get("notifier", {}),
            "threshold": 0.5,
            "result_key": "anomaly_score",
            "description_key": "description",
            "decision_mode": "moving_average",
            "avg_window_size": 1,
            "consecutive_required": 1,
            "log": "INFO",
        },
        "log": "INFO",
    }


def _chunk_ranges(video_total_s: float, chunk_seconds: int) -> List[Tuple[int, float, float]]:
    if video_total_s <= 0 or chunk_seconds <= 0 or video_total_s <= chunk_seconds:
        return [(0, 0.0, video_total_s if video_total_s > 0 else 0.0)]

    chunks: List[Tuple[int, float, float]] = []
    start = 0.0
    idx = 0
    while start < video_total_s:
        end = min(video_total_s, start + float(chunk_seconds))
        chunks.append((idx, start, end))
        start = end
        idx += 1
    return chunks


def _chunk_run_dir(run_dir: Path, chunk_idx: int) -> Path:
    return run_dir / "chunks" / f"chunk_{chunk_idx:04d}"


def _chunk_inference_path(chunk_dir: Path) -> Optional[Path]:
    output_dir = chunk_dir / "output"
    if not output_dir.exists():
        return None
    json_files = list(output_dir.glob("*_inference_data.json"))
    return json_files[0] if json_files else None


def _load_run_inference_data(run_dir: Path) -> Optional[Dict[str, Any]]:
    output_dir = run_dir / "output"
    if not output_dir.exists():
        return None
    json_files = list(output_dir.glob("*_inference_data.json"))
    if not json_files:
        return None
    try:
        return json.loads(json_files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _short_log_text(text: str, max_chars: int = 140) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def _new_caption_feedback_lines(
    captioner_path: Path,
    *,
    chunk_label: str,
    start_index: int,
) -> Tuple[List[str], int]:
    if not captioner_path.exists():
        return [], start_index

    try:
        captioner_text = captioner_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [], start_index

    blocks = _captioner_blocks(captioner_text)
    if start_index >= len(blocks):
        return [], len(blocks)

    lines: List[str] = []
    for idx, block in enumerate(blocks[start_index:], start=start_index + 1):
        frame_line = next((line for line in block.splitlines() if line.startswith("- frames: ")), "")
        frames = frame_line.removeprefix("- frames: ").strip() or "unknown frames"
        parsed = _parse_json_object(block)
        try:
            score = float(parsed.get("anomaly_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        description = _short_log_text(str(parsed.get("description", "") or ""))
        detail = f" | {description}" if description else ""
        lines.append(f"{chunk_label}: window {idx} ready | frames {frames} | score={score:.2f}{detail}\n")

    return lines, len(blocks)


def _clear_generated_run_outputs(run_dir: Path) -> None:
    for path in (run_dir / "frames_selected").glob("frame_*.*"):
        try:
            path.unlink()
        except OSError:
            pass
    for path in (run_dir / "output").glob("*_inference_data.json"):
        try:
            path.unlink()
        except OSError:
            pass
    captioner_path = run_dir / "output" / "captioner.txt"
    if captioner_path.exists():
        try:
            captioner_path.unlink()
        except OSError:
            pass


def _merge_chunk_results(
    *,
    run_dir: Path,
    video_name: str,
    prompt: str,
    threshold: float,
    window_step: int,
    video_total_s: float,
) -> Optional[Path]:
    chunk_json_paths = sorted((run_dir / "chunks").glob("chunk_*/output/*_inference_data.json"))
    if not chunk_json_paths:
        return None

    output_dir = run_dir / "output"
    frames_selected_dir = run_dir / "frames_selected"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_selected_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_run_outputs(run_dir)

    records: List[Dict[str, Any]] = []
    captioner_blocks: List[str] = []
    for json_path in chunk_json_paths:
        try:
            chunk_data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        records.extend(chunk_data.get("windows", []) or [])
        captioner_path = json_path.parent / "captioner.txt"
        if captioner_path.exists():
            try:
                captioner_blocks.append(captioner_path.read_text(encoding="utf-8", errors="ignore").strip())
            except OSError:
                pass

        source_frames_dir = json_path.parent.parent / "frames_selected"
        for frame_path in source_frames_dir.glob("frame_*.*"):
            target = frames_selected_dir / frame_path.name
            if not target.exists():
                shutil.copy2(frame_path, target)

    selected_frames = sorted(frames_selected_dir.glob("frame_*.*"), key=_selected_frame_timestamp)
    selected_index = {path.name: idx for idx, path in enumerate(selected_frames)}

    for record in records:
        frame_names = [name for name in record.get("frame_names", []) if name in selected_index]
        if not frame_names:
            continue
        start_name = frame_names[0]
        center_name = frame_names[len(frame_names) // 2]
        end_name = frame_names[-1]
        record["frame_names"] = frame_names
        record["frame_name_start"] = start_name
        record["frame_name_center"] = center_name
        record["frame_name_end"] = end_name
        record["frame_index_start"] = int(selected_index[start_name])
        record["frame_index_center"] = int(selected_index[center_name])
        record["frame_index_end"] = int(selected_index[end_name])

    records = sorted(
        records,
        key=lambda record: float(record.get("timestamp_center", record.get("timestamp_start", 0.0)) or 0.0),
    )
    merged_captioner = "\n".join(block for block in captioner_blocks if block)
    if merged_captioner:
        (output_dir / "captioner.txt").write_text(merged_captioner + "\n", encoding="utf-8")

    data = {
        "video_name": video_name,
        "video_total_s": video_total_s,
        "threshold": float(threshold),
        "window_step": int(window_step),
        "prompt": prompt,
        "segments": _compute_segments_from_records(records, float(threshold)),
        "selected_frames": [
            {"name": path.name, "timestamp": _selected_frame_timestamp(path)}
            for path in selected_frames
        ],
        "windows": records,
        "pipeline": "async_main_workflow_chunked",
        "chunks": len(chunk_json_paths),
    }

    json_path = output_dir / f"{video_name}_inference_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return json_path


def _run_video_processing(
    video_path: str,
    llama_url: str,
    llama_model: str,
    prompt: str,
    autostart_server: bool = True,
    run_dir: Optional[Path] = None,
    llama_parallel_slots: Optional[int] = None,
    caption_parallel_windows: Optional[int] = None,
    llama_ctx_len: Optional[int] = None,
    llama_batch_size: Optional[int] = None,
    llama_ubatch_size: Optional[int] = None,
    caption_window_size: Optional[int] = None,
    llama_flash_attn: Optional[bool] = None,
    llama_cont_batching: Optional[bool] = None,
    chunk_workers_override: Optional[int] = None,
) -> Iterator[str]:
    """Run the VAD pipeline on a video and yield log output."""
    if not video_path:
        yield "Error: No video selected."
        return

    video = Path(video_path)
    if not video.exists():
        yield f"Error: Video not found: {video_path}"
        return

    video_stem = _safe_runtime_stem(video.stem)
    run_dir = run_dir or _next_run_dir(video_stem)
    run_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = run_dir / "videos"
    output_dir = run_dir / "output"
    frames_selected_dir = run_dir / "frames_selected"
    for directory in (videos_dir, output_dir, frames_selected_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dest_video = videos_dir / _safe_runtime_filename(video)
    if dest_video.resolve() != video.resolve():
        try:
            shutil.copy2(video, dest_video)
        except OSError as exc:
            yield f"Error: could not stage video for processing: {exc}\n"
            return

    video_fps, video_total_s, video_frame_count = _read_video_metadata(dest_video)
    chunk_seconds = _env_int("MUVAD_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS)
    chunk_workers = int(chunk_workers_override) if chunk_workers_override else _env_int("MUVAD_CHUNK_WORKERS", DEFAULT_CHUNK_WORKERS)
    chunk_workers = max(1, chunk_workers)
    chunks = _chunk_ranges(video_total_s, chunk_seconds)
    analysis_fps = _env_float("MUVAD_ANALYSIS_FPS", DEFAULT_ANALYSIS_FPS)
    frame_stride = max(1, int(round(float(video_fps or 30.0) / analysis_fps)))
    preview_config = _build_async_pipeline_config(
        video_path=dest_video,
        run_dir=run_dir,
        llama_url=llama_url,
        llama_model=llama_model,
        prompt=prompt,
        autostart_server=autostart_server,
        video_fps=video_fps,
        llama_parallel_slots=llama_parallel_slots,
        caption_parallel_windows=caption_parallel_windows,
        llama_ctx_len=llama_ctx_len,
        llama_batch_size=llama_batch_size,
        llama_ubatch_size=llama_ubatch_size,
        caption_window_size=caption_window_size,
        llama_flash_attn=llama_flash_attn,
        llama_cont_batching=llama_cont_batching,
    )

    yield f"Starting processing: {video.name}\n"
    yield f"Run directory: {run_dir}\n"
    yield "Pipeline: async main workflow (Extractor -> Selector -> Captioner -> Detector -> Notifier)\n"
    yield f"Video FPS: {video_fps:.2f} | Frames: {video_frame_count} | Duration: {video_total_s:.2f}s\n"
    yield f"Extractor sampling: target_fps={analysis_fps:g}, frame_stride={frame_stride} (skips frames before decoding to PIL)\n"
    if len(chunks) > 1:
        yield f"Chunking: enabled, {len(chunks)} chunks of up to {chunk_seconds}s; completed chunks are reused on resume.\n"
        yield f"Chunk workers: {chunk_workers} (set MUVAD_CHUNK_WORKERS to tune; 1 keeps real-time per-window logs)\n"
    else:
        yield "Chunking: disabled for this video length.\n"
    llama_params = preview_config["captioner"]["parameters"]
    caption_cfg = preview_config["captioner"]
    yield (
        "llama.cpp memory settings: "
        f"np={llama_params.get('np')}, "
        f"ctx_len={llama_params.get('ctx_len')}, "
        f"batch={llama_params.get('batch')}, "
        f"ubatch={llama_params.get('ubatch')}, "
        f"cont_batching={llama_params.get('cont_batching')}, "
        f"flash_attn={llama_params.get('flash_attn')}\n"
    )
    yield (
        "Caption throughput settings: "
        f"window_size={caption_cfg.get('aggregate_window_size')}, "
        f"parallel_windows={caption_cfg.get('aggregate_max_workers')}, "
        f"captioner_batch_size={caption_cfg.get('batch_size')}, "
        f"image_format={llama_params.get('image_format')}, "
        f"image_quality={llama_params.get('image_quality')}\n"
    )
    if _is_server_running(_normalize_llama_host(llama_url)):
        yield "llama.cpp server: already running; restart it to apply changed startup settings if it was launched with older values.\n"
    if autostart_server:
        yield "Autostart server: enabled\n\n"
    else:
        yield "Autostart server: disabled (ensure server is running)\n\n"

    accumulated = ""
    final_json_path: Optional[Path] = None
    any_anomalous = False

    def process_chunk_sync(chunk: Tuple[int, float, float], allow_autostart: bool = False) -> Tuple[int, bool, Path, bool]:
        chunk_idx, start_time, end_time = chunk
        active_run_dir = _chunk_run_dir(run_dir, chunk_idx)
        active_run_dir.mkdir(parents=True, exist_ok=True)
        for directory in (active_run_dir / "output", active_run_dir / "frames_selected"):
            directory.mkdir(parents=True, exist_ok=True)

        existing_chunk_json = _chunk_inference_path(active_run_dir)
        if existing_chunk_json is not None:
            return chunk_idx, False, existing_chunk_json, True

        config = _build_async_pipeline_config(
            video_path=dest_video,
            run_dir=active_run_dir,
            llama_url=llama_url,
            llama_model=llama_model,
            prompt=prompt,
            autostart_server=allow_autostart,
            video_fps=video_fps,
            start_time=start_time,
            end_time=end_time if video_total_s > 0 else None,
            llama_parallel_slots=llama_parallel_slots,
            caption_parallel_windows=caption_parallel_windows,
            llama_ctx_len=llama_ctx_len,
            llama_batch_size=llama_batch_size,
            llama_ubatch_size=llama_ubatch_size,
            caption_window_size=caption_window_size,
            llama_flash_attn=llama_flash_attn,
            llama_cont_batching=llama_cont_batching,
        )
        from scripts.prediction.workflow import initialize_modules, workflow

        modules = initialize_modules(config)
        result = bool(workflow(*modules, config))
        json_path = _save_async_inference_data(
            run_dir=active_run_dir,
            video_name=video_stem,
            prompt=prompt,
            threshold=float(config["notifier"]["threshold"]),
            window_step=int(config["captioner"]["aggregate_window_size"]),
            video_total_s=video_total_s,
        )
        return chunk_idx, result, json_path, False

    if len(chunks) > 1 and chunk_workers > 1:
        accumulated += f"Parallel chunk processing enabled with {chunk_workers} workers.\n"
        yield accumulated

        pending_chunks = list(chunks)
        if autostart_server and not _is_server_running(_normalize_llama_host(llama_url)):
            first_chunk = pending_chunks.pop(0)
            accumulated += "Starting first chunk alone to warm up llama.cpp before parallel workers...\n"
            yield accumulated
            try:
                chunk_idx, result, json_path, skipped = process_chunk_sync(first_chunk, allow_autostart=True)
            except Exception as exc:
                accumulated += f"\nProcessing failed: {exc}\n"
                yield accumulated
                return
            any_anomalous = any_anomalous or bool(result)
            final_json_path = _merge_chunk_results(
                run_dir=run_dir,
                video_name=video_stem,
                prompt=prompt,
                threshold=float(preview_config["notifier"]["threshold"]),
                window_step=int(preview_config["captioner"]["aggregate_window_size"]),
                video_total_s=video_total_s,
            )
            accumulated += f"Chunk {chunk_idx + 1}/{len(chunks)}: {'already processed' if skipped else 'complete'} ({json_path})\n"
            yield accumulated

        with ThreadPoolExecutor(max_workers=min(chunk_workers, len(pending_chunks))) as executor:
            futures = {executor.submit(process_chunk_sync, chunk): chunk for chunk in pending_chunks}
            for future in as_completed(futures):
                try:
                    chunk_idx, result, json_path, skipped = future.result()
                except Exception as exc:
                    accumulated += f"\nProcessing failed: {exc}\n"
                    yield accumulated
                    return
                any_anomalous = any_anomalous or bool(result)
                final_json_path = _merge_chunk_results(
                    run_dir=run_dir,
                    video_name=video_stem,
                    prompt=prompt,
                    threshold=float(preview_config["notifier"]["threshold"]),
                    window_step=int(preview_config["captioner"]["aggregate_window_size"]),
                    video_total_s=video_total_s,
                )
                data = _load_run_inference_data(run_dir)
                windows_count = len(data.get("windows", [])) if data else 0
                selected_count = len(data.get("selected_frames", [])) if data else len(list(frames_selected_dir.glob("frame_*.*")))
                accumulated += f"Chunk {chunk_idx + 1}/{len(chunks)}: {'already processed' if skipped else 'complete'} ({json_path})\n"
                accumulated += f"Merged progress: selected frames={selected_count} | analysis windows={windows_count}\n"
                yield accumulated

        data = _load_run_inference_data(run_dir)
        if data:
            any_anomalous = any(
                bool(window.get("is_anomalous")) or float(window.get("anomaly_score", 0.0)) >= float(data.get("threshold", 0.5))
                for window in data.get("windows", [])
            )
            accumulated += "\nProcessing complete!\n"
            accumulated += f"Prediction: {'anomalous' if any_anomalous else 'normal'}\n"
            accumulated += f"Selected frames: {len(data.get('selected_frames', []))} | Analysis windows: {len(data.get('windows', []))}\n"
            if final_json_path:
                accumulated += f"Inference data saved to: {final_json_path}\n"
        else:
            accumulated += "\nProcessing complete, but no inference data was generated.\n"
        yield accumulated
        return

    for chunk_idx, start_time, end_time in chunks:
        chunked = len(chunks) > 1
        active_run_dir = _chunk_run_dir(run_dir, chunk_idx) if chunked else run_dir
        active_run_dir.mkdir(parents=True, exist_ok=True)
        for directory in (active_run_dir / "output", active_run_dir / "frames_selected"):
            directory.mkdir(parents=True, exist_ok=True)

        existing_chunk_json = _chunk_inference_path(active_run_dir) if chunked else None
        chunk_label = (
            f"Chunk {chunk_idx + 1}/{len(chunks)} [{start_time:.1f}s - {end_time:.1f}s]"
            if chunked
            else "Full video"
        )

        if existing_chunk_json is not None:
            accumulated += f"{chunk_label}: already processed, skipping.\n"
            final_json_path = _merge_chunk_results(
                run_dir=run_dir,
                video_name=video_stem,
                prompt=prompt,
                threshold=float(preview_config["notifier"]["threshold"]),
                window_step=int(preview_config["captioner"]["aggregate_window_size"]),
                video_total_s=video_total_s,
            )
            yield accumulated
            continue

        config = _build_async_pipeline_config(
            video_path=dest_video,
            run_dir=active_run_dir,
            llama_url=llama_url,
            llama_model=llama_model,
            prompt=prompt,
            autostart_server=autostart_server,
            video_fps=video_fps,
            start_time=start_time,
            end_time=end_time if video_total_s > 0 else None,
            llama_parallel_slots=llama_parallel_slots,
            caption_parallel_windows=caption_parallel_windows,
            llama_ctx_len=llama_ctx_len,
            llama_batch_size=llama_batch_size,
            llama_ubatch_size=llama_ubatch_size,
            caption_window_size=caption_window_size,
            llama_flash_attn=llama_flash_attn,
            llama_cont_batching=llama_cont_batching,
        )

        status_queue: queue.Queue[Tuple[str, Any]] = queue.Queue()

        def run_worker() -> None:
            try:
                from scripts.prediction.workflow import initialize_modules, workflow

                status_queue.put(("log", f"{chunk_label}: initializing modules...\n"))
                modules = initialize_modules(config)
                status_queue.put(("log", f"{chunk_label}: modules ready.\n"))
                result = bool(workflow(*modules, config))
                json_path = _save_async_inference_data(
                    run_dir=active_run_dir,
                    video_name=video_stem,
                    prompt=prompt,
                    threshold=float(config["notifier"]["threshold"]),
                    window_step=int(config["captioner"]["aggregate_window_size"]),
                    video_total_s=video_total_s,
                )
                status_queue.put(("done", (result, json_path)))
            except Exception as exc:
                status_queue.put(("error", exc))

        worker = threading.Thread(target=run_worker, daemon=True)
        worker.start()

        last_progress = 0.0
        reported_windows = 0
        active_frames_selected_dir = active_run_dir / "frames_selected"
        active_output_dir = active_run_dir / "output"
        captioner_path = active_output_dir / "captioner.txt"
        while worker.is_alive() or not status_queue.empty():
            try:
                kind, payload = status_queue.get(timeout=0.5)
            except queue.Empty:
                new_lines, reported_windows = _new_caption_feedback_lines(
                    captioner_path,
                    chunk_label=chunk_label,
                    start_index=reported_windows,
                )
                if new_lines:
                    accumulated += "".join(new_lines)
                    yield accumulated
                    continue

                now = time.time()
                if now - last_progress >= 5.0:
                    selected_count = len(list(active_frames_selected_dir.glob("frame_*.*")))
                    accumulated += (
                        f"{chunk_label}: selected frames={selected_count}, "
                        f"completed windows={reported_windows}\n"
                    )
                    last_progress = now
                    yield accumulated
                continue

            if kind == "log":
                accumulated += str(payload)
                yield accumulated
            elif kind == "done":
                result, json_path = payload
                any_anomalous = any_anomalous or bool(result)
                new_lines, reported_windows = _new_caption_feedback_lines(
                    captioner_path,
                    chunk_label=chunk_label,
                    start_index=reported_windows,
                )
                if new_lines:
                    accumulated += "".join(new_lines)
                if chunked:
                    final_json_path = _merge_chunk_results(
                        run_dir=run_dir,
                        video_name=video_stem,
                        prompt=prompt,
                        threshold=float(config["notifier"]["threshold"]),
                        window_step=int(config["captioner"]["aggregate_window_size"]),
                        video_total_s=video_total_s,
                    )
                    merged_data = _load_run_inference_data(run_dir)
                    windows_count = len(merged_data.get("windows", [])) if merged_data else 0
                    selected_count = len(merged_data.get("selected_frames", [])) if merged_data else len(list(frames_selected_dir.glob("frame_*.*")))
                    accumulated += f"{chunk_label}: complete. Chunk data saved to: {json_path}\n"
                    accumulated += f"Merged progress: selected frames={selected_count} | analysis windows={windows_count}\n"
                    if final_json_path:
                        accumulated += f"Merged inference data saved to: {final_json_path}\n"
                else:
                    final_json_path = json_path
                    data = _load_run_inference_data(run_dir)
                    windows_count = len(data.get("windows", [])) if data else 0
                    selected_count = len(data.get("selected_frames", [])) if data else len(list(frames_selected_dir.glob("frame_*.*")))
                    accumulated += "\nProcessing complete!\n"
                    accumulated += f"Prediction: {'anomalous' if result else 'normal'}\n"
                    accumulated += f"Selected frames: {selected_count} | Analysis windows: {windows_count}\n"
                    accumulated += f"Inference data saved to: {json_path}\n"
                yield accumulated
            elif kind == "error":
                accumulated += f"\nProcessing failed: {payload}\n"
                yield accumulated
                return

        worker.join(timeout=1.0)
        if worker.is_alive():
            yield accumulated + "\nProcessing is still shutting down.\n"
            return

    if len(chunks) > 1:
        final_json_path = final_json_path or _merge_chunk_results(
            run_dir=run_dir,
            video_name=video_stem,
            prompt=prompt,
            threshold=float(preview_config["notifier"]["threshold"]),
            window_step=int(preview_config["captioner"]["aggregate_window_size"]),
            video_total_s=video_total_s,
        )
        data = _load_run_inference_data(run_dir)
        if data:
            any_anomalous = any(
                bool(window.get("is_anomalous")) or float(window.get("anomaly_score", 0.0)) >= float(data.get("threshold", 0.5))
                for window in data.get("windows", [])
            )
            accumulated += "\nProcessing complete!\n"
            accumulated += f"Prediction: {'anomalous' if any_anomalous else 'normal'}\n"
            accumulated += f"Selected frames: {len(data.get('selected_frames', []))} | Analysis windows: {len(data.get('windows', []))}\n"
        if final_json_path:
            accumulated += f"Inference data saved to: {final_json_path}\n"
        yield accumulated


# ------------------------------
# Data loading functions
# ------------------------------


def _get_available_runs() -> List[str]:
    """List all available run directories that have inference data."""
    if not RUNS_DIR.exists():
        return []

    runs: List[str] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        output_dir = run_dir / "output"
        if not output_dir.exists():
            continue
        # Check if there's inference data JSON
        json_files = list(output_dir.glob("*_inference_data.json"))
        if json_files:
            runs.append(run_dir.name)

    return runs


def _get_inference_data_path(run_name: str) -> Optional[Path]:
    """Get the inference data JSON path for a given run."""
    if not run_name:
        return None

    run_dir = RUNS_DIR / run_name
    output_dir = run_dir / "output"
    if not output_dir.exists():
        return None

    json_files = list(output_dir.glob("*_inference_data.json"))
    if json_files:
        return json_files[0]
    return None


def _get_frames_dir(run_name: str) -> Optional[Path]:
    """Get the selected frames directory for a given run."""
    if not run_name:
        return None

    run_dir = RUNS_DIR / run_name
    frames_dir = run_dir / "frames_selected"
    if frames_dir.exists():
        return frames_dir
    return None


def _get_video_path(run_name: str) -> Optional[Path]:
    """Get the original video path for a given run."""
    if not run_name:
        return None

    run_dir = RUNS_DIR / run_name
    
    # The original video is stored in the 'videos' subdirectory of the run
    videos_dir = run_dir / "videos"
    if videos_dir.exists():
        video_files = _list_video_files(videos_dir)
        if video_files:
            return video_files[0]
            
    # Fallback: check output directory for showcase if original not found
    output_dir = run_dir / "output"
    if output_dir.exists():
        showcase_files = list(output_dir.glob("*_vad_showcase.mp4"))
        if showcase_files:
            return showcase_files[0]
            
    return None


def _load_inference_data(run_name: str) -> Optional[Dict[str, Any]]:
    """Load inference data from JSON file."""
    json_path = _get_inference_data_path(run_name)
    if json_path is None or not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _load_evaluations() -> Dict[str, Any]:
    """Load existing evaluations from file."""
    if EVALUATIONS_FILE.exists():
        try:
            with open(EVALUATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"evaluations": []}


def _save_evaluation(evaluation: Dict[str, Any]) -> None:
    """Save an evaluation to the file."""
    data = _load_evaluations()
    data["evaluations"].append(evaluation)
    EVALUATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_existing_rating(run_name: str) -> Optional[Dict[str, Any]]:
    """Check if a run already has a rating."""
    if not run_name:
        return None
    data = _load_evaluations()
    for ev in data["evaluations"]:
        if ev.get("run_name") == run_name:
            return ev
    return None


# ------------------------------
# Image generation functions
# ------------------------------


def _create_grid_image(
    frames_dir: Path,
    frame_names: List[str],
    cols: int = 3,
    tile_size: int = 200,
) -> np.ndarray:
    """Create a grid image from frame names."""
    imgs = []
    for name in frame_names:
        frame_path = frames_dir / name
        if frame_path.exists():
            imgs.append(Image.open(frame_path).convert("RGB"))

    if not imgs:
        # Return a blank image
        blank = Image.new("RGB", (tile_size * cols, tile_size), (200, 200, 200))
        return np.array(blank)

    cols = min(cols, len(imgs))
    rows = ceil(len(imgs) / cols)

    # For 6 frames, use 3x2 layout
    if len(imgs) == 6:
        cols, rows = 3, 2

    W, H = cols * tile_size, rows * tile_size
    grid = Image.new("RGB", (W, H), (255, 255, 255))

    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        thumb = ImageOps.fit(im, (tile_size, tile_size), method=Image.LANCZOS)
        grid.paste(thumb, (c * tile_size, r * tile_size))

    return np.array(grid)


def _create_anomaly_plot(
    windows: List[Dict],
    segments: List[List[int]],
    threshold: float,
    current_window_idx: int,
) -> go.Figure:
    """Create anomaly score plot using Plotly with timestamps on x-axis."""
    # Use timestamps instead of frame numbers
    timestamp_centers = [w["timestamp_center"] for w in windows]
    scores = [w["anomaly_score"] for w in windows]
    
    # Build a mapping from frame_index_center to timestamp_center for segments
    frame_to_ts = {w["frame_index_center"]: w["timestamp_center"] for w in windows}
    # Also need start/end timestamps for segments
    frame_to_ts_start = {w["frame_index_center"]: w["timestamp_start"] for w in windows}
    frame_to_ts_end = {w["frame_index_center"]: w["timestamp_end"] for w in windows}
    
    fig = go.Figure()
    
    # Main score line
    fig.add_trace(go.Scatter(
        x=timestamp_centers, 
        y=scores, 
        mode='lines+markers',
        name='Anomaly Score',
        line=dict(color='#4A76FF', width=2),
        marker=dict(size=6),
        hovertemplate='Time: %{x:.1f}s<br>Score: %{y:.2f}<extra></extra>'
    ))
    
    # Threshold line
    fig.add_hline(
        y=threshold, 
        line_dash="dash", 
        line_color="gray", 
        annotation_text=f"Threshold ({threshold})", 
        annotation_position="top right"
    )
    
    # Highlight anomalous segments (convert frame indices to timestamps)
    for seg in segments:
        # Find closest windows for segment start/end
        seg_start_ts = frame_to_ts_start.get(seg[0])
        seg_end_ts = frame_to_ts_end.get(seg[1])
        # If not exact match, find closest
        if seg_start_ts is None:
            closest_start = min(frame_to_ts_start.keys(), key=lambda f: abs(f - seg[0]), default=None)
            seg_start_ts = frame_to_ts_start.get(closest_start, 0)
        if seg_end_ts is None:
            closest_end = min(frame_to_ts_end.keys(), key=lambda f: abs(f - seg[1]), default=None)
            seg_end_ts = frame_to_ts_end.get(closest_end, 0)
        if seg_start_ts is not None and seg_end_ts is not None:
            fig.add_vrect(
                x0=seg_start_ts, x1=seg_end_ts,
                fillcolor="red", opacity=0.15,
                layer="below", line_width=0,
            )

    # Highlight current window
    if 0 <= current_window_idx < len(windows):
        curr_ts = timestamp_centers[current_window_idx]
        curr_score = scores[current_window_idx]
        fig.add_trace(go.Scatter(
            x=[curr_ts], y=[curr_score],
            mode='markers',
            name='Current Window',
            marker=dict(color='red', size=12, line=dict(color='white', width=2)),
            hoverinfo='skip'
        ))
        # Vertical line for current window
        fig.add_vline(x=curr_ts, line_color="red", line_width=2, opacity=0.7)

    fig.update_layout(
        title="Anomaly Score Timeline (Click point to navigate)",
        xaxis_title="Time (seconds)",
        yaxis_title="Anomaly score",
        yaxis_range=[0, 1.05],
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        showlegend=False
    )
    
    return fig


# ------------------------------
# Summary helpers
# ------------------------------


def _build_anomaly_summary_html(
    windows: List[Dict[str, Any]], 
    threshold: float,
    run_name: str = "",
    use_llm: bool = True,
    lang: str = "en",
) -> str:
    """Build a prominent HTML classification summary with colors, optionally using LLM."""
    
    # Labels based on language
    labels = {
        "en": {
            "no_data": "No Analysis Data",
            "no_anomaly": "NO ANOMALY DETECTED",
            "analyzed": "The model analyzed <strong>{total} windows</strong> and found no anomalies above threshold ({threshold}).",
            "normal": "The video appears to show normal activity.",
            "anomaly": "ANOMALY DETECTED",
            "time": "TIME",
            "peak": "PEAK SCORE",
            "windows": "ANOMALOUS WINDOWS",
            "reason": "Reason",
            "no_desc": "No description available from the model.",
        },
        "it": {
            "no_data": "Nessun Dato di Analisi",
            "no_anomaly": "NESSUNA ANOMALIA RILEVATA",
            "analyzed": "Il modello ha analizzato <strong>{total} finestre</strong> e non ha trovato anomalie sopra la soglia ({threshold}).",
            "normal": "Il video mostra attività normale.",
            "anomaly": "ANOMALIA RILEVATA",
            "time": "TEMPO",
            "peak": "PUNTEGGIO MASSIMO",
            "windows": "FINESTRE ANOMALE",
            "reason": "Motivo",
            "no_desc": "Nessuna descrizione disponibile dal modello.",
        },
    }
    lbl = labels.get(lang, labels["en"])
    
    # Default/empty state
    if not windows:
        return f'''
        <div style="padding: 35px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 36px;">{lbl["no_data"]}</h2>
        </div>
        '''

    total_windows = len(windows)
    anomalous = [
        w for w in windows
        if bool(w.get("is_anomalous"))
        or float(w.get("anomaly_score", 0.0)) >= float(threshold)
    ]
    
    # NO ANOMALY - Green theme
    if not anomalous:
        analyzed_text = lbl["analyzed"].format(total=total_windows, threshold=threshold)
        return f'''
        <div style="padding: 35px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <h1 style="color: white; margin: 0 0 20px 0; font-size: 44px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                {lbl["no_anomaly"]}
            </h1>
            <p style="color: white; margin: 0; font-size: 22px; opacity: 0.95;">
                {analyzed_text}
            </p>
            <p style="color: white; margin: 15px 0 0 0; font-size: 20px; opacity: 0.85;">
                {lbl["normal"]}
            </p>
        </div>
        '''

    # ANOMALY DETECTED - Red theme
    start_ts = min(float(w.get("timestamp_start", 0.0)) for w in anomalous)
    end_ts = max(float(w.get("timestamp_end", 0.0)) for w in anomalous)
    max_score = max(float(w.get("anomaly_score", 0.0)) for w in anomalous)
    
    descs = [
        str(w.get("description", "")).strip()
        for w in anomalous
        if str(w.get("description", "")).strip()
    ]
    
    # Get reason text
    reason = lbl["no_desc"]
    if descs:
        cache_key = f"{run_name}_{threshold}_{lang}"
        if cache_key in _summary_cache:
            reason = _summary_cache[cache_key]
        elif use_llm and len(descs) > 1:
            llm_summary = summarize_anomaly_descriptions(descs, start_ts, end_ts, lang=lang)
            if llm_summary and llm_summary != " ".join(descs):
                _summary_cache[cache_key] = llm_summary
                reason = llm_summary
            else:
                reason = descs[0]
                if lang == "it":
                    reason = translate_text(reason, "it")
                _summary_cache[cache_key] = reason
        else:
            reason = descs[0]
            if lang == "it":
                reason = translate_text(reason, "it")
            _summary_cache[cache_key] = reason
    
    return f'''
    <div style="padding: 35px; background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h1 style="color: white; margin: 0 0 20px 0; font-size: 44px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            {lbl["anomaly"]}
        </h1>
        <div style="display: flex; gap: 25px; margin-bottom: 20px; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.2); padding: 15px 20px; border-radius: 10px;">
                <span style="color: white; font-size: 16px; opacity: 0.8;">{lbl["time"]}</span><br>
                <span style="color: white; font-size: 26px; font-weight: bold;">{start_ts:.1f}s - {end_ts:.1f}s</span>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px 20px; border-radius: 10px;">
                <span style="color: white; font-size: 16px; opacity: 0.8;">{lbl["peak"]}</span><br>
                <span style="color: white; font-size: 26px; font-weight: bold;">{max_score:.2f}</span>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px 20px; border-radius: 10px;">
                <span style="color: white; font-size: 16px; opacity: 0.8;">{lbl["windows"]}</span><br>
                <span style="color: white; font-size: 26px; font-weight: bold;">{len(anomalous)}/{total_windows}</span>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 12px; border-left: 5px solid white;">
            <p style="color: white; margin: 0; font-size: 22px; line-height: 1.6;">
                <strong>{lbl["reason"]}:</strong> {reason}
            </p>
        </div>
    </div>
    '''


def _build_anomaly_summary(
    windows: List[Dict[str, Any]], 
    threshold: float,
    run_name: str = "",
    use_llm: bool = True,
    lang: str = "en",
) -> str:
    """Wrapper that returns HTML summary."""
    return _build_anomaly_summary_html(windows, threshold, run_name, use_llm, lang)


def _format_seconds(seconds: Any) -> str:
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        value = 0.0
    minutes = int(value // 60)
    remaining = value - (minutes * 60)
    if minutes:
        return f"{minutes}m {remaining:04.1f}s"
    return f"{remaining:.1f}s"


def _score_style(score: float, threshold: float) -> Tuple[str, str]:
    if score >= threshold:
        return "#b42318", "#fee4e2"
    if score >= threshold * 0.75:
        return "#b54708", "#fef0c7"
    return "#027a48", "#dcfae6"


def _build_window_detail_html(
    window: Dict[str, Any],
    windows: List[Dict[str, Any]],
    window_idx: int,
    threshold: float,
    *,
    description: str = "",
    lang: str = "en",
) -> str:
    labels = {
        "en": {
            "current_window": "Current Window",
            "score": "Score",
            "threshold": "Threshold",
            "status": "Status",
            "anomalous": "Anomalous",
            "normal": "Normal",
            "time_range": "Time Range",
            "duration": "Duration",
            "trend": "Trend",
            "rank": "Score Rank",
            "frames": "Frames",
            "model_description": "Model Description",
            "no_description": "No description returned for this window.",
            "prev": "prev",
            "next": "next",
            "above": "above threshold",
            "below": "below threshold",
            "stable": "stable",
        },
        "it": {
            "current_window": "Finestra Corrente",
            "score": "Punteggio",
            "threshold": "Soglia",
            "status": "Stato",
            "anomalous": "Anomala",
            "normal": "Normale",
            "time_range": "Intervallo",
            "duration": "Durata",
            "trend": "Trend",
            "rank": "Posizione Score",
            "frames": "Frame",
            "model_description": "Descrizione del modello",
            "no_description": "Nessuna descrizione restituita per questa finestra.",
            "prev": "prec",
            "next": "succ",
            "above": "sopra soglia",
            "below": "sotto soglia",
            "stable": "stabile",
        },
    }
    lbl = labels.get(lang, labels["en"])

    try:
        idx = max(0, min(int(window_idx), len(windows) - 1))
    except Exception:
        idx = 0
    try:
        score = float(window.get("anomaly_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        threshold_value = 0.5

    start_ts = float(window.get("timestamp_start", window.get("timestamp_center", 0.0)) or 0.0)
    center_ts = float(window.get("timestamp_center", start_ts) or start_ts)
    end_ts = float(window.get("timestamp_end", center_ts) or center_ts)
    duration = max(0.0, end_ts - start_ts)
    frame_names = [str(name) for name in window.get("frame_names", [])]
    frame_count = len(frame_names)
    is_anomalous = bool(window.get("is_anomalous")) or score >= threshold_value
    status_text = lbl["anomalous"] if is_anomalous else lbl["normal"]
    margin_text = lbl["above"] if score >= threshold_value else lbl["below"]
    score_color, score_bg = _score_style(score, threshold_value)

    prev_score = None
    if idx > 0:
        try:
            prev_score = float(windows[idx - 1].get("anomaly_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            prev_score = None
    if prev_score is None:
        trend_text = "-"
        trend_color = "#475467"
    else:
        delta = score - prev_score
        if abs(delta) < 0.03:
            trend_text = lbl["stable"]
            trend_color = "#475467"
        else:
            sign = "+" if delta > 0 else ""
            trend_text = f"{sign}{delta:.2f} vs {lbl['prev']}"
            trend_color = "#b42318" if delta > 0 else "#027a48"

    sorted_scores = sorted(
        (float(w.get("anomaly_score", 0.0) or 0.0) for w in windows),
        reverse=True,
    )
    rank = (sorted_scores.index(score) + 1) if score in sorted_scores else idx + 1
    frame_preview = " - ".join(html.escape(name) for name in frame_names[:1] + frame_names[-1:] if name)
    if frame_count == 1:
        frame_preview = html.escape(frame_names[0])
    if not frame_preview:
        frame_preview = "-"

    prev_badge = ""
    next_badge = ""
    if idx > 0:
        prev = float(windows[idx - 1].get("anomaly_score", 0.0) or 0.0)
        prev_badge = f"<span style='color:#667085;'> {lbl['prev']}: {prev:.2f}</span>"
    if idx + 1 < len(windows):
        nxt = float(windows[idx + 1].get("anomaly_score", 0.0) or 0.0)
        next_badge = f"<span style='color:#667085;'> {lbl['next']}: {nxt:.2f}</span>"

    description_text = html.escape(description.strip() or lbl["no_description"])

    return f"""
    <div style="font-size: 15px; line-height: 1.45;">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; flex-wrap:wrap;">
        <div>
          <div style="font-size:13px; color:#667085;">{lbl['current_window']}</div>
          <div style="font-size:22px; font-weight:700; color:#101828;">#{idx + 1} / {len(windows)}</div>
        </div>
        <div style="padding:10px 14px; border-radius:999px; background:{score_bg}; color:{score_color}; font-weight:700;">
          {lbl['score']}: {score:.2f} ({margin_text})
        </div>
      </div>

      <div style="display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:10px; margin-bottom:12px;">
        <div style="padding:12px; border:1px solid #eaecf0; border-radius:12px; background:#fcfcfd;">
          <div style="color:#667085; font-size:12px;">{lbl['time_range']}</div>
          <div style="font-weight:650; color:#101828;">{_format_seconds(start_ts)} - {_format_seconds(end_ts)}</div>
        </div>
        <div style="padding:12px; border:1px solid #eaecf0; border-radius:12px; background:#fcfcfd;">
          <div style="color:#667085; font-size:12px;">{lbl['duration']}</div>
          <div style="font-weight:650; color:#101828;">{_format_seconds(duration)}</div>
        </div>
        <div style="padding:12px; border:1px solid #eaecf0; border-radius:12px; background:#fcfcfd;">
          <div style="color:#667085; font-size:12px;">{lbl['status']} / {lbl['threshold']}</div>
          <div style="font-weight:650; color:#101828;">{status_text} | {threshold_value:.2f}</div>
        </div>
        <div style="padding:12px; border:1px solid #eaecf0; border-radius:12px; background:#fcfcfd;">
          <div style="color:#667085; font-size:12px;">{lbl['trend']}</div>
          <div style="font-weight:650; color:{trend_color};">{trend_text}{prev_badge}{next_badge}</div>
        </div>
      </div>

      <div style="padding:12px; border:1px solid #eaecf0; border-radius:12px; background:#f9fafb; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;">
          <div><strong>{lbl['rank']}:</strong> #{rank} / {len(windows)}</div>
          <div><strong>{lbl['frames']}:</strong> {frame_count} ({frame_preview})</div>
        </div>
      </div>

      <div style="padding:14px; border-left:4px solid {score_color}; background:#ffffff; border-radius:10px; box-shadow:0 1px 3px rgba(16,24,40,0.08);">
        <div style="font-weight:700; color:#101828; margin-bottom:6px;">{lbl['model_description']}</div>
        <div style="color:#344054;">{description_text}</div>
      </div>
    </div>
    """


# ------------------------------
# Main interface callbacks
# ------------------------------


def _on_run_select(run_name: str, lang: str = "en") -> Tuple:
    """Handle run selection - load data and show first window."""
    # Labels for different languages
    labels = {
        "en": {
            "select_run": "Select a run to see classification",
            "select_details": "Select a run to see details",
            "select_eval": "Please select a run to evaluate.",
            "no_data": "No inference data found",
            "no_frames": "No frames found",
            "no_windows": "No analysis windows",
            "current_window": "Current Window",
            "window": "Window",
            "score": "Score",
            "loaded": "Loaded",
            "has_rating": "has previous rating",
        },
        "it": {
            "select_run": "Seleziona una run per vedere la classificazione",
            "select_details": "Seleziona una run per vedere i dettagli",
            "select_eval": "Seleziona una run da valutare.",
            "no_data": "Nessun dato di inferenza trovato",
            "no_frames": "Nessun frame trovato",
            "no_windows": "Nessuna finestra di analisi",
            "current_window": "Finestra Corrente",
            "window": "Finestra",
            "score": "Punteggio",
            "loaded": "Caricato",
            "has_rating": "ha una valutazione precedente",
        },
    }
    lbl = labels.get(lang, labels["en"])
    
    empty_html = f'<div style="padding: 20px; background: #f0f0f0; border-radius: 10px; text-align: center;"><h2 style="color: #666; font-size: 28px;">{lbl["select_run"]}</h2></div>'
    error_html = lambda msg: f'<div style="padding: 20px; background: #fff3cd; border-radius: 10px; text-align: center; border: 1px solid #ffc107;"><h3 style="color: #856404; font-size: 24px;">{msg}</h3></div>'
    
    if not run_name:
        return (
            None, None,
            empty_html, 
            f"**{lbl['current_window']}:** {lbl['select_details']}", 
            None,
            f"_{lbl['select_eval']}_",
            0, gr.update(maximum=0),
            3, 3, 3, 3, "",
        )

    try:
        data = _load_inference_data(run_name)
        if data is None:
            return (
                None, None,
                error_html(lbl["no_data"]), 
                f"**{lbl['current_window']}:** -", 
                None,
                f"_{lbl['no_data']}: {run_name}_",
                0, gr.update(maximum=0),
                3, 3, 3, 3, "",
            )

        frames_dir = _get_frames_dir(run_name)
        if frames_dir is None:
            return (
                None, None,
                error_html(lbl["no_frames"]), 
                f"**{lbl['current_window']}:** -", 
                None,
                f"_{lbl['no_frames']}: {run_name}_",
                0, gr.update(maximum=0),
                3, 3, 3, 3, "",
            )

        windows = data.get("windows", [])
        if not windows:
            return (
                None, None,
                error_html(lbl["no_windows"]), 
                f"**{lbl['current_window']}:** -", 
                None,
                f"_{lbl['no_windows']}_",
                0, gr.update(maximum=0),
                3, 3, 3, 3, "",
            )

        # Get video path (convert to string for Gradio)
        video_path_obj = _get_video_path(run_name)
        video_path = str(video_path_obj) if video_path_obj else None

        # Get first window
        window = windows[0]
        frame_names = window.get("frame_names", [])
        description = window.get("description", "")
        score = window.get("anomaly_score", 0)
        
        # Translate description if Italian
        if lang == "it" and description:
            description = translate_text(description, "it")

        # Create grid
        grid_img = _create_grid_image(frames_dir, frame_names)

        # Create plot
        segments = data.get("segments", [])
        threshold = data.get("threshold", 0.5)
        plot_img = _create_anomaly_plot(windows, segments, threshold, 0)

        # Overall summary - the main classification result
        summary_text = _build_anomaly_summary(windows, threshold, run_name=run_name, lang=lang)
        
        # Window description for detailed analysis
        window_desc = _build_window_detail_html(
            window,
            windows,
            0,
            threshold,
            description=description,
            lang=lang,
        )

        # Check for existing rating
        existing = _get_existing_rating(run_name)
        if existing:
            return (
                video_path,
                grid_img,
                summary_text,
                window_desc,
                plot_img,
                f"_{lbl['loaded']} `{run_name}` ({lbl['has_rating']}) - {lbl['window']} 1/{len(windows)}_",
                0,
                gr.update(maximum=len(windows) - 1),
                existing.get("score_consistency", 3),
                existing.get("description_quality", 3),
                existing.get("detection_accuracy", 3),
                existing.get("overall_quality", 3),
                existing.get("notes", ""),
            )

        return (
            video_path,
            grid_img,
            summary_text,
            window_desc,
            plot_img,
            f"_{lbl['loaded']} `{run_name}` - {lbl['window']} 1/{len(windows)}_",
            0,
            gr.update(maximum=len(windows) - 1),
            3, 3, 3, 3, "",
        )

    except Exception as e:
        import traceback
        error_msg = f"Error loading run: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        error_lbl = "Errore nel caricamento dati" if lang == "it" else "Error loading data"
        error_html = f'<div style="padding: 20px; background: #f8d7da; border-radius: 10px; text-align: center; border: 1px solid #f5c6cb;"><h3 style="color: #721c24; font-size: 24px;">{error_lbl}</h3></div>'
        return (
            None, None, 
            error_html, 
            f"**{lbl['current_window']}:** -", 
            None,
            f"_Error: {str(e)}_",
            0, gr.update(maximum=0),
            3, 3, 3, 3, "",
        )


def _on_window_change(run_name: str, window_idx: int, lang: str = "en") -> Tuple:
    """Handle window slider change - updates grid and plot, not video."""
    # Labels for different languages
    labels = {
        "en": {
            "no_data": "No data",
            "current_window": "Current Window",
            "no_run": "No run selected.",
            "no_data_loaded": "No data loaded.",
            "no_frames": "No frames directory.",
            "invalid_idx": "Invalid window index.",
            "window": "Window",
            "score": "Score",
            "frames": "Frames",
            "error": "Error",
        },
        "it": {
            "no_data": "Nessun dato",
            "current_window": "Finestra Corrente",
            "no_run": "Nessuna run selezionata.",
            "no_data_loaded": "Nessun dato caricato.",
            "no_frames": "Nessuna directory frame.",
            "invalid_idx": "Indice finestra non valido.",
            "window": "Finestra",
            "score": "Punteggio",
            "frames": "Frame",
            "error": "Errore",
        },
    }
    lbl = labels.get(lang, labels["en"])
    
    empty_html = f'<div style="padding: 20px; background: #f0f0f0; border-radius: 10px; text-align: center;"><h3 style="color: #666; font-size: 24px;">{lbl["no_data"]}</h3></div>'
    
    if not run_name:
        return (
            None, 
            empty_html, 
            f"**{lbl['current_window']}:** -", 
            None, 
            f"_{lbl['no_run']}_"
        )

    try:
        data = _load_inference_data(run_name)
        if data is None:
            return (
                None, 
                empty_html, 
                f"**{lbl['current_window']}:** -", 
                None, 
                f"_{lbl['no_data_loaded']}_"
            )

        frames_dir = _get_frames_dir(run_name)
        if frames_dir is None:
            return (
                None, 
                empty_html, 
                f"**{lbl['current_window']}:** -", 
                None, 
                f"_{lbl['no_frames']}_"
            )

        windows = data.get("windows", [])
        if not windows or window_idx >= len(windows):
            return (
                None, 
                empty_html, 
                f"**{lbl['current_window']}:** -", 
                None, 
                f"_{lbl['invalid_idx']}_"
            )

        window = windows[int(window_idx)]
        frame_names = window.get("frame_names", [])
        description = window.get("description", "")
        score = window.get("anomaly_score", 0)
        
        # Translate description if Italian
        if lang == "it" and description:
            description = translate_text(description, "it")

        # Create grid
        grid_img = _create_grid_image(frames_dir, frame_names)

        # Create plot
        segments = data.get("segments", [])
        threshold = data.get("threshold", 0.5)
        plot_img = _create_anomaly_plot(windows, segments, threshold, int(window_idx))

        # Use cached summary (already computed on run select)
        summary_text = _build_anomaly_summary(windows, threshold, run_name=run_name, use_llm=False, lang=lang)
        
        window_desc = _build_window_detail_html(
            window,
            windows,
            int(window_idx),
            threshold,
            description=description,
            lang=lang,
        )

        frame_range = f"{frame_names[0]} - {frame_names[-1]}" if frame_names else "-"
        status = f"_{lbl['window']} {int(window_idx) + 1}/{len(windows)} | {lbl['frames']}: {frame_range}_"

        return (
            grid_img,
            summary_text,
            window_desc,
            plot_img,
            status,
        )

    except Exception as e:
        print(f"Error in _on_window_change: {e}")
        error_html = f'<div style="padding: 20px; background: #f8d7da; border-radius: 10px; text-align: center;"><h3 style="color: #721c24; font-size: 24px;">{lbl["error"]}</h3></div>'
        return (
            None, 
            error_html, 
            f"**{lbl['current_window']}:** -", 
            None, 
            f"_{lbl['error']}: {str(e)}_"
        )


def _on_plot_click(run_name: str, lang: str, evt: gr.SelectData) -> Tuple:
    """Handle click on the plot to navigate to a window."""
    lbl_no_data = "Nessun dato" if lang == "it" else "No data"
    lbl_current = "Finestra Corrente" if lang == "it" else "Current Window"
    lbl_no_run = "Nessuna run selezionata." if lang == "it" else "No run selected."
    lbl_no_windows = "Nessuna finestra." if lang == "it" else "No windows."
    
    empty_html = f'<div style="padding: 20px; background: #f0f0f0; border-radius: 10px; text-align: center;"><h3 style="color: #666; font-size: 24px;">{lbl_no_data}</h3></div>'
    
    if not run_name:
        return (
            0, None, 
            empty_html, 
            f"**{lbl_current}:** -", 
            None, 
            f"_{lbl_no_run}_"
        )

    data = _load_inference_data(run_name)
    if data is None:
        return (
            0, None, 
            empty_html, 
            f"**{lbl_current}:** -", 
            None, 
            f"_{lbl_no_data}._"
        )

    windows = data.get("windows", [])
    if not windows:
        return (
            0, None, 
            empty_html, 
            f"**{lbl_current}:** -", 
            None, 
            f"_{lbl_no_windows}_"
        )

    # Get clicked point from Plotly event (x-axis is now timestamps)
    # evt.index gives the index of the point in the trace, which corresponds to window index
    # But sometimes evt.index can be None if clicking on background
    if evt.index is not None:
        closest_idx = evt.index
    else:
        # Fallback to x coordinate if available (less precise)
        # For now, just return existing if no index
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    # Return the window index and updated view
    result = _on_window_change(run_name, closest_idx, lang=lang)
    return (closest_idx,) + result


def _submit_evaluation(
    run_name: str,
    score_consistency: float,
    description_quality: float,
    detection_accuracy: float,
    overall_quality: float,
    notes: str,
) -> str:
    """Submit an evaluation for the current run."""
    if not run_name:
        return "Error: No run selected."

    data = _load_inference_data(run_name)
    video_name = data.get("video_name", "unknown") if data else "unknown"

    evaluation = {
        "run_name": run_name,
        "video_name": video_name,
        "timestamp": datetime.now().isoformat(),
        "score_consistency": float(score_consistency),
        "description_quality": float(description_quality),
        "detection_accuracy": float(detection_accuracy),
        "overall_quality": float(overall_quality),
        "notes": notes.strip(),
    }

    _save_evaluation(evaluation)

    return f"Evaluation saved for {run_name}!"


def _export_evaluations() -> Tuple[str, Any]:
    """Export all evaluations as a summary."""
    data = _load_evaluations()
    evaluations = data.get("evaluations", [])

    if not evaluations:
        return "No evaluations found.", None

    lines = ["# VAD Evaluation Summary", ""]
    lines.append(f"Total evaluations: {len(evaluations)}")
    lines.append("")

    if evaluations:
        avg_score = sum(e.get("score_consistency", 0) for e in evaluations) / len(evaluations)
        avg_desc = sum(e.get("description_quality", 0) for e in evaluations) / len(evaluations)
        avg_detect = sum(e.get("detection_accuracy", 0) for e in evaluations) / len(evaluations)
        avg_overall = sum(e.get("overall_quality", 0) for e in evaluations) / len(evaluations)

        lines.append("## Averages")
        lines.append(f"- Score Consistency: {avg_score:.2f}/5")
        lines.append(f"- Description Quality: {avg_desc:.2f}/5")
        lines.append(f"- Detection Accuracy: {avg_detect:.2f}/5")
        lines.append(f"- Overall Quality: {avg_overall:.2f}/5")
        lines.append("")

    lines.append("## Individual Evaluations")
    lines.append("")

    for ev in evaluations:
        lines.append(f"### {ev.get('run_name', 'Unknown')}")
        lines.append(f"- Video: {ev.get('video_name', 'N/A')}")
        lines.append(f"- Date: {ev.get('timestamp', 'N/A')}")
        lines.append(f"- Score Consistency: {ev.get('score_consistency', 'N/A')}/5")
        lines.append(f"- Description Quality: {ev.get('description_quality', 'N/A')}/5")
        lines.append(f"- Detection Accuracy: {ev.get('detection_accuracy', 'N/A')}/5")
        lines.append(f"- Overall Quality: {ev.get('overall_quality', 'N/A')}/5")
        if ev.get("notes"):
            lines.append(f"- Notes: {ev.get('notes')}")
        lines.append("")

    summary = "\n".join(lines)
    return summary, str(EVALUATIONS_FILE)


def _export_csv() -> Tuple[str, Optional[str]]:
    """Export evaluations to CSV file."""
    data = _load_evaluations()
    evaluations = data.get("evaluations", [])

    if not evaluations:
        return "No evaluations to export.", None

    csv_path = REPO_ROOT / "tmp" / "vad_evaluations.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "video_name",
        "timestamp",
        "score_consistency",
        "description_quality",
        "detection_accuracy",
        "overall_quality",
        "notes",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in evaluations:
            row = {field: ev.get(field, "") for field in fieldnames}
            writer.writerow(row)

    return f"Exported {len(evaluations)} evaluations to CSV.", str(csv_path)


# ------------------------------
# Build Gradio interface
# ------------------------------


def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""
    ui_css = """
    #eval-refresh-btn {
        min-width: 56px !important;
        max-width: 56px !important;
    }
    #eval-refresh-btn button {
        min-width: 56px !important;
        max-width: 56px !important;
        padding-left: 6px !important;
        padding-right: 6px !important;
    }
    #eval-video-upload [data-testid="file-upload"] svg {
        display: none !important;
    }
    #eval-video-upload [data-testid="file-upload"] span,
    #eval-video-upload [data-testid="file-upload"] p {
        font-size: 12px !important;
        line-height: 1.2 !important;
    }
    """

    with gr.Blocks(title="VAD Evaluation") as demo:
        # Language state
        lang_state = gr.State(value="en")
        gr.HTML(f"<style>{ui_css}</style>")

        # Header with language selector
        with gr.Row():
            with gr.Column(scale=5):
                title_md = gr.Markdown(t("title", "en") + "\n\n" + t("subtitle", "en"))
            with gr.Column(scale=1):
                lang_selector = gr.Dropdown(
                    label="Language / Lingua",
                    choices=["English", "Italiano"],
                    value="English",
                    interactive=True,
                )

        with gr.Tabs():
            # Tab 1: Evaluate runs (default)
            with gr.TabItem(t("tab_evaluate", "en")) as tab_evaluate:
                available_runs = _get_available_runs()

                with gr.Row():
                    run_selector = gr.Dropdown(
                        label=t("select_run", "en"),
                        choices=available_runs,
                        value=None,
                        allow_custom_value=False,
                        interactive=True,
                        scale=6,
                    )
                    refresh_btn = gr.Button(
                        t("refresh_short", "en"), size="sm", min_width=56, scale=0, elem_id="eval-refresh-btn",
                    )
                    video_upload = gr.File(
                        label=t("upload_video", "en"),
                        file_types=["video"],
                        type="filepath",
                        file_count="single",
                        scale=0,
                        min_width=260,
                        height=110,
                        elem_id="eval-video-upload",
                    )
                upload_status = gr.Markdown(value="")

                with gr.Accordion("Process Uploaded Video", open=False):
                    with gr.Row():
                        upload_llama_url_input = gr.Textbox(
                            label=t("llama_url", "en"),
                            value="http://localhost:8080",
                        )
                        upload_llama_model_input = gr.Textbox(
                            label=t("llama_model", "en"),
                            value=DEFAULT_LLM_MODEL,
                        )

                    upload_prompt_input = gr.Textbox(
                        label="Anomaly prompt",
                        value=ANOMALY_PROMPT,
                        lines=12,
                    )

                    upload_autostart_checkbox = gr.Checkbox(
                        label=t("autostart_label", "en"),
                        value=True,
                        info=t("autostart_info", "en"),
                    )

                    with gr.Accordion("Performance / VRAM settings", open=True):
                        with gr.Row():
                            upload_llama_np_input = gr.Slider(
                                label="llama.cpp parallel slots (np)",
                                minimum=1,
                                maximum=4,
                                value=DEFAULT_LLAMA_PARALLEL,
                                step=1,
                                info="Lower to 1 if you see OOM or too much VRAM usage.",
                            )
                            upload_caption_parallel_input = gr.Slider(
                                label="Caption parallel windows",
                                minimum=1,
                                maximum=4,
                                value=DEFAULT_CAPTION_PARALLEL_WINDOWS,
                                step=1,
                                info="Lower to 1 if the GPU runs out of memory.",
                            )
                        with gr.Row():
                            upload_caption_window_input = gr.Slider(
                                label="Caption window size",
                                minimum=1,
                                maximum=12,
                                value=DEFAULT_CAPTION_WINDOW_SIZE,
                                step=1,
                                info="Frames per analysis request. Higher can be faster but uses more VRAM/context.",
                            )
                            upload_chunk_workers_input = gr.Slider(
                                label="Chunk workers",
                                minimum=1,
                                maximum=4,
                                value=DEFAULT_CHUNK_WORKERS,
                                step=1,
                                info="Parallel video chunks. Keep 1 on 8GB GPUs unless you are testing.",
                            )
                        with gr.Row():
                            upload_ctx_len_input = gr.Slider(
                                label="llama.cpp context length",
                                minimum=4096,
                                maximum=16384,
                                value=DEFAULT_LLAMA_CTX_LEN,
                                step=1024,
                                info="Lower reduces KV cache VRAM; higher allows larger prompts/windows.",
                            )
                            upload_batch_input = gr.Slider(
                                label="llama.cpp batch size",
                                minimum=256,
                                maximum=2048,
                                value=DEFAULT_LLAMA_BATCH,
                                step=256,
                                info="Logical prompt batch. Lower if startup fails or VRAM is high.",
                            )
                            upload_ubatch_input = gr.Slider(
                                label="llama.cpp ubatch size",
                                minimum=64,
                                maximum=512,
                                value=DEFAULT_LLAMA_UBATCH,
                                step=64,
                                info="Physical GPU microbatch. Lower is safer for VRAM.",
                            )
                        with gr.Row():
                            upload_flash_attn_checkbox = gr.Checkbox(
                                label="Flash attention",
                                value=True,
                                info="Usually faster/lower memory. Disable if llama-server exits during startup.",
                            )
                            upload_cont_batching_checkbox = gr.Checkbox(
                                label="Continuous batching",
                                value=True,
                                info="Improves throughput when multiple requests are in flight.",
                            )

                    upload_process_btn = gr.Button("Process uploaded video", variant="secondary")
                    upload_process_logs = gr.Textbox(
                        label=t("processing_logs", "en"),
                        lines=10,
                        max_lines=16,
                        interactive=False,
                    )

                # ========================================
                # VIDEO PLAYER - First thing shown
                # ========================================
                video_player = gr.Video(
                    label=t("video_player", "en"),
                    height=600,
                    autoplay=False,
                )
                
                # ========================================
                # CLASSIFICATION RESULT - Big and colorful
                # ========================================
                summary_md = gr.HTML(
                    value='<div style="padding: 20px; background: #f0f0f0; border-radius: 10px; text-align: center;"><h2 style="color: #666;">Select a run to see classification</h2></div>',
                )
                
                # Status line
                status_text = gr.Markdown(
                    value=f"_{t('select_run_begin', 'en')}_",
                )

                # ========================================
                # DETAILED ANALYSIS - TWO COLUMNS LAYOUT
                # ========================================
                with gr.Accordion("Detailed Analysis", open=True):
                    evaluate_header_md = gr.Markdown("")
                    viz_header_md = gr.Markdown("#### Frame Analysis")
                    
                    # Top row: Grid image (left) and Description (right)
                    with gr.Row():
                        # Left column: Grid image
                        with gr.Column(scale=1):
                            grid_image = gr.Image(
                                label=t("window_frames", "en"),
                                height=250,
                            )
                        
                        # Right column: Descriptions
                        with gr.Column(scale=1):
                            # Current window description
                            window_desc_md = gr.Markdown(
                                value="**Current Window:** Select a run to see window details",
                            )
                    
                    # Timeline section - Full width
                    timeline_header_md = gr.Markdown("#### Anomaly Score Timeline")
                    timeline_hint_md = gr.Markdown(f"_{t('timeline_hint', 'en')}_")
                    
                    # Plot - Full width
                    plot_image = gr.Plot(
                        label=t("anomaly_plot", "en"),
                        show_label=False,
                    )
                    
                    # Window navigation - Full width
                    window_slider = gr.Slider(
                        label=t("window_nav", "en"),
                        minimum=0,
                        maximum=0,
                        step=1,
                        value=0,
                        info=t("window_nav_info", "en"),
                    )

                gr.Markdown("---")
                
                # ========================================
                # RATING SECTION: User Evaluation
                # ========================================
                rate_header_md = gr.Markdown("### Your Evaluation")
                rate_scale_md = gr.Markdown(f"_{t('rate_scale', 'en')}_")

                with gr.Row():
                    score_consistency = gr.Slider(
                        label=t("score_consistency", "en"),
                        info=t("score_consistency_info", "en"),
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                    )
                    description_quality = gr.Slider(
                        label=t("description_quality", "en"),
                        info=t("description_quality_info", "en"),
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                    )
                    detection_accuracy = gr.Slider(
                        label=t("detection_accuracy", "en"),
                        info=t("detection_accuracy_info", "en"),
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                    )
                    overall_quality = gr.Slider(
                        label=t("overall_quality", "en"),
                        info=t("overall_quality_info", "en"),
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                    )

                notes = gr.Textbox(
                    label=t("notes", "en"),
                    placeholder=t("notes_placeholder", "en"),
                    lines=2,
                )

                with gr.Row():
                    submit_btn = gr.Button(t("submit", "en"), variant="primary")
                    result_text = gr.Textbox(label=t("result", "en"), interactive=False)

                gr.Markdown("---")

                with gr.Accordion(t("view_all", "en"), open=False):
                    export_btn = gr.Button(t("load_summary", "en"))
                    summary_text = gr.Markdown()
                    file_path = gr.Textbox(label=t("eval_file", "en"), interactive=False)

                    gr.Markdown("---")
                    with gr.Row():
                        csv_btn = gr.Button(t("export_csv", "en"))
                        csv_status = gr.Textbox(label=t("csv_status", "en"), interactive=False)
                    csv_path = gr.Textbox(label=t("csv_path", "en"), interactive=False)

                # Event handlers
                def refresh_runs():
                    new_runs = _get_available_runs()
                    return gr.update(choices=new_runs)

                refresh_btn.click(
                    fn=refresh_runs,
                    inputs=[],
                    outputs=[run_selector],
                )

                def on_video_upload(file_path, lang):
                    """Handle video upload - show it in the video player."""
                    if file_path is None:
                        return None, ""
                    
                    file_name = Path(file_path).name
                    return file_path, f"_{file_name}_"

                video_upload.change(
                    fn=on_video_upload,
                    inputs=[video_upload, lang_state],
                    outputs=[video_player, upload_status],
                )

                def process_uploaded_video(
                    file_path,
                    url,
                    model,
                    prompt,
                    autostart,
                    llama_np,
                    caption_parallel,
                    caption_window,
                    chunk_workers,
                    ctx_len,
                    batch_size,
                    ubatch_size,
                    flash_attn,
                    cont_batching,
                ):
                    if not file_path:
                        yield "Error: Please upload a video first.", gr.update(), "_No uploaded video selected._"
                        return

                    uploaded_path = Path(file_path)
                    target_run_dir = _next_run_dir(uploaded_path.stem)
                    processing_status = f"_Processing uploaded video `{uploaded_path.name}`..._"

                    latest_log = ""
                    yield latest_log, gr.update(), processing_status
                    for log in _run_video_processing(
                        str(uploaded_path),
                        url,
                        model,
                        prompt,
                        autostart,
                        run_dir=target_run_dir,
                        llama_parallel_slots=int(llama_np or DEFAULT_LLAMA_PARALLEL),
                        caption_parallel_windows=int(caption_parallel or DEFAULT_CAPTION_PARALLEL_WINDOWS),
                        caption_window_size=int(caption_window or DEFAULT_CAPTION_WINDOW_SIZE),
                        chunk_workers_override=int(chunk_workers or DEFAULT_CHUNK_WORKERS),
                        llama_ctx_len=int(ctx_len or DEFAULT_LLAMA_CTX_LEN),
                        llama_batch_size=int(batch_size or DEFAULT_LLAMA_BATCH),
                        llama_ubatch_size=int(ubatch_size or DEFAULT_LLAMA_UBATCH),
                        llama_flash_attn=bool(flash_attn),
                        llama_cont_batching=bool(cont_batching),
                    ):
                        latest_log = log
                        yield latest_log, gr.update(), processing_status

                    new_runs = _get_available_runs()
                    run_ready = target_run_dir.name in new_runs and _get_inference_data_path(target_run_dir.name) is not None
                    final_status = (
                        f"_Uploaded video processed into run `{target_run_dir.name}`._"
                        if run_ready
                        else f"_Processing finished, but no evaluable run was found for `{uploaded_path.name}`._"
                    )
                    selector_update = gr.update(choices=new_runs, value=target_run_dir.name if run_ready else None)
                    yield latest_log, selector_update, final_status

                upload_process_btn.click(
                    fn=process_uploaded_video,
                    inputs=[
                        video_upload,
                        upload_llama_url_input,
                        upload_llama_model_input,
                        upload_prompt_input,
                        upload_autostart_checkbox,
                        upload_llama_np_input,
                        upload_caption_parallel_input,
                        upload_caption_window_input,
                        upload_chunk_workers_input,
                        upload_ctx_len_input,
                        upload_batch_input,
                        upload_ubatch_input,
                        upload_flash_attn_checkbox,
                        upload_cont_batching_checkbox,
                    ],
                    outputs=[upload_process_logs, run_selector, upload_status],
                )

                run_selector.change(
                    fn=_on_run_select,
                    inputs=[run_selector, lang_state],
                    outputs=[
                        video_player,
                        grid_image,
                        summary_md,
                        window_desc_md,
                        plot_image,
                        status_text,
                        window_slider,
                        window_slider,  # for maximum update
                        score_consistency,
                        description_quality,
                        detection_accuracy,
                        overall_quality,
                        notes,
                    ],
                )

                window_slider.change(
                    fn=_on_window_change,
                    inputs=[run_selector, window_slider, lang_state],
                    outputs=[
                        grid_image,
                        summary_md,
                        window_desc_md,
                        plot_image,
                        status_text,
                    ],
                )

                # plot_image.select(
                #     fn=_on_plot_click,
                #     inputs=[run_selector, lang_state],
                #     outputs=[
                #         window_slider,
                #         grid_image,
                #         summary_md,
                #         window_desc_md,
                #         plot_image,
                #         status_text,
                #     ],
                # )

                submit_btn.click(
                    fn=_submit_evaluation,
                    inputs=[
                        run_selector,
                        score_consistency,
                        description_quality,
                        detection_accuracy,
                        overall_quality,
                        notes,
                    ],
                    outputs=[result_text],
                )

                export_btn.click(
                    fn=_export_evaluations,
                    inputs=[],
                    outputs=[summary_text, file_path],
                )

                csv_btn.click(
                    fn=_export_csv,
                    inputs=[],
                    outputs=[csv_status, csv_path],
                )

            # Tab 2: Process new videos
            with gr.TabItem(t("tab_process", "en")) as tab_process:
                process_header_md = gr.Markdown(t("process_header", "en"))

                unprocessed = _get_unprocessed_videos()
                unprocessed_choices = [name for name, _ in unprocessed]
                unprocessed_paths = {name: str(path) for name, path in unprocessed}

                with gr.Row():
                    video_selector = gr.Dropdown(
                        label=f"{t('select_video', 'en')} ({len(unprocessed)} {t('unprocessed', 'en')})",
                        choices=unprocessed_choices,
                        value=None,
                        allow_custom_value=False,
                        interactive=True,
                        scale=10,
                    )
                    refresh_videos_btn = gr.Button(t("refresh", "en"), size="sm", min_width=50, scale=1)

                with gr.Row():
                    llama_url_input = gr.Textbox(
                        label=t("llama_url", "en"),
                        value="http://localhost:8080",
                    )
                    llama_model_input = gr.Textbox(
                        label=t("llama_model", "en"),
                        value=DEFAULT_LLM_MODEL,
                    )

                prompt_input = gr.Textbox(
                    label="Anomaly prompt",
                    value=ANOMALY_PROMPT,
                    lines=16,
                )

                autostart_checkbox = gr.Checkbox(
                    label=t("autostart_label", "en"),
                    value=True,
                    info=t("autostart_info", "en"),
                )

                with gr.Accordion("Performance / VRAM settings", open=True):
                    with gr.Row():
                        llama_np_input = gr.Slider(
                            label="llama.cpp parallel slots (np)",
                            minimum=1,
                            maximum=4,
                            value=DEFAULT_LLAMA_PARALLEL,
                            step=1,
                            info="Set to 1 if you see OOM or too much VRAM usage.",
                        )
                        caption_parallel_input = gr.Slider(
                            label="Caption parallel windows",
                            minimum=1,
                            maximum=4,
                            value=DEFAULT_CAPTION_PARALLEL_WINDOWS,
                            step=1,
                            info="Set to 1 for low-VRAM mode; higher values keep GPU busier.",
                        )
                    with gr.Row():
                        caption_window_input = gr.Slider(
                            label="Caption window size",
                            minimum=1,
                            maximum=12,
                            value=DEFAULT_CAPTION_WINDOW_SIZE,
                            step=1,
                            info="Frames per analysis request. Higher can be faster but uses more VRAM/context.",
                        )
                        chunk_workers_input = gr.Slider(
                            label="Chunk workers",
                            minimum=1,
                            maximum=4,
                            value=DEFAULT_CHUNK_WORKERS,
                            step=1,
                            info="Parallel video chunks. Keep 1 on 8GB GPUs unless you are testing.",
                        )
                    with gr.Row():
                        ctx_len_input = gr.Slider(
                            label="llama.cpp context length",
                            minimum=4096,
                            maximum=16384,
                            value=DEFAULT_LLAMA_CTX_LEN,
                            step=1024,
                            info="Lower reduces KV cache VRAM; higher allows larger prompts/windows.",
                        )
                        batch_input = gr.Slider(
                            label="llama.cpp batch size",
                            minimum=256,
                            maximum=2048,
                            value=DEFAULT_LLAMA_BATCH,
                            step=256,
                            info="Logical prompt batch. Lower if startup fails or VRAM is high.",
                        )
                        ubatch_input = gr.Slider(
                            label="llama.cpp ubatch size",
                            minimum=64,
                            maximum=512,
                            value=DEFAULT_LLAMA_UBATCH,
                            step=64,
                            info="Physical GPU microbatch. Lower is safer for VRAM.",
                        )
                    with gr.Row():
                        flash_attn_checkbox = gr.Checkbox(
                            label="Flash attention",
                            value=True,
                            info="Usually faster/lower memory. Disable if llama-server exits during startup.",
                        )
                        cont_batching_checkbox = gr.Checkbox(
                            label="Continuous batching",
                            value=True,
                            info="Improves throughput when multiple requests are in flight.",
                        )

                process_btn = gr.Button(t("start_processing", "en"), variant="primary")

                process_logs = gr.Textbox(
                    label=t("processing_logs", "en"),
                    lines=18,
                    max_lines=25,
                    interactive=False,
                )

                # State to store video paths
                video_paths_state = gr.State(value=unprocessed_paths)

                def refresh_unprocessed():
                    new_unprocessed = _get_unprocessed_videos()
                    new_choices = [name for name, _ in new_unprocessed]
                    new_paths = {name: str(path) for name, path in new_unprocessed}
                    return (
                        gr.update(
                            choices=new_choices,
                            label=f"Select video to process ({len(new_unprocessed)} unprocessed)",
                        ),
                        new_paths,
                    )

                refresh_videos_btn.click(
                    fn=refresh_unprocessed,
                    inputs=[],
                    outputs=[video_selector, video_paths_state],
                )

                def start_processing(
                    video_name,
                    paths,
                    url,
                    model,
                    prompt,
                    autostart,
                    llama_np,
                    caption_parallel,
                    caption_window,
                    chunk_workers,
                    ctx_len,
                    batch_size,
                    ubatch_size,
                    flash_attn,
                    cont_batching,
                ):
                    if not video_name or video_name not in paths:
                        yield "Error: Please select a video first."
                        return
                    video_path = paths[video_name]
                    for log in _run_video_processing(
                        video_path,
                        url,
                        model,
                        prompt,
                        autostart,
                        llama_parallel_slots=int(llama_np or DEFAULT_LLAMA_PARALLEL),
                        caption_parallel_windows=int(caption_parallel or DEFAULT_CAPTION_PARALLEL_WINDOWS),
                        caption_window_size=int(caption_window or DEFAULT_CAPTION_WINDOW_SIZE),
                        chunk_workers_override=int(chunk_workers or DEFAULT_CHUNK_WORKERS),
                        llama_ctx_len=int(ctx_len or DEFAULT_LLAMA_CTX_LEN),
                        llama_batch_size=int(batch_size or DEFAULT_LLAMA_BATCH),
                        llama_ubatch_size=int(ubatch_size or DEFAULT_LLAMA_UBATCH),
                        llama_flash_attn=bool(flash_attn),
                        llama_cont_batching=bool(cont_batching),
                    ):
                        yield log

                process_btn.click(
                    fn=start_processing,
                    inputs=[
                        video_selector,
                        video_paths_state,
                        llama_url_input,
                        llama_model_input,
                        prompt_input,
                        autostart_checkbox,
                        llama_np_input,
                        caption_parallel_input,
                        caption_window_input,
                        chunk_workers_input,
                        ctx_len_input,
                        batch_input,
                        ubatch_input,
                        flash_attn_checkbox,
                        cont_batching_checkbox,
                    ],
                    outputs=[process_logs],
                )

        # Language change handler
        def on_language_change(lang_choice):
            lang = "it" if lang_choice == "Italiano" else "en"
            unprocessed = _get_unprocessed_videos()
            num_unprocessed = len(unprocessed)
            return (
                lang,  # lang_state
                t("title", lang) + "\n\n" + t("subtitle", lang),  # title_md
                t("process_header", lang),  # process_header_md
                gr.update(label=f"{t('select_video', lang)} ({num_unprocessed} {t('unprocessed', lang)})"),  # video_selector
                gr.update(label=t("llama_url", lang)),  # llama_url_input
                gr.update(label=t("llama_model", lang)),  # llama_model_input
                gr.update(label=t("llama_url", lang)),  # upload_llama_url_input
                gr.update(label=t("llama_model", lang)),  # upload_llama_model_input
                gr.update(label=t("autostart_label", lang), info=t("autostart_info", lang)),  # upload_autostart_checkbox
                gr.update(value=t("start_processing", lang)),  # upload_process_btn
                gr.update(label=t("processing_logs", lang)),  # upload_process_logs
                gr.update(label=t("autostart_label", lang), info=t("autostart_info", lang)),  # autostart_checkbox
                gr.update(value=t("start_processing", lang)),  # process_btn
                gr.update(label=t("processing_logs", lang)),  # process_logs
                "",  # evaluate_header_md (hidden)
                gr.update(label=t("select_run", lang)),  # run_selector
                gr.update(label=t("upload_video", lang)),  # video_upload
                "",  # upload_status
                f"_{t('select_run_begin', lang)}_",  # status_text (Markdown)
                "#### Frame Analysis",  # viz_header_md
                gr.update(label=t("video_player", lang)),  # video_player
                gr.update(label=t("window_frames", lang)),  # grid_image
                '<div style="padding: 20px; background: #f0f0f0; border-radius: 10px; text-align: center;"><h2 style="color: #666;">Select a run to see classification</h2></div>',  # summary_md (HTML)
                f"**Current Window:** {t('select_run_details', lang)}",  # window_desc_md
                "#### Anomaly Score Timeline",  # timeline_header_md
                f"_{t('timeline_hint', lang)}_",  # timeline_hint_md
                gr.update(label=t("anomaly_plot", lang)),  # plot_image
                gr.update(label=t("window_nav", lang), info=t("window_nav_info", lang)),  # window_slider
                "### Your Evaluation",  # rate_header_md
                f"_{t('rate_scale', lang)}_",  # rate_scale_md
                gr.update(label=t("score_consistency", lang), info=t("score_consistency_info", lang)),
                gr.update(label=t("description_quality", lang), info=t("description_quality_info", lang)),
                gr.update(label=t("detection_accuracy", lang), info=t("detection_accuracy_info", lang)),
                gr.update(label=t("overall_quality", lang), info=t("overall_quality_info", lang)),
                gr.update(label=t("notes", lang), placeholder=t("notes_placeholder", lang)),
                gr.update(value=t("submit", lang)),  # submit_btn
                gr.update(label=t("result", lang)),  # result_text
                gr.update(value=t("load_summary", lang)),  # export_btn
                gr.update(label=t("eval_file", lang)),  # file_path
                gr.update(value=t("export_csv", lang)),  # csv_btn
                gr.update(label=t("csv_status", lang)),  # csv_status
                gr.update(label=t("csv_path", lang)),  # csv_path
            )

        lang_selector.change(
            fn=on_language_change,
            inputs=[lang_selector],
            outputs=[
                lang_state,
                title_md,
                process_header_md,
                video_selector,
                llama_url_input,
                llama_model_input,
                upload_llama_url_input,
                upload_llama_model_input,
                upload_autostart_checkbox,
                upload_process_btn,
                upload_process_logs,
                autostart_checkbox,
                process_btn,
                process_logs,
                evaluate_header_md,
                run_selector,
                video_upload,
                upload_status,
                status_text,
                viz_header_md,
                video_player,
                grid_image,
                summary_md,
                window_desc_md,
                timeline_header_md,
                timeline_hint_md,
                plot_image,
                window_slider,
                rate_header_md,
                rate_scale_md,
                score_consistency,
                description_quality,
                detection_accuracy,
                overall_quality,
                notes,
                submit_btn,
                result_text,
                export_btn,
                file_path,
                csv_btn,
                csv_status,
                csv_path,
            ],
        )

    return demo


def main(argv: Optional[Tuple[str, ...]] = None) -> None:
    """Main entry point."""
    global DATA_DIR, RUNS_DIR, EVALUATIONS_FILE

    parser = argparse.ArgumentParser(description="Gradio interface for VAD evaluation")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the Gradio server")
    parser.add_argument("--port", type=int, default=7861, help="Port for the Gradio server")
    parser.add_argument("--share", action="store_true", help="Share the interface publicly")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory scanned for videos to process (defaults to MUVAD_DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Directory containing Gradio runs (defaults to MUVAD_GRADIO_RUNS_DIR or ./tmp/gradio_runs)",
    )
    parser.add_argument(
        "--evaluations-file",
        type=str,
        default=None,
        help="JSON file used to store ratings (defaults to MUVAD_EVALUATIONS_FILE or ./tmp/vad_evaluations.json)",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Optional basic auth 'user:password'",
    )
    args = parser.parse_args(argv)

    if args.data_dir:
        DATA_DIR = _resolve_runtime_path(args.data_dir)
    if args.runs_dir:
        RUNS_DIR = _resolve_runtime_path(args.runs_dir)
    if args.evaluations_file:
        EVALUATIONS_FILE = _resolve_runtime_path(args.evaluations_file)

    demo = build_interface()

    auth = None
    if args.auth:
        if ":" not in args.auth:
            raise SystemExit("--auth must follow the format user:password")
        user, password = args.auth.split(":", 1)
        auth = (user, password)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
