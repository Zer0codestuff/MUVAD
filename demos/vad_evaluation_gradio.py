#!/usr/bin/env python3
"""Gradio app for evaluating VAD showcase videos and rating their performance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
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
DEFAULT_LLM_MODEL = "lmstudio-community/InternVL3_5-2B-GGUF:Q8_0"

# Cache for summaries (run_name -> summary)
_summary_cache: Dict[str, str] = {}

# Global variable to track if server was started by us
_llama_server_process: Optional[subprocess.Popen] = None
_server_started: bool = False


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
    ctx_len: int = 8192,
    server_bin: str = "llama-server",
) -> bool:
    """Start llama-server if not already running. Uses -hf flag to download from HuggingFace."""
    global _llama_server_process, _server_started
    
    if _is_server_running(f"http://localhost:{port}"):
        print(f"[LLM Server] Server already running on port {port}")
        return True
    
    print(f"[LLM Server] Starting llama-server with model: {model_name}")
    
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
            "--host", "0.0.0.0",
        ]
        
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


def _next_run_dir(video_stem: str) -> Path:
    """Allocate a run directory name without collisions."""
    run_dir = RUNS_DIR / video_stem
    if run_dir.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{video_stem}_{timestamp}"
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
) -> Dict[str, Any]:
    from scripts.prediction.workflow import read_config

    try:
        config = read_config("config.yml")
    except Exception:
        config = {}

    output_dir = run_dir / "output"
    select_fps = 2.0
    selector_stride = max(1, int(round(float(video_fps or 30.0) / select_fps)))

    captioner_params = dict(config.get("captioner", {}).get("parameters", {}) or {})
    captioner_params.update({
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 0.0,
        "cont_batching": True,
        "ngl": captioner_params.get("ngl", 999),
        "ctx_len": captioner_params.get("ctx_len", 8192),
        "np": captioner_params.get("np", 1),
        "autostart": bool(autostart_server),
        "ready_timeout": 300.0 if autostart_server else 20.0,
    })

    return {
        "evaluate": config.get("evaluate", {}),
        "extractor": {
            **config.get("extractor", {}),
            "video_url": str(video_path),
            "timeout": 0.001,
            "resize": [448, 448],
            "save_dir": "",
            "log": "INFO",
        },
        "selector": {
            **config.get("selector", {}),
            "batch_size": selector_stride,
            "save_dir": str(run_dir / "frames_selected"),
            "log": "INFO",
        },
        "captioner": {
            **config.get("captioner", {}),
            "model_name": llama_model,
            "prompt": prompt,
            "batch_size": 6,
            "warmup_timeout": 20,
            "random_seed": 1337,
            "aggregate": True,
            "aggregate_frames_tag": "FramesCount",
            "aggregate_timestamp_joiner": ", ",
            "parameters": captioner_params,
            "backend": "llamacpp",
            "host": _normalize_llama_host(llama_url),
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


def _run_video_processing(
    video_path: str,
    llama_url: str,
    llama_model: str,
    prompt: str,
    autostart_server: bool = True,
    run_dir: Optional[Path] = None,
) -> Iterator[str]:
    """Run the VAD pipeline on a video and yield log output."""
    if not video_path:
        yield "Error: No video selected."
        return

    video = Path(video_path)
    if not video.exists():
        yield f"Error: Video not found: {video_path}"
        return

    video_stem = video.stem
    run_dir = run_dir or _next_run_dir(video_stem)
    run_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = run_dir / "videos"
    output_dir = run_dir / "output"
    frames_selected_dir = run_dir / "frames_selected"
    for directory in (videos_dir, output_dir, frames_selected_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dest_video = videos_dir / video.name
    if dest_video.resolve() != video.resolve():
        shutil.copy2(video, dest_video)

    video_fps, video_total_s, video_frame_count = _read_video_metadata(dest_video)
    config = _build_async_pipeline_config(
        video_path=dest_video,
        run_dir=run_dir,
        llama_url=llama_url,
        llama_model=llama_model,
        prompt=prompt,
        autostart_server=autostart_server,
        video_fps=video_fps,
    )

    yield f"Starting processing: {video.name}\n"
    yield f"Run directory: {run_dir}\n"
    yield "Pipeline: async main workflow (Extractor -> Selector -> Captioner -> Detector -> Notifier)\n"
    yield f"Video FPS: {video_fps:.2f} | Frames: {video_frame_count} | Duration: {video_total_s:.2f}s\n"
    if autostart_server:
        yield "Autostart server: enabled\n\n"
    else:
        yield "Autostart server: disabled (ensure server is running)\n\n"

    status_queue: queue.Queue[Tuple[str, Any]] = queue.Queue()

    def run_worker() -> None:
        try:
            from scripts.prediction.workflow import initialize_modules, workflow

            status_queue.put(("log", "Initializing modules...\n"))
            modules = initialize_modules(config)
            status_queue.put(("log", "Modules ready. Streaming frames through the async pipeline...\n"))
            result = bool(workflow(*modules, config))
            json_path = _save_async_inference_data(
                run_dir=run_dir,
                video_name=video.stem,
                prompt=prompt,
                threshold=float(config["notifier"]["threshold"]),
                window_step=int(config["captioner"]["batch_size"]),
                video_total_s=video_total_s,
            )
            status_queue.put(("done", (result, json_path)))
        except Exception as exc:
            status_queue.put(("error", exc))

    worker = threading.Thread(target=run_worker, daemon=True)
    worker.start()

    accumulated = ""
    last_progress = 0.0
    while worker.is_alive() or not status_queue.empty():
        try:
            kind, payload = status_queue.get(timeout=0.5)
        except queue.Empty:
            now = time.time()
            if now - last_progress >= 2.0:
                selected_count = len(list(frames_selected_dir.glob("frame_*.*")))
                captioner_path = output_dir / "captioner.txt"
                window_count = 0
                if captioner_path.exists():
                    try:
                        window_count = captioner_path.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        ).count("- frames: ")
                    except Exception:
                        window_count = 0
                accumulated += (
                    f"Progress: selected frames={selected_count}, "
                    f"completed windows={window_count}\n"
                )
                last_progress = now
                yield accumulated
            continue

        if kind == "log":
            accumulated += str(payload)
            yield accumulated
        elif kind == "done":
            result, json_path = payload
            data = _load_inference_data(run_dir.name)
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

    worker.join(timeout=1.0)
    if worker.is_alive():
        yield accumulated + "\nProcessing is still shutting down.\n"


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
        window_header = f"**{lbl['window']} 1 | {lbl['score']}: {score:.2f}**"
        window_desc = f"<div style='font-size: 1.2em; line-height: 1.6;'>{window_header}\n\n{description}</div>"

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
        
        window_header = f"**{lbl['window']} {int(window_idx) + 1} | {lbl['score']}: {score:.2f}**"
        window_desc = f"<div style='font-size: 1.2em; line-height: 1.6;'>{window_header}\n\n{description}</div>"
        
        status = f"_{lbl['window']} {int(window_idx) + 1}/{len(windows)} | {lbl['frames']}: {frame_names[0]} - {frame_names[-1]}_"

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

                def process_uploaded_video(file_path, url, model, prompt, autostart):
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

                def start_processing(video_name, paths, url, model, prompt, autostart):
                    if not video_name or video_name not in paths:
                        yield "Error: Please select a video first."
                        return
                    video_path = paths[video_name]
                    for log in _run_video_processing(video_path, url, model, prompt, autostart):
                        yield log

                process_btn.click(
                    fn=start_processing,
                    inputs=[video_selector, video_paths_state, llama_url_input, llama_model_input, prompt_input, autostart_checkbox],
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
