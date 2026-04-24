#!/usr/bin/env python3
"""
VAD Showcase (script)

Given an input video, extracts frames, selects subsampled ones (e.g. ~2 fps),
queries a vision-chat model via llama-server to get anomaly scores
on windows of 6 frames, identifies anomalous segments and generates an animated video
that scrolls through the frames showing:
 - preview of current frame
 - grid (6) of the analyzed window with its score
 - description/justification
 - timeline of the window in the video
 - anomaly score curve and threshold, with highlighted segments

Main requirements: python3, opencv-python, pillow, numpy, pandas, matplotlib, requests.
For MP4 saving via ffmpeg, have ffmpeg available in PATH; otherwise OpenCV fallback.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
import subprocess
import signal
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import shlex

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")  # for saving without display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter, writers

try:
    import cv2
except Exception as _e:  # pragma: no cover
    cv2 = None


# ------------------------------
# Prompt and llama-server API
# ------------------------------

ANOMALY_PROMPT = (
    "You are a highly trained security analyst, reviewing footage from a surveillance camera.\n"
    "You will receive a sequence of frame from a video.\n"
    "Your task is to determine whether there is a clear and serious anomalous activity in the video that would require an immediate security response.\n\n"
    "Strict Definition of Anomalous Activity:\n\n"
    "An activity is considered anomalous only if it is an immediate security threat or highly unusual in a dangerous way.\n"
    "This includes but is not limited to physical violence, explosions or fires, severe accidents or injuries, threatening or dangerous behavior.\n\n"
    "Do NOT consider as anomalies normal daily activities (e.g., sitting, talking, walking, opening doors) unless there is clear evidence of danger.\n\n"
    "Your response format:\n\n"
    "- Short Summary: Describe what happens in the frames concisely.\n\n"
    "- Anomaly Assessment; Clearly state whether the activity is truly an anomaly based on the strict definition above.\n\n"
    "- Anomaly Score: Assign a value from 0 (completely normal) to 1 (clearly dangerous); only assign above 0.5 if the event is truly alarming.\n\n"
    "- JSON object:\n\n"
    "{\n\n\"anomaly_score\": <value between 0 and 1>,\n\n\"description\": \"<your justification>\"\n\n}\n\n"
    "Here are the frames you have to analyze:\n\n<frames>"
)


def resolve_anomaly_prompt(prompt: Optional[str] = None, prompt_file: Optional[str] = None) -> str:
    """Resolve the active anomaly prompt from inline text or a file."""
    if prompt is not None and str(prompt).strip():
        return str(prompt)
    if prompt_file:
        return Path(prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    return ANOMALY_PROMPT


def _image_to_data_url(img: Image.Image | str) -> str:
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@lru_cache(maxsize=64)
def _load_rgb_image(path_str: str) -> Image.Image:
    """Load an image once and reuse it across inference/rendering steps."""
    with Image.open(path_str) as img:
        return img.convert("RGB").copy()


def _parse_json_any(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _wait_llama_health(base_url: str, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    print(f"    Waiting for llama-server health check ({base_url})…", flush=True)
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200:
                # Extra check: some versions return 200 while loading
                # Wait a bit more to be sure tensors are loaded
                time.sleep(5.0)
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass
        time.sleep(2.0)
    return False


def _parse_port_from_url(base_url: str, default: int = 1234) -> int:
    try:
        parsed = urlparse(base_url)
        if parsed.port:
            return int(parsed.port)
        # if port is missing, return default
        return int(default)
    except Exception:
        return int(default)


def start_llama_server(
    *,
    base_url: str,
    model: str,
    server_bin: str = "llama-server",
    ctx_size: int = 8192,
    ngl: int = 999,
    extra_args: str = "",
    log_path: Path,
) -> subprocess.Popen:
    """Starts llama-server in background and redirects logs to file."""
    port = _parse_port_from_url(base_url, default=1234)
    cmd = [
        server_bin,
        "-hf", model,
        "--ctx-size", str(ctx_size),
        "-ngl", str(ngl),
        "--port", str(port),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        close_fds=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return proc


def stop_llama_server(proc: Optional[subprocess.Popen]) -> None:
    if not proc:
        return
    try:
        if hasattr(os, "getpgid") and hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
    except Exception:
        pass


def vision_chat(
    images: Sequence[str | Image.Image],
    prompt: str,
    *,
    base_url: str = "http://localhost:1234",
    model: str = "lmstudio-community/InternVL3_5-2B-GGUF:Q8_0",
    max_tokens: int = 512,
    temperature: float = 0.5,
    top_p: float = 0.9,
    request_timeout: int = 300,  # Increased
) -> str:
    """Performs an OpenAI-style vision-chat call to llama-server."""
    contents: List[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        contents.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": contents}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    # Retry logic for 503 errors (server busy/loading)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=request_timeout)
            if r.status_code == 503 and attempt < max_retries - 1:
                print(f"    [LLM] Server busy (503), retrying in {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"    [LLM] HTTP Error: {e}, retrying...")
            time.sleep(5)
    
    return ""


# ------------------------------
# Video IO and frame selection
# ------------------------------


def extract_frames(video_path: str | Path, output_dir: Path, resize: Optional[Tuple[int, int]] = None) -> Tuple[float, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available: install opencv-python")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    timestamps: List[float] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame_rgb = cv2.resize(frame_rgb, resize)
        image = Image.fromarray(frame_rgb)
        frame_name = output_dir / f"frame_{frame_index:06d}.png"
        image.save(frame_name)
        timestamps.append(frame_index / fps)
        frame_index += 1
    cap.release()
    return float(fps), np.array(timestamps, dtype=np.float32)


def select_frames(
    frames_dir: Path,
    timestamps: np.ndarray,
    output_dir: Path,
    *,
    stride: int,
    diff_threshold: float = 1e18,
) -> List[Tuple[Path, float]]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available: install opencv-python")
    frame_paths = sorted(frames_dir.glob("*.png"))
    selected_entries: List[Tuple[Path, float]] = []
    previous_gray = None
    for idx, (frame_path, ts) in enumerate(zip(frame_paths, timestamps)):
        frame_bgr = cv2.imread(str(frame_path))
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if idx % stride == 0:
            selected_entries.append((frame_path, float(ts)))
        elif previous_gray is not None:
            diff = cv2.absdiff(frame_gray, previous_gray)
            mean_diff = float(diff.mean())
            if mean_diff > diff_threshold:
                selected_entries.append((frame_path, float(ts)))
        previous_gray = frame_gray
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_path, _ in selected_entries:
        destination = output_dir / frame_path.name
        shutil.copy(frame_path, destination)
    return selected_entries


# ------------------------------
# Windowed inference and segmentation
# ------------------------------


# ------------------------------
# Video selection
# ------------------------------

def list_video_files(video_dir: Path) -> List[Path]:
    exts = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv"]
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(video_dir.glob(ext)))
        files.extend(sorted(video_dir.glob(ext.upper())))
    return files


def get_first_video(video_dir: Path) -> Optional[Path]:
    """Returns the first video file found in the directory, or None if no videos exist."""
    vids = list_video_files(video_dir)
    if vids:
        print(f"Found {len(vids)} video(s) in {video_dir}")
        print(f"Using first video: {vids[0].name}")
        return vids[0].resolve()
    return None


@dataclass
class WorkDirs:
    base: Path
    videos: Path
    frames_full: Path
    frames_selected: Path
    output: Path


def setup_workdirs(base_dir: str) -> WorkDirs:
    base = Path(base_dir).expanduser().resolve()
    videos = base / "videos"
    frames_full = base / "frames_full"
    frames_selected = base / "frames_selected"
    output = base / "output"
    for directory in (base, videos, frames_full, frames_selected, output):
        directory.mkdir(parents=True, exist_ok=True)
    return WorkDirs(base=base, videos=videos, frames_full=frames_full, frames_selected=frames_selected, output=output)


def parse_resize_arg(resize: Optional[str]) -> Optional[Tuple[int, int]]:
    if not resize:
        return None
    m = re.match(r"(\d+)x(\d+)$", resize.strip())
    if not m:
        raise ValueError("--resize invalid. Use format WxH, e.g. 1280x720")
    return int(m.group(1)), int(m.group(2))


def resolve_input_video_path(input_arg: Optional[str], video_dir: Path) -> Optional[Path]:
    if input_arg:
        candidate = Path(input_arg).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    return get_first_video(video_dir)


def copy_video_to_workdir(source_video: Path, video_dir: Path) -> Path:
    destination = video_dir / source_video.name
    if destination == source_video:
        return source_video
    try:
        shutil.copy2(str(source_video), str(destination))
        return destination
    except Exception:
        return source_video


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceParams:
    window_size: int = 6
    window_step: int = 6
    threshold: float = 0.5
    base_url: str = "http://localhost:1234"
    model: str = "lmstudio-community/InternVL3_5-8B-GGUF:Q8_0"
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    health_timeout: float = 60.0
    prompt: str = ANOMALY_PROMPT


def extract_and_select_frames(
    *,
    video_path: Path,
    frames_dir: Path,
    selected_dir: Path,
    resize: Optional[Tuple[int, int]],
    select_fps: float,
    diff_threshold: float,
) -> Tuple[float, np.ndarray, List[Tuple[Path, float]]]:
    ensure_empty_dir(frames_dir)
    ensure_empty_dir(selected_dir)
    print("[1/4] Extracting and selecting frames…", flush=True)

    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available: install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(video_fps / float(max(0.1, select_fps)))))
    timestamps: List[float] = []
    selected_frames: List[Tuple[Path, float]] = []
    previous_gray = None
    frame_index = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if resize is not None:
                frame_bgr = cv2.resize(frame_bgr, resize, interpolation=cv2.INTER_AREA)

            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            timestamp = (frame_index / video_fps) if video_fps else 0.0
            timestamps.append(timestamp)

            keep_frame = frame_index % stride == 0
            if not keep_frame and previous_gray is not None:
                diff = cv2.absdiff(frame_gray, previous_gray)
                keep_frame = float(diff.mean()) > float(diff_threshold)

            if keep_frame:
                frame_path = selected_dir / f"frame_{frame_index:06d}.jpg"
                ok_write = cv2.imwrite(
                    str(frame_path),
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                if not ok_write:
                    raise RuntimeError(f"Failed to save frame: {frame_path}")
                selected_frames.append((frame_path, float(timestamp)))

            previous_gray = frame_gray
            frame_index += 1
    finally:
        cap.release()

    frame_timestamps = np.array(timestamps, dtype=np.float32)
    duration = float(frame_timestamps[-1]) if len(frame_timestamps) else 0.0
    print(
        "    "
        f"Frames: {len(frame_timestamps)} | Selected: {len(selected_frames)} | "
        f"Video FPS ≈ {video_fps:.2f} | Duration ≈ {duration:.2f}s | "
        f"stride={stride} | diff_th={diff_threshold}"
    )
    return video_fps, frame_timestamps, selected_frames


def build_inference_params(args: argparse.Namespace) -> InferenceParams:
    return InferenceParams(
        window_size=int(args.window_size),
        window_step=int(args.window_step),
        threshold=float(args.threshold),
        base_url=str(args.llama_url),
        model=str(args.llama_model),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        health_timeout=float(args.health_timeout),
        prompt=resolve_anomaly_prompt(getattr(args, "prompt", None), getattr(args, "prompt_file", None)),
    )


def maybe_autostart_server(args: argparse.Namespace, out_dir: Path) -> Tuple[Optional[subprocess.Popen], Optional[int]]:
    if not args.autostart_server:
        return None, None

    log_path = Path(args.server_log).expanduser().resolve() if args.server_log else (out_dir / "llama_server.log")
    print(f"    Starting llama-server on {args.llama_url} (log: {log_path})…", flush=True)
    server_proc = start_llama_server(
        base_url=str(args.llama_url),
        model=str(args.llama_model),
        server_bin=str(args.server_bin),
        ctx_size=int(args.server_ctx_size),
        ngl=int(args.server_ngl),
        extra_args=str(args.server_extra or ""),
        log_path=log_path,
    )
    timeout = float(max(args.health_timeout, 30.0))
    if not _wait_llama_health(str(args.llama_url), timeout_s=timeout):
        print("    Server not ready within expected timeframe. Check logs.")
        return server_proc, 5
    return server_proc, None


def run_window_inference(
    selected_dir: Path,
    selected_frames: List[Tuple[Path, float]],
    params: InferenceParams,
) -> pd.DataFrame:
    if not _wait_llama_health(params.base_url, timeout_s=params.health_timeout):
        print("[WARN] llama-server not ready on", params.base_url)
    frame_paths = sorted([p for p, _ in selected_frames], key=lambda p: p.name)
    name_to_ts: Dict[str, float] = {p.name: float(ts) for p, ts in selected_frames}
    selected_ts_sorted: List[float] = [name_to_ts[p.name] for p in frame_paths]
    records: List[dict] = []
    W = int(params.window_size)
    S = int(params.window_step)
    for b in range(0, len(frame_paths) - (W - 1), S):
        window_paths = frame_paths[b:b + W]
        images = [_load_rgb_image(str(selected_dir / p.name)) for p in window_paths]
        raw = vision_chat(
            images,
            params.prompt,
            base_url=params.base_url,
            model=params.model,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
        )
        data = _parse_json_any(raw) or {}
        score = float(data.get("anomaly_score", 0.0))
        score = max(0.0, min(1.0, score))
        desc = str(data.get("description", "") or "").strip()
        start_name = window_paths[0].name
        center_name = window_paths[W // 2].name
        end_name = window_paths[-1].name
        start_idx = int(re.search(r"(\d+)", start_name).group(1))
        center_idx_name = int(re.search(r"(\d+)", center_name).group(1))
        end_idx = int(re.search(r"(\d+)", end_name).group(1))
        start_ts = float(selected_ts_sorted[b])
        center_ts = float(selected_ts_sorted[b + (W // 2)])
        end_ts = float(selected_ts_sorted[b + W - 1])
        records.append({
            "frame_names": [p.name for p in window_paths],
            "frame_name_start": start_name,
            "frame_name_center": center_name,
            "frame_name_end": end_name,
            "frame_index_start": start_idx,
            "frame_index_center": center_idx_name,
            "frame_index_end": end_idx,
            "timestamp_start": start_ts,
            "timestamp_center": center_ts,
            "timestamp_end": end_ts,
            "anomaly_score": score,
            "description": desc,
            "raw": raw,
        })
    df = pd.DataFrame.from_records(records)
    df["is_anomalous"] = df["anomaly_score"] >= float(params.threshold)
    return df


def compute_segments(df: pd.DataFrame) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    if df.empty:
        return segments
    above = df["is_anomalous"].to_numpy()
    frames_centers = df["frame_index_center"].to_numpy()
    if above.any():
        start: Optional[int] = None
        for i, flag in enumerate(above):
            if flag and start is None:
                start = i
            if (not flag or i == len(above) - 1) and start is not None:
                end = i if (flag and i == len(above) - 1) else i - 1
                segments.append((int(frames_centers[start]), int(frames_centers[end])))
                start = None
    return segments


# ------------------------------
# Animation rendering
# ------------------------------


@dataclass
class RenderParams:
    # layout
    grid_cols: int = 6
    grid_tile: int = 480
    fig_w_in: float = 24.0
    fig_h_in: float = 13.0
    fig_dpi: int = 120
    # timing
    render_fps: float = 2.5  # scrolling speed of output video


def _wrap_text(text: str, max_chars: int = 220, line_w: int = 44, max_sent: int = 2) -> str:
    import textwrap
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    head = " ".join(parts[:max_sent]) if parts else ""
    short = textwrap.shorten(head, width=max_chars, placeholder="…")
    return textwrap.fill(short, width=line_w, break_long_words=False, break_on_hyphens=False)


def _grid_image(selected_dir: Path, names: Sequence[str], cols: int = 3, tile: int = 320) -> Image.Image:
    imgs = [_load_rgb_image(str(selected_dir / p)) for p in names]
    if not imgs:
        return Image.new("RGB", (200, 200), (240, 240, 240))
    cols = max(1, min(cols, len(imgs)))
    rows = ceil(len(imgs) / cols)
    if len(imgs) == 6 and cols >= 6:
        cols, rows = 6, 1
    W, H = cols * tile, rows * tile
    grid = Image.new("RGB", (W, H), (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        thumb = ImageOps.fit(im, (tile, tile), method=Image.LANCZOS)
        grid.paste(thumb, (c * tile, r * tile))
    return grid


def _window_idx_for_frame_index(frame_idx: int, window_step: int, n_win: int) -> int:
    if n_win == 0:
        return 0
    w = frame_idx // max(1, window_step)
    return int(min(max(0, w), n_win - 1))


def _draw_frame(
    fig,
    frame_i: int,
    *,
    output_name: str,
    selected_dir: Path,
    all_names: List[str],
    name_to_ts: Dict[str, float],
    window_names: List[List[str]],
    scores: np.ndarray,
    descs: List[str],
    win_start: np.ndarray,
    win_end: np.ndarray,
    frames_center: np.ndarray,
    segments: List[Tuple[int, int]],
    video_total_s: float,
    window_step: int,
    threshold: float,
    rparams: RenderParams,
) -> None:
    fig.clear()
    gs = GridSpec(3, 2, height_ratios=[2.0, 0.6, 0.8], width_ratios=[1.0, 3.0], figure=fig)
    n_win = len(window_names)
    w_idx = _window_idx_for_frame_index(frame_i, window_step, n_win)
    ts_sel = float(name_to_ts.get(all_names[frame_i], 0.0))

    ax_prev = fig.add_subplot(gs[0, 0])
    ax_prev.set_axis_off()
    curr_path = selected_dir / all_names[frame_i]
    ax_prev.imshow(_load_rgb_image(str(curr_path)))
    ax_prev.set_title(f"Frame {frame_i+1}/{len(all_names)}", fontsize=12)

    ax_grid = fig.add_subplot(gs[0, 1])
    ax_grid.set_axis_off()
    grid = _grid_image(selected_dir, window_names[w_idx], cols=int(rparams.grid_cols), tile=int(rparams.grid_tile))
    ax_grid.imshow(grid)
    ax_grid.set_title(
        f"Window: {window_names[w_idx][0]} → {window_names[w_idx][-1]}  •  score={scores[w_idx]:.2f}",
        fontsize=14,
    )

    ax_txt = fig.add_subplot(gs[1, 0])
    ax_txt.set_axis_off()
    ax_txt.text(
        0.5,
        0.5,
        _wrap_text(descs[w_idx], max_chars=260, line_w=48),
        ha="center",
        va="center",
        fontsize=14,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.8", fc="white", ec="black", lw=1.6),
    )

    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.set_xlim(0, max(1e-6, video_total_s))
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Video time (s)", fontsize=12)
    ax_bar.axvspan(float(win_start[w_idx]), float(win_end[w_idx]), color="#FFC107", alpha=.35, label="Window")
    ax_bar.axvline(ts_sel, color="red", lw=2, label="Selected frame")
    ax_bar.legend(loc="upper right")

    ax_plot = fig.add_subplot(gs[2, :])
    ax_plot.plot(frames_center, scores, color="#4A76FF", lw=2, label="Anomaly score")
    ax_plot.set_ylim(0, 1)
    ax_plot.set_xlabel("Frame number", fontsize=11)
    ax_plot.set_ylabel("Anomaly score", fontsize=11)
    ax_plot.grid(axis="y", ls="--", alpha=.4)
    ax_plot.axhline(float(threshold), color="gray", ls="--", lw=1.5)
    for s0, s1 in segments:
        ax_plot.axvspan(s0, s1, color="red", alpha=.18)
    ax_plot.axvline(frames_center[w_idx], color="red", lw=2)
    ax_plot.set_title(output_name, fontsize=12)

    fig.tight_layout(rect=[0.02, 0.04, 0.995, 0.98])


def render_video(
    output_path: Path,
    *,
    selected_dir: Path,
    all_names: List[str],
    name_to_ts: Dict[str, float],
    df: pd.DataFrame,
    segments: List[Tuple[int, int]],
    video_total_s: float,
    window_step: int,
    threshold: float,
    rparams: RenderParams,
) -> None:
    frames_center = df["frame_index_center"].to_numpy()
    win_start = df["timestamp_start"].to_numpy()
    win_end = df["timestamp_end"].to_numpy()
    scores = df["anomaly_score"].to_numpy()
    descs = df["description"].fillna("").astype(str).tolist()
    window_names: List[List[str]] = df["frame_names"].tolist()
    n_win = len(window_names)
    S = len(all_names)
    if S == 0 or n_win == 0:
        raise RuntimeError("No frames selected or no windows available for rendering")

    use_ffmpeg = writers.is_available("ffmpeg")

    if use_ffmpeg:
        fig = plt.figure(figsize=(rparams.fig_w_in, rparams.fig_h_in), dpi=rparams.fig_dpi)
        writer = FFMpegWriter(fps=float(max(0.1, rparams.render_fps)))
        with writer.saving(fig, str(output_path), dpi=rparams.fig_dpi):
            for frame_i in range(S):
                _draw_frame(
                    fig,
                    frame_i,
                    output_name=output_path.name,
                    selected_dir=selected_dir,
                    all_names=all_names,
                    name_to_ts=name_to_ts,
                    window_names=window_names,
                    scores=scores,
                    descs=descs,
                    win_start=win_start,
                    win_end=win_end,
                    frames_center=frames_center,
                    segments=segments,
                    video_total_s=video_total_s,
                    window_step=window_step,
                    threshold=threshold,
                    rparams=rparams,
                )
                writer.grab_frame()
        return

    if cv2 is None:
        raise RuntimeError("Neither ffmpeg nor OpenCV available to write video")

    tmp_dir = output_path.parent / f"_{output_path.stem}_frames"
    ensure_empty_dir(tmp_dir)
    png_paths: List[Path] = []
    for frame_i in range(S):
        fig = plt.figure(figsize=(rparams.fig_w_in, rparams.fig_h_in), dpi=rparams.fig_dpi)
        _draw_frame(
            fig,
            frame_i,
            output_name=output_path.name,
            selected_dir=selected_dir,
            all_names=all_names,
            name_to_ts=name_to_ts,
            window_names=window_names,
            scores=scores,
            descs=descs,
            win_start=win_start,
            win_end=win_end,
            frames_center=frames_center,
            segments=segments,
            video_total_s=video_total_s,
            window_step=window_step,
            threshold=threshold,
            rparams=rparams,
        )
        png_path = tmp_dir / f"render_{frame_i:06d}.png"
        fig.savefig(png_path)
        plt.close(fig)
        png_paths.append(png_path)

    img0 = cv2.imread(str(png_paths[0]))
    H, W = img0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(output_path), fourcc, float(max(0.1, rparams.render_fps)), (W, H))
    for png_path in png_paths:
        im_bgr = cv2.imread(str(png_path))
        if im_bgr.shape[:2] != (H, W):
            im_bgr = cv2.resize(im_bgr, (W, H))
        vw.write(im_bgr)
    vw.release()
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ------------------------------
# Main CLI
# ------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAD Showcase: generates an animated video with anomaly score/description",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, default=None, help="Input video path (if absent, uses first video in directory)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output MP4 file path")
    parser.add_argument("--workdir", type=str, default=str(Path("assets") / "videos_and_frames"), help="Working directory (frames, output)")
    parser.add_argument("--resize", type=str, default=None, help="Resize frames: e.g. 1280x720 (optional)")
    parser.add_argument("--select-fps", type=float, default=2.0, help="Frequency (approx) of selected frames")
    parser.add_argument("--diff-th", type=float, default=1e18, help="Difference threshold for extra selection (very high=off)")
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--window-step", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--llama-url", type=str, default="http://localhost:1234")
    parser.add_argument("--llama-model", type=str, default="lmstudio-community/InternVL3_5-2B-GGUF:Q8_0")
    parser.add_argument("--prompt", type=str, default=None, help="Inline anomaly prompt override")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a UTF-8 text file containing the anomaly prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--health-timeout", type=float, default=60.0)
    parser.add_argument("--autostart-server", action="store_true", help="Automatically start llama-server")
    parser.add_argument("--server-bin", type=str, default="llama-server", help="llama-server binary")
    parser.add_argument("--server-ctx-size", type=int, default=8192, help="ctx-size for llama-server")
    parser.add_argument("--server-ngl", type=int, default=999, help="-ngl for llama-server")
    parser.add_argument("--server-extra", type=str, default="", help="Extra arguments for llama-server")
    parser.add_argument("--server-log", type=str, default=None, help="Log file for llama-server")
    parser.add_argument("--keep-server", action="store_true", help="Don't stop the server at the end")
    parser.add_argument("--render-fps", type=float, default=2.5, help="FPS of the output video")
    parser.add_argument("--fig-dpi", type=int, default=120)
    parser.add_argument("--fig-w", type=float, default=24.0)
    parser.add_argument("--fig-h", type=float, default=13.0)
    parser.add_argument("--grid-cols", type=int, default=6)
    parser.add_argument("--grid-tile", type=int, default=480)
    return parser.parse_args(argv)


def save_inference_data(
    output_dir: Path,
    video_name: str,
    df: pd.DataFrame,
    segments: List[Tuple[int, int]],
    selected_frames: List[Tuple[Path, float]],
    video_total_s: float,
    threshold: float,
    window_step: int,
) -> Path:
    """Save inference data to JSON for later use by Gradio app."""
    # Convert DataFrame to serializable format
    records = []
    for _, row in df.iterrows():
        records.append({
            "frame_names": row["frame_names"],
            "frame_name_start": row["frame_name_start"],
            "frame_name_center": row["frame_name_center"],
            "frame_name_end": row["frame_name_end"],
            "frame_index_start": int(row["frame_index_start"]),
            "frame_index_center": int(row["frame_index_center"]),
            "frame_index_end": int(row["frame_index_end"]),
            "timestamp_start": float(row["timestamp_start"]),
            "timestamp_center": float(row["timestamp_center"]),
            "timestamp_end": float(row["timestamp_end"]),
            "anomaly_score": float(row["anomaly_score"]),
            "description": row["description"],
            "is_anomalous": bool(row["is_anomalous"]),
        })

    data = {
        "video_name": video_name,
        "video_total_s": video_total_s,
        "threshold": threshold,
        "window_step": window_step,
        "prompt": df.attrs.get("prompt", ANOMALY_PROMPT),
        "segments": segments,
        "selected_frames": [
            {"name": p.name, "timestamp": float(ts)}
            for p, ts in selected_frames
        ],
        "windows": records,
    }

    json_path = output_dir / f"{video_name}_inference_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"    Saved inference data to {json_path}")
    return json_path


def run_pipeline(args: argparse.Namespace) -> int:
    dirs = setup_workdirs(args.workdir)

    input_video = resolve_input_video_path(args.input, dirs.videos)
    if not input_video or not input_video.exists():
        print("No video selected.")
        return 2

    output_path = Path(args.output).expanduser().resolve() if args.output else (dirs.output / f"{input_video.stem}_vad_showcase.mp4")
    dest_video = copy_video_to_workdir(input_video, dirs.videos)

    try:
        resize_tuple = parse_resize_arg(args.resize)
    except ValueError as exc:
        print(str(exc))
        return 2

    video_fps, frame_timestamps, selected_frames = extract_and_select_frames(
        video_path=dest_video,
        frames_dir=dirs.frames_full,
        selected_dir=dirs.frames_selected,
        resize=resize_tuple,
        select_fps=float(args.select_fps),
        diff_threshold=float(args.diff_th),
    )
    if not selected_frames:
        print("No frames selected.")
        return 2

    all_names = [p.name for p, _ in selected_frames]
    name_to_ts: Dict[str, float] = {p.name: float(ts) for p, ts in selected_frames}
    video_total_s = float(frame_timestamps[-1]) if len(frame_timestamps) else 0.0

    print("[2/4] Window inference (vision-chat)…", flush=True)
    server_proc: Optional[subprocess.Popen] = None
    try:
        server_proc, start_error = maybe_autostart_server(args, dirs.output)
        if start_error is not None:
            return start_error

        infer_params = build_inference_params(args)
        df = run_window_inference(dirs.frames_selected, selected_frames, infer_params)
        df.attrs["prompt"] = infer_params.prompt
        if df.empty:
            print("No results from model.")
            return 3

        print("[3/4] Computing anomalous segments…", flush=True)
        segments = compute_segments(df)
        if segments:
            print("    Segments (frame_start, frame_end):", segments)
        else:
            print("    No segments above threshold.")

        # Save inference data for Gradio app
        save_inference_data(
            output_dir=dirs.output,
            video_name=input_video.stem,
            df=df,
            segments=segments,
            selected_frames=selected_frames,
            video_total_s=video_total_s,
            threshold=float(args.threshold),
            window_step=int(args.window_step),
        )

        print(f"[4/4] Rendering video → {output_path}", flush=True)
        rparams = RenderParams(
            grid_cols=int(args.grid_cols),
            grid_tile=int(args.grid_tile),
            fig_w_in=float(args.fig_w),
            fig_h_in=float(args.fig_h),
            fig_dpi=int(args.fig_dpi),
            render_fps=float(args.render_fps),
        )
        try:
            render_video(
                output_path,
                selected_dir=dirs.frames_selected,
                all_names=all_names,
                name_to_ts=name_to_ts,
                df=df,
                segments=segments,
                video_total_s=video_total_s,
                window_step=int(args.window_step),
                threshold=float(args.threshold),
                rparams=rparams,
            )
        except Exception as exc:
            print("Error during rendering:", exc)
            return 4
    finally:
        if server_proc and not args.keep_server:
            print("    Stopping llama-server…", flush=True)
            stop_llama_server(server_proc)

    print("Done.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
