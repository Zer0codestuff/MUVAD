#!/usr/bin/env python3
"""Small Gradio app that runs the VAD showcase pipeline."""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
GRADIO_TMP_DIR = REPO_ROOT / "tmp" / "gradio_temp"
GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TMP_DIR))
ASSETS_VIDEOS_DIR = (REPO_ROOT / "assets" / "videos_and_frames").resolve()

try:
    import gradio as gr  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Gradio is not installed. Install it with 'pip install gradio'."
    ) from exc


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demos.vad_showcase import ANOMALY_PROMPT, parse_args, run_pipeline  # type: ignore


def _server_video_choices() -> list[str]:
    videos_subdir = ASSETS_VIDEOS_DIR / "videos"
    if not videos_subdir.exists():
        return []

    allowed_suffixes = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    choices: list[str] = []

    for path in sorted(videos_subdir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        try:
            relative = path.relative_to(videos_subdir)
        except ValueError:
            continue
        choices.append(relative.as_posix())

    return choices


def _resolve_server_video(selection: Optional[str]) -> Optional[Path]:
    if not selection:
        return None

    videos_subdir = ASSETS_VIDEOS_DIR / "videos"
    candidate = (videos_subdir / selection).resolve()
    try:
        candidate.relative_to(videos_subdir)
    except ValueError:
        return None

    return candidate


def _build_args(
    *,
    video_path: Path,
    prompt: str,
    resize: str,
    select_fps: float,
    diff_th: float,
    window_size: int,
    window_step: int,
    threshold: float,
    llama_url: str,
    llama_model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    health_timeout: float,
    autostart_server: bool,
    server_bin: str,
    server_ctx_size: int,
    server_ngl: int,
    server_extra: str,
    server_log: str,
    keep_server: bool,
    render_fps: float,
    fig_dpi: int,
    fig_w: float,
    fig_h: float,
    grid_cols: int,
    grid_tile: int,
) -> Tuple[argparse.Namespace, Path]:
    args = parse_args([])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (REPO_ROOT / "tmp" / "gradio_runs" / f"run_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    args.workdir = str(run_dir)
    args.input = str(video_path)
    args.prompt = str(prompt)
    args.prompt_file = None
    args.resize = resize.strip() or None
    args.select_fps = float(select_fps)
    args.diff_th = float(diff_th)
    args.window_size = int(window_size)
    args.window_step = int(window_step)
    args.threshold = float(threshold)
    args.llama_url = str(llama_url)
    args.llama_model = str(llama_model)
    args.max_tokens = int(max_tokens)
    args.temperature = float(temperature)
    args.top_p = float(top_p)
    args.health_timeout = float(health_timeout)
    args.autostart_server = bool(autostart_server)
    args.server_bin = str(server_bin)
    args.server_ctx_size = int(server_ctx_size)
    args.server_ngl = int(server_ngl)
    args.server_extra = str(server_extra)
    args.server_log = server_log.strip() or None
    args.keep_server = bool(keep_server)
    args.render_fps = float(render_fps)
    args.fig_dpi = int(fig_dpi)
    args.fig_w = float(fig_w)
    args.fig_h = float(fig_h)
    args.grid_cols = int(grid_cols)
    args.grid_tile = int(grid_tile)

    output_path = (run_dir / "output" / f"{video_path.stem}_vad_showcase.mp4").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(output_path)

    return args, output_path


def _run_analysis(
    video_path: Optional[str],
    server_video: Optional[str],
    prompt: str,
    resize: str,
    select_fps: float,
    diff_th: float,
    window_size: int,
    window_step: int,
    threshold: float,
    llama_url: str,
    llama_model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    health_timeout: float,
    autostart_server: bool,
    server_bin: str,
    server_ctx_size: int,
    server_ngl: int,
    server_extra: str,
    server_log: str,
    keep_server: bool,
    render_fps: float,
    fig_dpi: int,
    fig_w: float,
    fig_h: float,
    grid_cols: int,
    grid_tile: int,
) -> Iterator[Tuple[Any, str, Any]]:
    video_idle = gr.update(value=None, label="Generated video (16:9)")
    loading_hide = gr.update(visible=False)

    selected_path: Optional[Path] = None

    if video_path:
        selected_path = Path(video_path)
    elif server_video:
        selected_path = _resolve_server_video(server_video)
        if selected_path is None:
            yield video_idle, "Invalid server video selection.", loading_hide
            return

    if selected_path is None:
        yield video_idle, "Please upload a video or select one from the server.", loading_hide
        return

    source = selected_path
    if not source.exists():
        yield video_idle, f"File not found: {source}", loading_hide
        return

    args, output_path = _build_args(
        video_path=source,
        prompt=prompt,
        resize=resize,
        select_fps=select_fps,
        diff_th=diff_th,
        window_size=window_size,
        window_step=window_step,
        threshold=threshold,
        llama_url=llama_url,
        llama_model=llama_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        health_timeout=health_timeout,
        autostart_server=autostart_server,
        server_bin=server_bin,
        server_ctx_size=server_ctx_size,
        server_ngl=server_ngl,
        server_extra=server_extra,
        server_log=server_log,
        keep_server=keep_server,
        render_fps=render_fps,
        fig_dpi=fig_dpi,
        fig_w=fig_w,
        fig_h=fig_h,
        grid_cols=grid_cols,
        grid_tile=grid_tile,
    )

    video_placeholder = gr.update(value=None, label="Generated video (16:9) - processing...")
    loading_show = gr.update(visible=True)
    accumulated_logs = "Starting pipeline...\n"
    yield video_placeholder, accumulated_logs, loading_show

    import subprocess
    import threading
    import queue

    def stream_output(pipe, q):
        try:
            for line in iter(pipe.readline, ""):
                if line:
                    q.put(line)
        finally:
            pipe.close()

    # Run pipeline in subprocess to capture real-time output
    cmd = [
        sys.executable,
        "-u",  # Unbuffered
        str(REPO_ROOT / "demos" / "vad_showcase.py"),
        "--workdir", args.workdir,
        "--input", args.input,
        "--output", args.output,
        "--prompt", args.prompt,
        "--select-fps", str(args.select_fps),
        "--diff-th", str(args.diff_th),
        "--window-size", str(args.window_size),
        "--window-step", str(args.window_step),
        "--threshold", str(args.threshold),
        "--llama-url", args.llama_url,
        "--llama-model", args.llama_model,
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--health-timeout", str(args.health_timeout),
        "--server-bin", args.server_bin,
        "--server-ctx-size", str(args.server_ctx_size),
        "--server-ngl", str(args.server_ngl),
        "--render-fps", str(args.render_fps),
        "--fig-dpi", str(args.fig_dpi),
        "--fig-w", str(args.fig_w),
        "--fig-h", str(args.fig_h),
        "--grid-cols", str(args.grid_cols),
        "--grid-tile", str(args.grid_tile),
    ]

    if args.resize:
        cmd.extend(["--resize", args.resize])
    if args.autostart_server:
        cmd.append("--autostart-server")
    if args.server_extra:
        cmd.extend(["--server-extra", args.server_extra])
    if args.server_log:
        cmd.extend(["--server-log", args.server_log])
    if args.keep_server:
        cmd.append("--keep-server")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        q: queue.Queue = queue.Queue()
        thread = threading.Thread(target=stream_output, args=(proc.stdout, q))
        thread.daemon = True
        thread.start()

        while proc.poll() is None:
            try:
                line = q.get(timeout=0.5)
                accumulated_logs += line
                yield video_placeholder, accumulated_logs, loading_show
            except queue.Empty:
                continue

        # Get remaining lines
        while not q.empty():
            line = q.get_nowait()
            accumulated_logs += line

        exit_code = proc.wait()

        if exit_code == 5:
            accumulated_logs += "\n\nERROR CODE 5: llama-server failed to start or is not responding.\n"
            accumulated_logs += "Check that:\n"
            accumulated_logs += "  1. llama-server is installed and available in PATH\n"
            accumulated_logs += "  2. The configured model is available\n"
            accumulated_logs += "  3. The target port is not already in use\n"
            accumulated_logs += "  4. There is enough GPU memory\n"
            yield video_idle, accumulated_logs, loading_hide
            return

        if exit_code != 0:
            accumulated_logs += f"\n\nPipeline finished with exit code {exit_code}.\n"
            yield video_idle, accumulated_logs, loading_hide
            return

        if not output_path.exists():
            accumulated_logs += "\n\nThe output video was not created.\n"
            yield video_idle, accumulated_logs, loading_hide
            return

        accumulated_logs += f"\n\nGenerated video: {output_path}\n"
        video_ready = gr.update(value=str(output_path), label="Generated video (16:9)")
        yield video_ready, accumulated_logs, loading_hide

    except Exception as exc:
        yield video_idle, f"Unexpected error: {exc}\n", loading_hide


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="VAD Showcase UI") as demo:
        gr.Markdown(
            """
            ## VAD Showcase

            Upload or select a video below, configure settings, then press **Run analysis**.
            """.strip()
        )

        with gr.Row():
            video_input = gr.File(
                label="Upload video",
                file_types=["video"],
                type="filepath",
            )
            server_video_choices = _server_video_choices()
            server_video = gr.Dropdown(
                label="Server video library (assets/videos_and_frames/videos)",
                choices=server_video_choices,
                value=None,
                allow_custom_value=False,
                interactive=True,
            )

        video_output = gr.Video(
            label="Generated video (16:9)",
            autoplay=False,
            height=900,
            width=1600,
        )
        loading_indicator = gr.HTML(
            """
            <style>
            .vad-spinner {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin-top: 16px;
                color: #4a4a4a;
                gap: 12px;
            }
            .vad-spinner__circle {
                width: 48px;
                height: 48px;
                border: 4px solid rgba(0, 0, 0, 0.12);
                border-top-color: #4a76ff;
                border-radius: 50%;
                animation: vad-spin 0.9s linear infinite;
            }
            @keyframes vad-spin {
                to { transform: rotate(360deg); }
            }
            </style>
            <div class="vad-spinner">
                <div class="vad-spinner__circle"></div>
                <div>Generating video...</div>
            </div>
            """,
            visible=False,
        )

        with gr.Accordion("Analysis options", open=False):
            prompt = gr.Textbox(label="Anomaly prompt", value=ANOMALY_PROMPT, lines=16)
            resize = gr.Textbox(label="Resize (e.g. 1280x720)", value="", placeholder="Leave empty to keep the original size")
            select_fps = gr.Slider(label="Select FPS", minimum=0.1, maximum=10.0, value=2.0, step=0.1)
            diff_th = gr.Number(label="Diff threshold", value=1e18)
            window_size = gr.Slider(label="Window size", minimum=3, maximum=12, value=6, step=1)
            window_step = gr.Slider(label="Window step", minimum=1, maximum=12, value=6, step=1)
            threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05)
            llama_url = gr.Textbox(label="Llama server URL", value="http://localhost:1234")
            llama_model = gr.Textbox(label="Llama model", value="lmstudio-community/InternVL3_5-2B-GGUF:Q8_0")
            max_tokens = gr.Slider(label="Max tokens", minimum=64, maximum=2048, value=512, step=16)
            temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            health_timeout = gr.Slider(label="Health timeout (s)", minimum=10, maximum=180, value=60, step=5)
            autostart_server = gr.Checkbox(label="Autostart llama-server", value=True)
            server_bin = gr.Textbox(label="llama-server bin", value="llama-server")
            server_ctx_size = gr.Slider(label="Server ctx-size", minimum=1024, maximum=32768, value=8192, step=512)
            server_ngl = gr.Slider(label="Server -ngl", minimum=0, maximum=4096, value=999, step=1)
            server_extra = gr.Textbox(label="Extra args", value="")
            server_log = gr.Textbox(label="Log file (optional)", value="")
            keep_server = gr.Checkbox(label="Keep server running", value=False)
            render_fps = gr.Slider(label="Render FPS", minimum=0.5, maximum=10.0, value=2.5, step=0.1)
            fig_dpi = gr.Slider(label="Figure DPI", minimum=60, maximum=300, value=120, step=10)
            fig_w = gr.Slider(label="Figure width", minimum=10.0, maximum=32.0, value=24.0, step=0.5)
            fig_h = gr.Slider(label="Figure height", minimum=6.0, maximum=24.0, value=13.0, step=0.5)
            grid_cols = gr.Slider(label="Grid columns", minimum=1, maximum=12, value=6, step=1)
            grid_tile = gr.Slider(label="Grid tile size", minimum=160, maximum=640, value=480, step=10)

        run_button = gr.Button("Run analysis", variant="primary")
        logs_box = gr.Textbox(label="Pipeline logs", lines=18)

        run_button.click(
            fn=_run_analysis,
            inputs=[
                video_input,
                server_video,
                prompt,
                resize,
                select_fps,
                diff_th,
                window_size,
                window_step,
                threshold,
                llama_url,
                llama_model,
                max_tokens,
                temperature,
                top_p,
                health_timeout,
                autostart_server,
                server_bin,
                server_ctx_size,
                server_ngl,
                server_extra,
                server_log,
                keep_server,
                render_fps,
                fig_dpi,
                fig_w,
                fig_h,
                grid_cols,
                grid_tile,
            ],
            outputs=[video_output, logs_box, loading_indicator],
            stream_every=0.5,
        )

    return demo


def main(argv: Optional[Tuple[str, ...]] = None) -> None:
    parser = argparse.ArgumentParser(description="Gradio interface for the VAD showcase")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the Gradio server")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server")
    parser.add_argument("--share", action="store_true", help="Share the interface publicly")
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Optional basic auth 'user:password'",
    )
    args = parser.parse_args(argv)

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

