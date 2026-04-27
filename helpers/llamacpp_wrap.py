import time
import os
import atexit
import subprocess
import requests
import json
import io
import base64
import shlex
from concurrent.futures import ThreadPoolExecutor

# Get logger
from helpers.logger import getLogger
logger = getLogger("llama.cpp")


def _normalize_host(host: str | None) -> str:
    if not host:
        return "http://localhost:8080"
    base = host.strip()
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"http://{base}"
    return base.rstrip("/")


def _parse_port_from_host(host: str) -> int:
    try:
        without_proto = host.replace("http://", "").replace("https://", "")
        parts = without_proto.split(":", 1)
        if len(parts) == 2:
            return int(parts[1].split("/")[0])
    except Exception:
        pass
    return 8080


def _safe_text(text: object) -> str:
    return str(text or "").encode("utf-8", errors="replace").decode("utf-8", errors="replace")


# Track spawned llama.cpp servers by host
llamacpp_servers: dict[str, subprocess.Popen] = {}
llamacpp_log_handles: dict[str, object] = {}


def restart_llamacpp_server(host: str, server_cfg: dict | None = None, timeout: float = 1.5) -> None:
    """Start or restart a local llama.cpp server using provided options."""
    global llamacpp_servers

    # Stop if already running
    stop_llamacpp_server(host)

    # Build command
    port = _parse_port_from_host(host)
    cmd: list[str] = []

    server_cfg = server_cfg or {}
    hf = server_cfg.get("hf")
    hf_repo = server_cfg.get("hf_repo")
    hf_file = server_cfg.get("hf_file")
    model_path = server_cfg.get("model")  # local path alternative
    ngl = server_cfg.get("ngl")
    np_val = server_cfg.get("np")
    cont_batching = server_cfg.get("cont_batching") or server_cfg.get("cont-batching")
    flash_attn = server_cfg.get("flash_attn") or server_cfg.get("flash-attn") or server_cfg.get("flash_attention")
    batch_size = server_cfg.get("batch") or server_cfg.get("b")
    ubatch_size = server_cfg.get("ub") or server_cfg.get("ubatch")
    ctx_len = server_cfg.get("ctx_len") or server_cfg.get("c")
    extra_args: list[str] = server_cfg.get("extra", [])

    # Allow full command override
    explicit_cmd = server_cfg.get("cmd") if isinstance(server_cfg, dict) else None
    if explicit_cmd:
        # If provided as a string, split by spaces; otherwise assume list
        if isinstance(explicit_cmd, str):
            cmd = shlex.split(explicit_cmd)
        else:
            cmd = list(explicit_cmd)
        # Ensure the desired port is included; append if missing
        if "--port" not in cmd and "-p" not in cmd:
            cmd += ["--port", str(port)]
    else:
        cmd = [os.environ.get("MUVAD_LLAMA_SERVER_CMD", "llama-server"), "--port", str(port)]

    if not explicit_cmd and model_path:
        cmd += ["-m", str(model_path)]
    elif not explicit_cmd and hf:
        # Allow shorthand: -hf <repo[:variant|file]>
        cmd += ["-hf", str(hf)]
    elif not explicit_cmd:
        if hf_repo:
            cmd += ["--hf-repo", str(hf_repo)]
        if hf_file:
            cmd += ["--hf-file", str(hf_file)]

    if not explicit_cmd and ngl is not None:
        cmd += ["-ngl", str(ngl)]
    if not explicit_cmd and np_val is not None:
        cmd += ["-np", str(np_val)]
    if not explicit_cmd and cont_batching:
        cmd += ["--cont-batching"]
    if not explicit_cmd and flash_attn:
        flash_value = "on" if flash_attn is True else str(flash_attn)
        cmd += ["--flash-attn", flash_value]
    if not explicit_cmd and batch_size is not None:
        cmd += ["-b", str(batch_size)]
    if not explicit_cmd and ubatch_size is not None:
        cmd += ["-ub", str(ubatch_size)]
    if not explicit_cmd and ctx_len is not None:
        cmd += ["-c", str(ctx_len)]
    if not explicit_cmd and extra_args:
        cmd += [str(a) for a in extra_args]

    show_output = bool(server_cfg.get("show_output", False))
    log_path = server_cfg.get("log_file") or server_cfg.get("log_path")
    log_handle = None
    stdout_target = None
    stderr_target = None
    if not show_output:
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(str(log_path))), exist_ok=True)
            log_handle = open(str(log_path), "a", buffering=1, encoding="utf-8", errors="replace")
            stdout_target = log_handle
            stderr_target = subprocess.STDOUT
        else:
            stdout_target = subprocess.DEVNULL
            stderr_target = subprocess.DEVNULL
    logger.debug(f"Starting llama.cpp server on {host}: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=None if show_output else stdout_target,
            stderr=None if show_output else stderr_target,
            env=os.environ.copy(),
        )
    except FileNotFoundError as err:
        if log_handle:
            log_handle.close()
        raise RuntimeError("llama-server executable not found. Install llama.cpp or set parameters.server.cmd.") from err

    time.sleep(timeout)
    if proc.poll() is None:
        llamacpp_servers[host] = proc
        if log_handle:
            llamacpp_log_handles[host] = log_handle
        logger.info(f"llama.cpp server started on {host}")
    else:
        if log_handle:
            log_handle.close()
        raise RuntimeError(f"llama.cpp server exited during startup on {host}")

    atexit.register(stop_llamacpp_server, host)


def stop_llamacpp_server(host: str) -> None:
    global llamacpp_servers
    if host in llamacpp_servers:
        logger.debug(f"Stopping llama.cpp server on {host}...")
        try:
            llamacpp_servers[host].kill()
        except Exception:
            pass
        llamacpp_servers.pop(host, None)
    log_handle = llamacpp_log_handles.pop(host, None)
    if log_handle:
        try:
            log_handle.close()
        except Exception:
            pass


class LlamaCppModel:
    """Minimal wrapper for llama.cpp OpenAI-compatible server.

    Expects a running llama.cpp server (llama-server) exposing /v1 endpoints.
    """

    def __init__(
            self,
            model_name: str,
            parameters: dict = None,
            device: str | None = None,
            host: str | None = None,
        ) -> None:
        if parameters is None:
            parameters = {}
        params = dict(parameters)

        self.model_name = model_name
        self.device = device
        self.host = _normalize_host(host)

        self.max_token = params.pop("max_token", None)
        self.max_tokens = params.pop("max_tokens", None) or self.max_token
        self.temperature = params.pop("temperature", None)
        self.top_p = params.pop("top_p", None)
        self.stop = params.pop("stop", None)
        self.seed = params.pop("seed", None)
        self.autostart = bool(params.pop("autostart", True))
        self.request_timeout = float(params.pop("request_timeout", 120.0))
        self.allow_text_fallback = bool(params.pop("allow_text_fallback", False))
        self.require_ready = bool(params.pop("require_ready", True))
        self.image_format = str(params.pop("image_format", "JPEG") or "JPEG").upper()
        if self.image_format == "JPG":
            self.image_format = "JPEG"
        if self.image_format not in {"JPEG", "PNG", "WEBP"}:
            self.image_format = "JPEG"
        try:
            self.image_quality = max(1, min(100, int(params.pop("image_quality", 85))))
        except Exception:
            self.image_quality = 85

        server_cfg = {}
        nested = params.pop("llamacpp_server", None)
        if isinstance(nested, dict):
            server_cfg.update({k: v for k, v in nested.items() if v is not None})
        nested_server = params.pop("server", None)
        if isinstance(nested_server, dict):
            server_cfg.update({k: v for k, v in nested_server.items() if v is not None})
        local_model = params.pop("model_path", None) or params.pop("local_model", None)
        if local_model:
            server_cfg["model"] = local_model
        for key in ("ngl", "np", "flash_attn", "flash-attn", "flash_attention",
                    "cont_batching", "cont-batching", "ctx_len", "c", "batch", "b",
                    "ubatch", "ub", "extra", "ready_timeout", "show_output",
                    "log_file", "log_path", "startup_timeout"):
            if key in params:
                server_cfg[key] = params.pop(key)
        if "extra" in server_cfg and not isinstance(server_cfg["extra"], list):
            server_cfg["extra"] = [server_cfg["extra"]]

        if "flash-attn" in server_cfg:
            server_cfg["flash_attn"] = server_cfg.pop("flash-attn")
        if "flash_attention" in server_cfg:
            server_cfg["flash_attn"] = server_cfg.pop("flash_attention")
        if "cont-batching" in server_cfg:
            server_cfg["cont_batching"] = server_cfg.pop("cont-batching")
        if "c" in server_cfg:
            server_cfg["ctx_len"] = server_cfg.pop("c")
        if "b" in server_cfg:
            server_cfg["batch"] = server_cfg.pop("b")
        if "ub" in server_cfg:
            server_cfg["ub"] = server_cfg.pop("ub")

        if "model" not in server_cfg and "hf" not in server_cfg:
            server_cfg["hf"] = model_name

        self.parameters = params

        wait_timeout = 5.0
        if any(h in self.host for h in ("localhost", "127.0.0.1", "0.0.0.0")):
            if self.autostart:
                restart_llamacpp_server(
                    self.host,
                    server_cfg,
                    timeout=float(server_cfg.get("startup_timeout", 1.5)),
                )
            else:
                logger.info("llama.cpp autostart disabled for %s", self.host)
            heavy = bool(server_cfg.get("hf") or server_cfg.get("model"))
            wait_timeout = float(server_cfg.get("ready_timeout", 300.0 if heavy else 20.0))
        else:
            logger.info("llama.cpp autostart skipped for remote host %s", self.host)

        self._wait_server_ready(timeout_s=wait_timeout)

    def _wait_server_ready(self, timeout_s: float = 5.0) -> None:
        end = time.time() + timeout_s
        urls = (f"{self.host}/health", f"{self.host}/v1/models")
        while time.time() < end:
            for url in urls:
                try:
                    r = requests.get(url, timeout=0.5)
                    if r.status_code == 200:
                        logger.info(f"Connected to llama.cpp server at {self.host}")
                        return
                except Exception:
                    pass
            time.sleep(0.2)
        message = f"Could not verify llama.cpp server at {self.host}"
        if self.require_ready:
            raise RuntimeError(message)
        logger.warning(f"{message}; continuing anyway")

    def _prepare_common_params(self) -> dict:
        params: dict = {}
        if self.max_tokens is not None:
            params["max_tokens"] = int(self.max_tokens)
        if self.temperature is not None:
            params["temperature"] = float(self.temperature)
        if self.top_p is not None:
            params["top_p"] = float(self.top_p)
        if self.stop is not None:
            params["stop"] = self.stop
        if self.seed is not None:
            params["seed"] = int(self.seed)
        return params

    def _chat_completions(self, prompt: str) -> str:
        url = f"{self.host}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        payload.update(self._prepare_common_params())
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=self.request_timeout)
        r.raise_for_status()
        data = r.json()
        try:
            return _safe_text(data["choices"][0]["message"]["content"])
        except Exception:
            # Fallback for variations
            if data.get("choices"):
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    if "text" in choice:
                        return _safe_text(choice["text"])
                    if "message" in choice and isinstance(choice["message"], dict):
                        return _safe_text(choice["message"].get("content", ""))
            return ""

    def _completions(self, prompt: str) -> str:
        url = f"{self.host}/v1/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        payload.update(self._prepare_common_params())
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=self.request_timeout)
        r.raise_for_status()
        data = r.json()
        try:
            return _safe_text(data["choices"][0]["text"])
        except Exception:
            return ""

    def _encode_image_to_data_url(self, img) -> str | None:
        try:
            if isinstance(img, bytes):
                b = img
            elif isinstance(img, str):
                if img.startswith("data:image"):
                    return img
                if os.path.exists(img):
                    with open(img, "rb") as f:
                        b = f.read()
                else:
                    return None
            elif hasattr(img, "save"):
                buf = io.BytesIO()
                image_format = self.image_format
                mime_subtype = "jpeg" if image_format == "JPEG" else image_format.lower()
                save_kwargs = {}
                if image_format in {"JPEG", "WEBP"}:
                    save_kwargs["quality"] = self.image_quality
                img.convert("RGB").save(buf, format=image_format, **save_kwargs)
                b = buf.getvalue()
            else:
                return None
            b64 = base64.b64encode(b).decode("utf-8")
            if "mime_subtype" not in locals():
                mime_subtype = "png"
            return f"data:image/{mime_subtype};base64,{b64}"
        except Exception:
            return None

    def _chat_with_vision(self, prompt: str, images: list) -> str:
        url = f"{self.host}/v1/chat/completions"
        content = [{"type": "text", "text": prompt}]
        for img in images or []:
            data_url = self._encode_image_to_data_url(img)
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }
        payload.update(self._prepare_common_params())
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=self.request_timeout)
        if r.status_code == 404 or r.status_code == 400:
            # Try selecting default loaded model
            if self._maybe_select_default_model():
                payload["model"] = self.model_name
                r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=self.request_timeout)
        r.raise_for_status()
        data = r.json()
        try:
            return _safe_text(data["choices"][0]["message"]["content"])
        except Exception:
            return ""

    def _maybe_select_default_model(self) -> bool:
        try:
            r = requests.get(f"{self.host}/v1/models", timeout=3)
            r.raise_for_status()
            models = r.json().get("data", [])
            if models:
                mid = models[0].get("id") or models[0].get("object") or models[0].get("name")
                if mid:
                    logger.warning(f"Using default llama.cpp model id: {mid}")
                    self.model_name = mid
                    return True
        except Exception:
            pass
        return False

    def generate_aggregate(self, prompt: str, images: list) -> str:
        """Generate a single response while attaching all images at once."""

        if not images:
            result = self.generate(prompt, images)
            return result if isinstance(result, str) else "".join(result)

        try:
            text = self._chat_with_vision(prompt, images)
            if text:
                return text
        except Exception as err:
            logger.warning(f"Vision aggregate call failed ({err})")

        if not self.allow_text_fallback:
            return ""
        try:
            fallback = self._chat_completions(prompt)
            if fallback:
                return fallback
        except Exception as err2:
            logger.warning(f"Aggregate text-only fallback failed ({err2}); returning empty string")

        return ""

    def generate(self, prompt: str, images: list = None) -> str | list[str]:
        if images is None:
            images = []
        # If images are provided, return one caption per image (list[str])
        if images:
            # Issue one request per image, optionally in parallel (preserve order)
            def _one(idx_img: int, img) -> tuple[int, str]:
                try:
                    text = self._chat_with_vision(prompt, [img])
                    return idx_img, (text or "")
                except Exception as err:
                    logger.warning(f"Vision call failed ({err})")
                    if not self.allow_text_fallback:
                        return idx_img, ""
                    try:
                        return idx_img, (self._completions(prompt) or "")
                    except Exception as err2:
                        logger.warning(f"Completions fallback failed ({err2}); returning empty string")
                        return idx_img, ""

           
            max_workers = max(1, len(images))
            if max_workers == 1:
                outputs: list[str] = []
                for i, img in enumerate(images):
                    _, text = _one(i, img)
                    outputs.append(text)
                return outputs

            outputs: list[str] = [""] * len(images)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_one, i, img) for i, img in enumerate(images)]
                for fut in futures:
                    idx_img, text = fut.result()
                    outputs[idx_img] = text
            return outputs

        # Text-only: return a single string
        try:
            text = self._chat_completions(prompt)
            if not text:
                raise RuntimeError("empty response")
            return text
        except Exception as err:
            logger.warning(f"Chat API failed ({err}); retrying with Completions endpoint...")
            try:
                return self._completions(prompt)
            except Exception as err2:
                logger.warning(f"Completions endpoint failed ({err2}); returning empty string")
                return ""
