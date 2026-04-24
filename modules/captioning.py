from queue import Queue
import os
import time
import torch
import sys

from helpers.structs import Frame
from helpers.module import Module

# Create logger
from helpers.logger import getLogger
logger = getLogger("captioner")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Captioner(Module):
    def __init__(self,
            queue_in: Queue,
            queue_out: Queue,
            model_name: str,
            prompt: str = "", 
            batch_size: int = 1,
            parameters: dict = None,
            device: str = "cpu",
            host: str | None = None,
            backend: str | None = None,
            log: str = "INFO",
            aggregate: bool = False,
            aggregate_frames_tag: str = "FramesCount",
            aggregate_timestamp_joiner: str = ", ",
            **kwargs,
        ) -> None:
        """Loading and initialize the model"""
        
        if parameters is None:
            parameters = {}

        # Initialize object variables
        self.prompt = prompt
        logger.setLevel(log.upper())
        self._backend = (backend or "").lower()
        self.warmup_seconds: float = 0.0
        self.aggregate_outputs: bool = bool(aggregate)
        self.aggregate_frames_tag: str = str(aggregate_frames_tag)
        self.aggregate_timestamp_joiner: str = str(aggregate_timestamp_joiner)

        # Load model
        backend_norm = self._backend
        if backend_norm.startswith("llama"):
            from helpers.llamacpp_wrap import LlamaCppModel
            self.model = LlamaCppModel(model_name, parameters, device, host)
            logger.info(f"Using llama.cpp backend at {host}")
        else:
            if "florence" in model_name.lower():
                from models.florence2 import Florence2 as SpecificModel
            elif "deepseek-vl2" in model_name.lower():
                from models.deepseekvl2 import DeepSeekVL2 as SpecificModel
            elif "blip2" in model_name.lower():
                from models.blip2 import Blip2 as SpecificModel
            else:
                raise NotImplementedError(f"Model {model_name} is not implemented")

            self.model = SpecificModel(model_name, parameters, device)
            logger.info(f"Model {model_name} loaded")

        # Initialize module
        super().__init__(queue_in, self.generate, queue_out, batch_size)


    def start(self, save_file: str = "", random_seed: int | None = None, warmup_timeout: float | None = None, **kwargs) -> None:
        """Start the model"""

        # Initialize instance variables
        self.save_file = os.path.expanduser(save_file)
        self.times_in = {}
        self.times_out = {}
        self.frames_captioned_count: int = 0
        self.max_retries = int(kwargs.get("max_retries", 8))

        # Set random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)

        if "aggregate" in kwargs:
            self.aggregate_outputs = bool(kwargs["aggregate"])
        if "aggregate_frames_tag" in kwargs:
            try:
                self.aggregate_frames_tag = str(kwargs["aggregate_frames_tag"])
            except Exception:
                self.aggregate_frames_tag = "FramesCount"
        if "aggregate_timestamp_joiner" in kwargs:
            try:
                self.aggregate_timestamp_joiner = str(kwargs["aggregate_timestamp_joiner"])
            except Exception:
                self.aggregate_timestamp_joiner = ", "

        # Check save file
        if self.save_file:
            save_dir = os.path.dirname(self.save_file)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            open(self.save_file, "w", encoding="utf-8", errors="replace").close()

        # Warm up backend (avoid losing first frames when llama.cpp is still loading)
        try:
            if self._backend.startswith("llama"):
                _t0 = time.time()
                self._warmup_backend(timeout_s=float(warmup_timeout) if warmup_timeout is not None else 120.0)
                self.warmup_seconds = time.time() - _t0
        except Exception as _:
            # Non-fatal: continue even if warmup fails
            self.warmup_seconds = 0.0

        # Start module
        super().start()


    def generate(self, batch: list[Frame]) -> list[str]:
        """Thread function for generate model response"""

        if not batch:
            return []

        t = time.time()
        for frame in batch:
            self.times_in[frame.timestamp] = t

        # Generate captions
        captions: list[str] = []
        retries = 0
        max_retries = max(1, self.max_retries)
        aggregate_response: str = ""
        if self.aggregate_outputs:
            while True:
                aggregate_response = self._generate_aggregate(batch).strip()
                if aggregate_response:
                    break
                retries += 1
                if retries >= max_retries:
                    break
                time.sleep(0.5)
            if not aggregate_response and self._backend.startswith("llama"):
                raise RuntimeError("llama.cpp captioner returned an empty aggregate response")
            results = [self._format_aggregate_output(batch, aggregate_response)]
        else:
            while True:
                captions = self.model.generate(self.prompt, [frame.image for frame in batch])
                # Accept only when all captions are non-empty strings
                if isinstance(captions, list) and captions and all((c or "").strip() for c in captions):
                    break
                retries += 1
                if retries >= max_retries:
                    # Give up to avoid infinite loop; return whatever we got
                    break
                time.sleep(0.5)
            if self._backend.startswith("llama") and (
                not isinstance(captions, list)
                or len(captions) != len(batch)
                or not all((c or "").strip() for c in captions)
            ):
                raise RuntimeError("llama.cpp captioner returned incomplete per-frame responses")

            # Add timestamps
            results = [f"- {frame.timestamp}: {caption}" for caption, frame in zip(captions, batch)]

        t = time.time()
        for frame in batch:
            self.times_out[frame.timestamp] = t

        # Update processed frames counter (frames that reached the captioner)
        try:
            self.frames_captioned_count += len(batch)
        except Exception:
            pass

        # Save outputs in file
        if self.save_file:
            with open(self.save_file, "a", encoding="utf-8", errors="replace") as save_file:
                for caption in results:
                    save_file.write(f"{caption}\n")

        return results

    def _generate_aggregate(self, batch: list[Frame]) -> str:
        images = [frame.image for frame in batch]
        prompt = self.prompt
        try:
            if hasattr(self.model, "generate_aggregate"):
                return str(self.model.generate_aggregate(prompt, images) or "")
            response = self.model.generate(prompt, images)
        except Exception as err:
            logger.warning(f"Caption generation failed: {err}")
            return ""

        if isinstance(response, list):
            return "\n".join(str(item or "") for item in response)
        return str(response or "")

    def _format_aggregate_output(self, batch: list[Frame], response: str) -> str:
        timestamps = self.aggregate_timestamp_joiner.join(str(frame.timestamp) for frame in batch)
        metadata_lines = [
            f"- frames: {timestamps}",
            f"{self.aggregate_frames_tag}={len(batch)}",
        ]
        response = response if response else ""
        if response and not response.endswith("\n"):
            response = f"{response}"
        return "\n".join(metadata_lines + [response])

    def _warmup_backend(self, timeout_s: float = 120.0) -> None:
        """Block until the backend responds with a non-empty output for a trivial request."""
        try:
            from PIL import Image as pil
        except Exception:
            pil = None

        end = time.time() + max(1.0, float(timeout_s))
        dummy_image = None
        if pil is not None:
            try:
                dummy_image = pil.new("RGB", (8, 8), (255, 255, 255))
            except Exception:
                dummy_image = None

        last_log = 0.0
        while time.time() < end:
            try:
                # Require a successful vision response (no text-only fallback)
                if (dummy_image is not None) and hasattr(self.model, "_chat_with_vision"):
                    out = getattr(self.model, "_chat_with_vision")("ping", [dummy_image])
                    if (out or "").strip():
                        return
                else:
                    # If we cannot create a dummy image or vision path is unavailable, keep waiting
                    pass
            except Exception:
                pass
            now = time.time()
            if now - last_log > 5.0:
                logger.info("Waiting for captioner backend to be ready...")
                last_log = now
            time.sleep(1.0)
        logger.warning("Captioner backend warmup timed out; continuing anyway")
