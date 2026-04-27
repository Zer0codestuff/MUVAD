from queue import Queue
import os
import shutil
import time

# Import structures
from helpers.structs import Frame
from helpers.module import Module

# Get logger
from helpers.logger import getLogger
logger = getLogger("selector")

class Selector(Module):
    """Frame selector to pass through only few frames"""

    def __init__(self,
            frames_database: Queue[Frame],
            selected_frames_database: Queue[Frame],
            batch_size: int,
            log: str = "INFO",
            **kwargs,
        ) -> None:
        """Initialize the selector"""

        # Initialize object variables
        logger.setLevel(log.upper())

        # Initialize module
        super().__init__(frames_database, self.select, selected_frames_database, batch_size)


    def start(self, save_dir: str = "", **kwargs) -> None:
        """Start the selector"""
        
        # Initialize instance variables
        self.save_dir = os.path.expanduser(save_dir)
        self.times_out = {}
        self.frames_selected_count = 0
        self.save_ext = str(kwargs.get("save_ext", "png") or "png").lower().lstrip(".")
        if self.save_ext not in {"png", "jpg", "jpeg", "webp"}:
            self.save_ext = "png"
        try:
            self.jpeg_quality = max(1, min(100, int(kwargs.get("jpeg_quality", 85))))
        except Exception:
            self.jpeg_quality = 85
        
        # Check save directory
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            elif not os.path.isdir(self.save_dir):
                logger.error(f"{self.save_dir} already exists and it isn't a directory. Removing it...")
                os.remove(self.save_dir)
                os.mkdir(self.save_dir)
            elif len(os.listdir(self.save_dir)) != 0:
                logger.error(f"{self.save_dir} already exists and it isn't empty. Removing it...")
                shutil.rmtree(self.save_dir, ignore_errors=True)
                os.mkdir(self.save_dir)

        # Start module
        super().start()


    def select(self, batch: list[Frame]) -> list[Frame]:
        """Thread function for select the frames"""

        if not batch:
            return []

        # Insert selected frame in results
        selected_frame = batch[0]
        logger.debug(f"Frame {selected_frame.timestamp} selected")

        # Save frame in save_dir
        if self.save_dir:
            save_path = os.path.join(self.save_dir, f"frame_{selected_frame.timestamp}.{self.save_ext}")
            if self.save_ext in {"jpg", "jpeg"}:
                selected_frame.image.convert("RGB").save(save_path, quality=self.jpeg_quality, optimize=False)
            elif self.save_ext == "webp":
                selected_frame.image.convert("RGB").save(save_path, quality=self.jpeg_quality, method=0)
            else:
                selected_frame.image.save(save_path)

        self.times_out[selected_frame.timestamp] = time.time()
        try:
            self.frames_selected_count += 1
        except Exception:
            pass

        # Return selected frames
        return [selected_frame]
