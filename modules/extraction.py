from queue import Queue
import os
import shutil
from PIL import Image as pil
import cv2

from helpers.structs import Frame
from helpers.module import Module

# Get logger
from helpers.logger import getLogger
logger = getLogger("extractor")


class Extractor(Module):
    """Frame extractor from video file or stream"""

    def __init__(self,
            frames_database: Queue[Frame],
            timeout: float = 0.1,
            resize: list[int] | None = None,
            log: str = "INFO",
            **kwargs,
        ) -> None:
        """Initialize the extractor"""

        logger.setLevel(log.upper())
        self.resize = resize

        # Initialize module
        super().__init__(None, self.extract, frames_database, timeout=timeout)


    def start(self, video_url: str, save_dir: str = "", **kwargs) -> None:
        """Start the extractor"""

        # Initialize instance variables
        self.video_url = os.path.expanduser(video_url)
        self.save_dir = os.path.expanduser(save_dir)
        self.frame_stride = max(1, int(kwargs.get("frame_stride", 1) or 1))
        self.start_time = max(0.0, float(kwargs.get("start_time", 0.0) or 0.0))
        raw_end_time = kwargs.get("end_time")
        self.end_time = float(raw_end_time) if raw_end_time is not None else None
        self.timestamp = self.start_time

        # Open video
        self.video = cv2.VideoCapture(self.video_url)
        self.video_fps = float(self.video.get(cv2.CAP_PROP_FPS) or 30.0)
        # Try to read total frames from metadata (may be 0 or unavailable depending on container)
        try:
            _tot = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_total_frames = _tot if _tot > 0 else None
        except Exception:
            self.video_total_frames = None
        # Runtime processed frames counter
        self.processed_frames = 0
        if self.video.isOpened():
            logger.info(f"Video {self.video_url} opened")
        else:
            logger.error(f"Unable to open the video {self.video_url}")
            exit()

        self.current_frame_index = int(round(self.start_time * self.video_fps))
        if self.current_frame_index > 0:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        self.end_frame_index = None
        if self.end_time is not None:
            self.end_frame_index = max(self.current_frame_index, int(round(self.end_time * self.video_fps)))

        # Check save directory
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            elif not os.path.isdir(self.save_dir):
                logger.warning(f"{self.save_dir} already exists and it isn't a directory. Removing it...")
                os.remove(self.save_dir)
                os.mkdir(self.save_dir)
            elif len(os.listdir(self.save_dir)) != 0:
                logger.warning(f"{self.save_dir} already exists and it isn't empty. Removing it...")
                shutil.rmtree(self.save_dir, ignore_errors=True)
                os.mkdir(self.save_dir)

        # Start module
        super().start()


    def extract(self, batch: list) -> list[Frame]:
        """Thread function for extract one frame from video"""

        if self.end_frame_index is not None and self.current_frame_index >= self.end_frame_index:
            self.video.release()
            self.queue_in_end()
            return []

        success, frame = self.video.read() # Extract frame

        # If no more frame, close video and end thread
        if not success:
            self.video.release()
            self.queue_in_end()
            return []

        timestamp = self.current_frame_index / self.video_fps

        # Convert frame to image
        image = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        logger.debug(f"Frame {timestamp} read")
        if self.resize:
            image = image.resize(self.resize)

        # Save frame in save_dir
        if self.save_dir:
            image.save(os.path.join(self.save_dir, f"frame_{timestamp}.png"))

        frame = Frame(image, timestamp) # Create new frame object
        self.processed_frames += 1
        self.current_frame_index += 1

        skipped = 0
        while skipped < self.frame_stride - 1:
            if self.end_frame_index is not None and self.current_frame_index >= self.end_frame_index:
                break
            if not self.video.grab():
                break
            self.current_frame_index += 1
            skipped += 1
        self.timestamp = self.current_frame_index / self.video_fps

        # Return new frames to advance
        return [frame]
