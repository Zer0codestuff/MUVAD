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
        self.timestamp = 0

        # Open video
        self.video = cv2.VideoCapture(self.video_url)
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
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

        success, frame = self.video.read() # Extract frame

        # If no more frame, close video and end thread
        if not success:
            self.video.release()
            self.queue_in_end()
            return []

        # Convert frame to image
        image = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        logger.debug(f"Frame {self.timestamp} read")
        if self.resize:
            image = image.resize(self.resize)

        # Save frame in save_dir
        if self.save_dir:
            image.save(os.path.join(self.save_dir, f"frame_{self.timestamp}.png"))

        frame = Frame(image, self.timestamp) # Create new frame object
        self.timestamp += 1/self.video_fps # Read next frame with its timestamp
        self.processed_frames += 1

        # Return new frames to advance
        return [frame]
