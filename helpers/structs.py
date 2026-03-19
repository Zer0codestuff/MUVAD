from PIL import Image as pil

class Frame:
    def __init__(self, image: pil.Image, timestamp: float) -> None:
        self.image = image
        self.timestamp = round(timestamp, 3)
