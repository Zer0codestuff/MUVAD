from PIL import Image as pil
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Get logger
from helpers.logger import getLogger
logger = getLogger("blip2")


class Blip2:
    def __init__(self,
            model_name: str,
            parameters: dict = None,
            device: str = "cpu",
        ) -> None:
        if parameters is None:
            parameters = {}

        # Initialize object variables
        self.device = torch.device(device)
        self.dtype = getattr(torch, parameters.pop("dtype", "float32"))
        self.parameters = parameters

        status = False
        while not status:
            try:
                # Initialize model
                self.processor: Blip2Processor = Blip2Processor.from_pretrained(model_name)
                self.model: Blip2ForConditionalGeneration = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                )
                status = True

            # Handle errors
            except torch.cuda.OutOfMemoryError:
                logger.error(f"Unable to allocate memory. Retrying...")


    def generate(self, prompt: str, attachments: list[pil.Image]) -> list[str]:
        try:
            with torch.inference_mode():
                # Encode
                inputs = self.processor(
                    text=[prompt]*len(attachments),
                    images=attachments,
                    return_tensors="pt",
                ).to(self.device, self.dtype)

                # Generate
                outputs = self.model.generate(**inputs, **self.parameters)

                # Decode
                captions: list[str] = self.processor.batch_decode(outputs, skip_special_tokens=True)
                return [caption.replace("\n", " ") for caption in captions]

        # Handle errors
        except torch.cuda.OutOfMemoryError:
            logger.error("Memory unsufficient for captioning, retrying...")

        # Clear cuda cache
        torch.cuda.empty_cache()
        return []