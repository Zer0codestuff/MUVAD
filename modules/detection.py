from queue import Queue
import os
import time

from helpers.ollama_wrap import OllamaModel
from helpers.module import Module

# Create logger
from helpers.logger import getLogger
logger = getLogger("detector")


class Detector(Module):
    def __init__(self,
            queue_in: Queue,
            queue_out: Queue,
            model_name: str,
            prompt: str, 
            batch_size: int = 1,
            parameters: dict = None,
            device: str | None = None,
            host: str | None = None,
            backend: str = "ollama",
            log: str = "INFO",
            **kwargs,
        ) -> None:
        """Loading and initialize the model"""
        
        if parameters is None:
            parameters = {}

        # Initialize object variables
        self.prompt = prompt
        logger.setLevel(log.upper())

        # Load model
        if model_name is None:
            self.model = None
        else:
            backend_norm = (backend or "ollama").lower()
            if backend_norm.startswith("llama"):
                from helpers.llamacpp_wrap import LlamaCppModel
                self.model = LlamaCppModel(model_name, parameters, device, host)
                logger.info(f"Using llama.cpp backend at {host}")
            else:
                self.model = OllamaModel(model_name, parameters, device, host)
                logger.info(f"Using Ollama backend at {host}")
            logger.info(f"Model {model_name} loaded")

        # Initialize module
        super().__init__(queue_in, self.generate, queue_out, batch_size)


    def start(self, save_file: str = "", **kwargs) -> None:
        """Start the model"""

        # Initialize instance variables
        self.save_file = os.path.expanduser(save_file)
        self.times_in = {}
        self.times_out = {}
        self.counter_in = 0
        self.counter_out = 0

        # Check save file
        if self.save_file:
            open(self.save_file, "w").close()

        # Start module
        super().start()


    def generate(self, batch: list[str]) -> list[str]:
        """Thread function for generate model response"""

        if self.model is None:
            return batch
        if not batch:
            return []

        t = time.time()
        for frame in batch:
            self.times_in[self.counter_in] = t
            self.counter_in += 0.5

        # Generate outputs
        formatted_captions = "\n".join(batch)
        prompt = self.prompt.replace("<frames>", formatted_captions).strip()
        response = self.model.generate(prompt)
        output = f"{prompt}\n\n\n{response}\n\n\n\n\n"

        t = time.time()
        for frame in batch:
            self.times_out[self.counter_out] = t
            self.counter_out += 0.5

        # Save outputs in file
        if self.save_file:
            with open(self.save_file, "a") as save_file:
                save_file.write(output)

        return [output]
