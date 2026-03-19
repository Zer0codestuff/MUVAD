from PIL import Image as pil
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

# Get logger
from helpers.logger import getLogger
logger = getLogger("deepseek-vl2")


class DeepSeekVL2:
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
                self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_name)
                self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                )
                self.tokenizer = self.processor.tokenizer
                status = True

            # Handle errors
            except torch.cuda.OutOfMemoryError:
                logger.error(f"Unable to allocate memory. Retrying...")


    def generate(self, prompt: str, attachments: list[pil.Image]) -> list[str]:
        prompt += '\n' + '\n'.join(['<image>' for _ in range(len(attachments))])
        conversation = [
            {"role": "<|User|>", "content": f"<|ref|>{prompt}<|/ref|>."},
            {"role": "<|Assistant|>", "content": ""},
        ]
        try:
            with torch.inference_mode():
                # Encode
                inputs = self.processor(
                    conversations=conversation,
                    images=attachments,
                    force_batchify=True,
                    system_prompt="",
                ).to(self.device, self.dtype)

                # Embed
                inputs_embeds = self.model.prepare_inputs_embeds(**inputs)

                # Generate
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **self.parameters,
                )

                # Decode
                response: str = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                return [response.replace("\n", " ")]

        # Handle errors
        except torch.cuda.OutOfMemoryError:
            logger.error("Memory unsufficient for captioning, retrying...")

        # Clear cuda cache
        torch.cuda.empty_cache()
        return []
