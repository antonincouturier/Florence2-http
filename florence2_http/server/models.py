# Using code from https://colab.research.google.com/#scrollTo=43333b69-5484-4c16-b3cf-331d74c36780&fileId=https%3A//huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb

import base64
from io import BytesIO
from typing import Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from florence2_http.shared import FlorenceModel, FlorenceTask


class Florence2:
    def __init__(self, model_type: FlorenceModel):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = model_type.value
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype="auto"
            )
            .eval()
            .to(self.device)
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def run_task(
        self, task: FlorenceTask, image_base64: str, text_input: Optional[str] = None
    ) -> Dict:
        prompt = task.value
        if text_input is not None:
            prompt = f"{prompt}{text_input}"
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, torch.float16
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task.value, image_size=(image.width, image.height)
        )
        return parsed_answer
