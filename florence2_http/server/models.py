# Using code from https://colab.research.google.com/#scrollTo=43333b69-5484-4c16-b3cf-331d74c36780&fileId=https%3A//huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
from enum import Enum 
from typing import Optional, Dict

class FlorenceModels(Enum):
    BASE = "microsoft/Florence-2-base" 
    LARGE = "microsoft/Florence-2-large"
    BASE_FT = "microsoft/Florence-2-base-ft"
    LARGE_FT = "microsoft/Florence-2-large-ft"

class FlorenceTask(Enum):
    CAPTION = "<CAPTION>"
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    OBJECT_DETECTION = "<OD>"
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    REFERRING_EXPRESSION_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"
    REGION_TO_SEGMENTATION = "<REGION_TO_SEGMENTATION>"
    OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    REGION_TO_CATEGORY = "<REGION_TO_CATEGORY>"
    REGION_TO_DESCRIPTION = "<REGION_TO_DESCRIPTION>"
    OCR = "<OCR>"
    OCR_WITH_REGION = "<OCR_WITH_REGION>"

class Florence2:
    def __init__(self, model_type: FlorenceModels):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_id = model_type.value
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def run_task(self, task: FlorenceTask, image: Image, text_input: Optional[str] = None) -> Dict:
        prompt = FlorenceTask.value
        if text_input is not None:
            prompt = f"{prompt}{text_input}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
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
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer

