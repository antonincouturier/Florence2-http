import base64
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional

import requests

from florence2_http.shared import FlorenceTask


class CaptionVerbosity(Enum):
    SIMPLE = auto()
    DETAILED = auto()
    VERY_DETAILED = auto()


class ObjectDetectionMode(Enum):
    DEFAULT = auto()
    DENSE_CAPTION = auto()
    REGION_PROPOSAL = auto()
    CAPTION_GROUNDING = auto()
    REGION_CATEGORY = auto()
    REGION_DESCRIPTION = auto()


class SegmentationMode(Enum):
    REFERRING_EXPRESSION = auto()
    REGION = auto()


@dataclass
class Region:
    x1: int
    y1: int
    x2: int
    y2: int


class Florence2Client:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _encode_image(self, image: Path) -> Optional[str]:
        "Given Path to image, encode image to base64"
        assert image.exists(), f"Cannot find image {image}"
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        return None

    def _post_request(self, payload: Dict) -> Dict:
        print(f"Sending payload for task {payload['task']}")
        response = requests.post(f"{self.url}/run_task", json=payload)
        response.raise_for_status()
        return response.json()["result"]

    def caption(
        self, image: Path, verbosity: CaptionVerbosity = CaptionVerbosity.SIMPLE
    ):
        # TODO return type
        task_mapping = {
            CaptionVerbosity.SIMPLE: FlorenceTask.CAPTION,
            CaptionVerbosity.DETAILED: FlorenceTask.DETAILED_CAPTION,
            CaptionVerbosity.VERY_DETAILED: FlorenceTask.MORE_DETAILED_CAPTION,
        }
        image_base64 = self._encode_image(image)
        payload = {"task": task_mapping[verbosity].value, "image_base64": image_base64}
        result = self._post_request(payload)
        return result

    def object_detection(
        self,
        image: Path,
        mode: ObjectDetectionMode = ObjectDetectionMode.DEFAULT,
        prompt: Optional[str] = None,
        region: Optional[Region] = None,
    ):
        # TODO return type
        task_mapping = {
            ObjectDetectionMode.DEFAULT: FlorenceTask.OBJECT_DETECTION,
            ObjectDetectionMode.DENSE_CAPTION: FlorenceTask.DENSE_REGION_CAPTION,
            ObjectDetectionMode.REGION_PROPOSAL: FlorenceTask.REGION_PROPOSAL,
            ObjectDetectionMode.CAPTION_GROUNDING: FlorenceTask.CAPTION_TO_PHRASE_GROUNDING,
            ObjectDetectionMode.REGION_CATEGORY: FlorenceTask.REGION_TO_CATEGORY,
            ObjectDetectionMode.REGION_DESCRIPTION: FlorenceTask.REGION_TO_DESCRIPTION,
        }
        image_base64 = self._encode_image(image)
        task = task_mapping[mode]
        payload = {"task": task.value, "image_base64": image_base64}
        if task is FlorenceTask.CAPTION_TO_PHRASE_GROUNDING:
            assert prompt is not None, "Cannot use caption grounding without a prompt"
            payload["text_input"] = prompt
        elif task in [
            FlorenceTask.REGION_TO_CATEGORY,
            FlorenceTask.REGION_TO_DESCRIPTION,
        ]:
            assert (
                region is not None
            ), "Cannot use region tasks in object detection without providing a region"
            payload["text_input"] = (
                f"<loc_{region.x1}><loc_{region.y1}><loc_{region.x2}><loc_{region.y2}>"
            )
        result = self._post_request(payload)
        return result

    def segmentation(
        self,
        image: Path,
        mode=SegmentationMode,
        prompt: Optional[str] = None,
        region: Optional[Region] = None,
    ):
        task_mapping = {
            SegmentationMode.REFERRING_EXPRESSION: FlorenceTask.REFERRING_EXPRESSION_SEGMENTATION,
            SegmentationMode.REGION: FlorenceTask.REGION_TO_SEGMENTATION,
        }
        image_base64 = self._encode_image(image)
        task = task_mapping[mode]
        payload = {"task": task.value, "image_base64": image_base64}
        if task is FlorenceTask.REFERRING_EXPRESSION_SEGMENTATION:
            assert (
                prompt is not None
            ), "Cannot use referring expression without a prompt"
            payload["text_input"] = prompt
        elif task is FlorenceTask.REGION_TO_SEGMENTATION:
            assert (
                region is not None
            ), "Cannot use region task in segmentation without providing a region"
            payload["text_input"] = (
                f"<loc_{region.x1}><loc_{region.y1}><loc_{region.x2}><loc_{region.y2}>"
            )
        else:
            return None
        result = self._post_request(payload)
        return result

    def ocr(self, image: Path, region: Optional[Region] = None):
        image_base64 = self._encode_image(image)
        payload = {"image_base64": image_base64}
        task = FlorenceTask.OCR.value
        if region is not None:
            task = FlorenceTask.OCR_WITH_REGION.value
            payload["text_input"] = (
                f"<loc_{region.x1}><loc_{region.y1}><loc_{region.x2}><loc_{region.y2}>"
            )
        payload["task"] = task
        result = self._post_request(payload)
        return result
