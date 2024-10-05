import base64
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Union

import requests

from florence2_http.shared import FlorenceTask


class CaptionVerbosity(Enum):
    """ Supported levels of verbosity for captioning tasks"""
    SIMPLE = auto()
    DETAILED = auto()
    VERY_DETAILED = auto()


class ObjectDetectionMode(Enum):
    """ Supported object detection subtasks """
    DEFAULT = auto()
    DENSE_CAPTION = auto()
    REGION_PROPOSAL = auto()
    CAPTION_GROUNDING = auto()
    REGION_CATEGORY = auto()
    REGION_DESCRIPTION = auto()


class SegmentationMode(Enum):
    """ Supported segmentation subtasks """
    REFERRING_EXPRESSION = auto()
    REGION = auto()


@dataclass
class Region:
    """ Data class for regions """
    x1: int
    y1: int
    x2: int
    y2: int


class Florence2Client:
    """
    Main entry point for interacting with the Florence 2 HTTP server.

    This client provides methods for captioning, object detection, segmentation, 
    and optical character recognition on images. The images are encoded 
    in base64 format and sent to the server for processing

    Parameters
    ----------
    url : str
        Url of HTTP server
    """
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _encode_image(self, image: Path) -> Optional[str]:
        """
        Encode an image from a given file path to a base64-encoded UTF-8 string

        Parameters
        ----------
        image : Path
            Path to image

        Returns
        -------
        Optional[str]
            The base64-encoded image as a UTF-8 string, or None if there is an issue during encoding

        Raises
        ------
        AssertionError
            If the image does not exist
        """
        assert image.exists(), f"Cannot find image {image}"
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        return None

    def _post_request(self, payload: Dict) -> Dict:
        """
        Send a POST request to HTTP server with the given payload

        Parameters
        ----------
        payload : Dict
            The payload to be sent to the API in JSON format

        Returns
        -------
        Dict
            JSON response from the API

        Raises
        ------
        requests.HTTPError
            If the API request fails
        """
        response = requests.post(f"{self.url}/run_task", json=payload)
        response.raise_for_status()
        return response.json()["result"]

    def caption(
        self, image: Path, verbosity: CaptionVerbosity = CaptionVerbosity.SIMPLE
    ) -> str:
        """
        Generate a caption for an image based on the specified verbosity level

        Parameters
        ----------
        image : Path
            Path to image
        verbosity : CaptionVerbosity, optional
            The level of verbosity for the caption. Default: `CaptionVerbosity.SIMPLE`

        Returns
        -------
        str
            Generated caption for the image

        Raises
        ------
        AssertionError
            If the image does not exist
        requests.HTTPError
            If the API request fails
        """
        task_mapping = {
            CaptionVerbosity.SIMPLE: FlorenceTask.CAPTION,
            CaptionVerbosity.DETAILED: FlorenceTask.DETAILED_CAPTION,
            CaptionVerbosity.VERY_DETAILED: FlorenceTask.MORE_DETAILED_CAPTION,
        }
        task = task_mapping[verbosity].value
        image_base64 = self._encode_image(image)
        payload = {"task": task, "image_base64": image_base64}
        result = self._post_request(payload)
        return result[task]

    def object_detection(
        self,
        image: Path,
        mode: ObjectDetectionMode = ObjectDetectionMode.DEFAULT,
        prompt: Optional[str] = None,
        region: Optional[Region] = None,
    ) -> Union[Dict, str]:
        """
        Perform object detection on an image with an optional prompt or region

        Parameters
        ----------
        image : Path
            Path to image
        mode : ObjectDetectionMode, optional
            The mode of object detection. Default: `ObjectDetectionMode.DEFAULT`
        prompt : Optional[str], optional
            Prompt used for grounding sub-tasks (necessary if mode is `ObjectDetectionMode.CAPTION_GROUNDING`)
        region : Optional[Region], optional
            Region used for region-based sub-tasks (necessary if mode is `ObjectDetectionMode.REGION_CATEGORY` or `ObjectDetectionMode.REGION_DESCRIPTION`)

        Returns
        -------
        Union[Dict, str]
            Return dictionary with bounding boxes and labels
            Return string with description if mode is  `ObjectDetectionMode.REGION_CATEGORY` or `ObjectDetectionMode.REGION_DESCRIPTION`

        Raises
        ------
        AssertionError
            If the image does not exist
        requests.HTTPError
            If the API request fails
        """
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
        return result[task.value]

    def segmentation(
        self,
        image: Path,
        mode: SegmentationMode,
        prompt: Optional[str] = None,
        region: Optional[Region] = None,
    ) -> Dict:
        """
        Perform segmentation on an image, either by referring expression or region

        Parameters
        ----------
        image : Path
            Path to image
        mode : SegmentationMode
            The mode of segmentation
        prompt : Optional[str], optional
            Prompt used for referring expression (necessary if mode is `SegmentationMode.REFERRING_EXPRESSION`)
        region : Optional[Region], optional
            Region used for region-based sub-tasks (necessary if mode is `SegmentationMode.REGION`)

        Returns
        -------
        Dict
            Return dictionary with polygons and labels

        Raises
        ------
        AssertionError
            If the image does not exist
        requests.HTTPError
            If the API request fails
        """
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
        return result[task.value]

    def ocr(self, image: Path, find_bbox: bool = False) -> Union[Dict, str]:
        """
        Perform Optical Character Recognition on an image.

        Parameters
        ----------
        image : Path
            Path to image
        find_bbox : bool, optional
            If True, the OCR will return bounding boxes and detected text. Default: `False`.

        Returns
        -------
        Union[Dict, str]
            If find_bbox is False, returns the extracted text. If find_bbox is True, return a dictionary
            with bounding boxes and extracted text

        Raises
        ------
        AssertionError
            If the image does not exist
        requests.HTTPError
            If the API request fails
        """
        image_base64 = self._encode_image(image)
        task = FlorenceTask.OCR.value
        if find_bbox:
            task = FlorenceTask.OCR_WITH_REGION.value
        payload = {"image_base64": image_base64, "task": task}
        result = self._post_request(payload)
        return result[task]
