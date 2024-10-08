# Florence2 HTTP

 Web service that exposes Microsoft’s Florence-2 model (via HuggingFace) behind an HTTP server (via FastAPI, uvicorn, pydantic models).

## Components 

- **Client Library (`client`):** Provides a `Florence2Client` class with methods corresponding to different tasks. Handles image encoding, request formation, and response parsing.
- **Server (`server`):** Hosts the FastAPI application, defines API endpoints, and handles requests by invoking the Florence-2 model.
- **Models (`server/models.py`):** Contains the `Florence2` class that wraps model loading and inference logic.
- **Schemas (`server/schemas.py`):** Defines Pydantic models for request and response data structures.
- **Shared Definitions (`shared.py`):** Contains enumerations for tasks and model types, ensuring consistency between client and server.

## Install 
```bash
pip install .
```

## Run server 

```bash
uvicorn florence2_http.server.main:app --reload
```

By default, the server uses the `Florence-2-base` model. To change the model edit `server/main.py` to use e.g `FlorenceModel.LARGE`.


## Run client 

### Captioning 

```python
from pathlib import Path

from florence2_http.client import Florence2Client, CaptionVerbosity

client = Florence2Client("http://127.0.0.1:8000")
image_path = Path("data/car.jpg")
# Caption
caption = client.caption(image=image_path)
print(caption)
# Detailed Caption
caption = client.caption(image=image_path, verbosity=CaptionVerbosity.DETAILED)
print(caption)
# More Detailed Caption
caption = client.caption(image=image_path, verbosity=CaptionVerbosity.VERY_DETAILED)
print(caption)
```

### Object detection 
```python
from pathlib import Path

from florence2_http.client import Florence2Client, ObjectDetectionMode, Region

client = Florence2Client("http://127.0.0.1:8000")
image_path = Path("data/car.jpg")
# Object Detection 
result = client.object_detection(image=image_path)
print(result)
# Dense Region Caption
result = client.object_detection(image=image_path, mode=ObjectDetectionMode.DENSE_CAPTION)
print(result)
# Region Proposal
result = client.object_detection(image=image_path, mode=ObjectDetectionMode.REGION_PROPOSAL)
print(result)
# Caption to Phrase Grounding
prompt = "A green car parked in front of a yellow building."
result = client.object_detection(image=image_path, mode=ObjectDetectionMode.CAPTION_GROUNDING, prompt=prompt)
print(result)
# Region to Category
region = Region(x1=52, y1=332, x2=932, y2=774)
result = client.object_detection(image=image_path, mode=ObjectDetectionMode.REGION_CATEGORY, region=region)
print(result)
# Region to Description
result = client.object_detection(image=image_path, mode=ObjectDetectionMode.REGION_DESCRIPTION, region=region)
print(result)
# Open vocabulary detection 
prompt = "a green car"
result = client.object_detection(image=image_path, mode=ObjectDetectionMode.OPEN_VOCABULARY, prompt=prompt)
print(result)
```

### Segmentation 
```python
from pathlib import Path

from florence2_http.client import Florence2Client, SegmentationMode, Region

client = Florence2Client("http://127.0.0.1:8000")
image_path = Path("data/car.jpg")
# Referring Expression Segmentation
prompt = "a green car"
result = client.segmentation(image=image_path, mode=SegmentationMode.REFERRING_EXPRESSION, prompt=prompt)
print(result)
# Region to Segmentation
region = Region(x1=702, y1=575, x2=866, y2=772)
result = client.segmentation(image=image_path, mode=SegmentationMode.REGION, region=region)
print(result)
```

### OCR
```python
from pathlib import Path

from florence2_http.client import Florence2Client

client = Florence2Client("http://127.0.0.1:8000")
image_path = Path("data/cuda_book.jpg")
# OCR
result = client.ocr(image=image_path)
print(result)
# OCR with Region
result = client.ocr(image=image_path, find_bbox=True)
print(result)
```

## Florence2 supported tasks and required inputs 

| Task                            | Description                                                                                                                                                            | Required Inputs                          | API Usage              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|------------------------|
| Caption                         | Generate a simple caption for the input image.                                                                                                                         | Image                                    | client.caption(image_path)           |
| Detailed Caption                | Generate a detailed caption the input image.                                                                                                                           | Image                                    | client.caption(image_path, verbosity=CaptionVerbosity.DETAILED)   |
| More Detailed Caption           | Generate a very detailed caption for the image,                                                                                                                        | Image                                    | client.caption(iamge_path, verbosity=CaptionVerbosity.VERY_DETAILED)|
| Object Detection            | Generate bounding boxes and labels for each object in the input image.                                                                                                 | Image                                    | client.object_detection(image_path)                |
| Dense Region Caption            | Generate bounding boxes and dense captions for each object in the input image.                                                                                         | Image                                    | client.object_detection(image_path, mode=ObjectDetectionMode.DENSE_CAPTION) |
| Region Proposal                 | Generate bounding boxes with no labels for each object in the input image.                                                                                             | Image                                    | client.object_detection(image_path, mode=ObjectDetectionMode.REGION_PROPOSAL)    |
| Caption to Phrase Grounding     | Generate bounding boxes and labels for specific parts described in input caption for the input image.                                                                  | Image, Text                              | client.object_detection(image_path, mode=ObjectDetectionMode.CAPTION_GROUNDING, prompt=prompt)|
| Referring Expression Segmentation | Generate segmentation masks based of input referring expression for input image.                                                                                     | Image, Text                              | client.segmentation(image_path, mode=SegmentationMode.REFERRING_EXPRESSION, prompt=prompt)|
| Region to Segmentation          | Generate segmentation mask for specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                     | Image, Coordinates                       | client.segmentation(image_path, mode=SegmentationMode.REGION, region=Region(x1, y1, x2, y2)) |
| Open vocabulary detection          | Generate boudning boxes and labels based on input prompt for input image                                     | Image, Text                       | client.object_detection(image_path, mode=ObjectDetectionMode.OPEN_VOCABULARY, prompt=prompt) |
| Region to Category              | Generate label of object in specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                        | Image, Coordinates                       | client.object_detection(image_path, mode=ObjectDetectionMode.REGION_CATEGORY, region=Region(x1, y1, x2, y2))  |
| Region to Description           | Generate description of object in specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                  | Image, Coordinates                       | client.object_detection(image_path, mode=ObjectDetectionMode.REGION_DESCRIPTION, region=Region(x1, y1, x2, y2))|
| OCR                             | Extract text in the input image                                                                                                                                        | Image                                    | client.ocr(image_path)                |
| OCR with Region                 | Generate bounding boxes for regions of text in the input image and extract text.                                                                                       | Image                                    | client.ocr(image_path, find_bbox=True)    |

