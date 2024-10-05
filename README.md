# Florence2 HTTP

 Web service that exposes Microsoft’s Florence-2 model behind an HTTP server

## Install 
```bash
pip install .
```

## Run server 

```bash
uvicorn florence2_http.server.main:app --reload
```

## Run client 
```python
from pathlib import Path

from florence2_http.client import Florence2Client

client = Florence2Client("http://127.0.0.1:8000")
image_path = Path("/path/to/image.png")
caption = client.caption(image=image_path)
```

## Florence2 supported tasks and required inputs 

| Task                            | Description                                                                                                                                                            | Required Inputs                          | API Usage              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|------------------------|
| Caption                         | Generate a simple caption for the input image.                                                                                                                         | Image                                    | client.caption(image_path)           |
| Detailed Caption                | Generate a detailed caption the input image.                                                                                                                           | Image                                    | client.caption(image_path, verbosity=CaptionVerbosity.DETAILED)   |
| More Detailed Caption           | Generate a very detailed caption for the image,                                                                                                                        | Image                                    | client.caption(iamge_path, verbosity=CaptionVerbosity.VERY_DETAILED)|
| Object Detection (OD)           | Generate bounding boxes and labels for each object in the input image.                                                                                                 | Image                                    | client.object_detection(image_path)                |
| Dense Region Caption            | Generate bounding boxes and dense captions for each object in the input image.                                                                                         | Image                                    | client.object_detection(image_path, mode=ObjectDetectionMode.DENSE_CAPTION) |
| Region Proposal                 | Generate bounding boxes with no labels for each object in the input image.                                                                                             | Image                                    | client.object_detection(image_path, mode=ObjectDetectionMode.REGION_PROPOSAL)    |
| Caption to Phrase Grounding     | Generate bounding boxes and labels for specific parts described in input caption for the input image.                                                                  | Image, Text                              | client.object_detection(image_path, mode=ObjectDetectionMode.CAPTION_GROUNDING, prompt=prompt)|
| Referring Expression Segmentation | Generate segmentation masks based of input referring expression for input image.                                                                                     | Image, Text                              | client.segmentation(image_path, prompt=prompt)|
| Region to Segmentation          | Generate segmentation mask for specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                     | Image, Coordinates                       | client.segmentation(image_path, mode=SegmentationMode.REGION, region=Region(x1, y1, x2, y2)) |
| Region to Category              | Generate label of object in specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                        | Image, Coordinates                       | client.object_detection(image_path, mode=ObjectDetectionMode.REGION_CATEGORY, region=Region(x1, y1, x2, y2))  |
| Region to Description           | Generate description of object in specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                  | Image, Coordinates                       | client.object_detection(image_path, mode=ObjectDetectionMode.REGION_DESCRIPTION, region=Region(x1, y1, x2, y2))|
| OCR                             | Extract text in the input image                                                                                                                                        | Image                                    | client.ocr(image_path)                |
| OCR with Region                 | Generate bounding boxes for regions of text in the input image and extract text.                                                                                       | Image                                    | client.ocr(image_path, region=Region(x1, y1, x2, y2))    |

# TODO also include open vocabulary detection

