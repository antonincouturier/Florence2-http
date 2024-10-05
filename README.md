# Florence2 HTTP

 Web service that exposes Microsoft’s Florence-2 model behind an HTTP server

## Install 

## Run server 

## Run client 

## Florence2 supported tasks and required inputs 

| Task                            | Description                                                                                                                                                            | Required Inputs                          | API Usage              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|------------------------|
| Caption                         | Generate a simple caption for the input image.                                                                                                                         | Image                                    | `<CAPTION>`            |
| Detailed Caption                | Generate a detailed caption the input image.                                                                                                                           | Image                                    | `<DETAILED_CAPTION>`    |
| More Detailed Caption           | Generate a very detailed caption for the image,                                                                                                                        | Image                                    | `<MORE_DETAILED_CAPTION>`|
| Object Detection (OD)           | Generate bounding boxes and labels for each object in the input image.                                                                                                 | Image                                    | `<OD>`                 |
| Dense Region Caption            | Generate bounding boxes and dense captions for each object in the input image.                                                                                         | Image                                    | `<DENSE_REGION_CAPTION>`|
| Region Proposal                 | Generate bounding boxes with no labels for each object in the input image.                                                                                             | Image                                    | `<REGION_PROPOSAL>`     |
| Caption to Phrase Grounding     | Generate bounding boxes and labels for specific parts described in input caption for the input image.                                                                  | Image, Text                              | `<CAPTION_TO_PHRASE_GROUNDING>` |
| Referring Expression Segmentation | Generate segmentation masks based of input referring expression for input image.                                                                                     | Image, Text                              | `<REFERRING_EXPRESSION_SEGMENTATION>` |
| Region to Segmentation          | Generate segmentation mask for specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                     | Image, Coordinates                       | `<REGION_TO_SEGMENTATION>`|
| Region to Category              | Generate label of object in specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                        | Image, Coordinates                       | `<REGION_TO_CATEGORY>`  |
| Region to Description           | Generate description of object in specific region of input image based on quantized input coordinates ([x1, y1, x2, y2] in [0, 999]).                                  | Image, Coordinates                       | `<REGION_TO_DESCRIPTION>`|
| OCR                             | Extract text in the input image                                                                                                                                        | Image                                    | `<OCR>`                 |
| OCR with Region                 | Generate bounding boxes for regions of text in the input image and extract text.                                                                                       | Image                                    | `<OCR_WITH_REGION>`     |

# TODO also include open vocabulary detection

