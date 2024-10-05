
from pydantic import BaseModel, Field
from typing import Optional, Dict

from florence2_http.shared import FlorenceTask, Region

class TaskRequest(BaseModel):
    task: FlorenceTask = Field(..., description="Task to perform")
    image_base64: str = Field(..., description="Base64-encoded image data")
    text_input: Optional[str] = Field(None, description="Additional text input")
    
class TaskResponse(BaseModel):
    result: Dict = Field(..., description="Result from model")"
