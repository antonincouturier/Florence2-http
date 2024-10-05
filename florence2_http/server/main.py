from fastapi import FastAPI

from florence2_http.server.models import Florence2
from florence2_http.server.schemas import TaskRequest, TaskResponse
from florence2_http.shared import FlorenceModel

app = FastAPI()

model_type = FlorenceModel.BASE
model = Florence2(model_type)


@app.post("/run_task", response_model=TaskResponse)
async def run_task(request: TaskRequest):
    result = model.run_task(
        task=request.task,
        image_base64=request.image_base64,
        text_input=request.text_input,
    )
    return TaskResponse(result=result)
