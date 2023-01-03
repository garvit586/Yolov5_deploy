# Install requirements
# pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
# pip install python-multipart

# Import section
import io
import json
from PIL import Image
from fastapi import File, FastAPI
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/object_detection/Zypp_Yolo/best.pt', force_reload=True)

# create your API
app = FastAPI()


# Set up your API and integrate your ML model
@app.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    if len(results_json)==0:
        return  {"result" : "Not Detected"}
    return {"result": "Detected"}