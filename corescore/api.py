"""API accepting requests sent by LabelTool,
for use with its ML Assist mode in image annotation
This is what it should respond with

{
  "predictions": [
    {
       "det_boxes": [<ymin>, <xmin>, <ymax>, <xmax>],
       "det_class": <str>,
       "det_score": <0 ~ 1 floating number
    },
    ...,
    ...
    ]
}

"""

import base64
import io
from typing import List

from fastapi import FastAPI, Depends
from fastai.vision.image import Image as fastaiImage, pil2tensor
import numpy as np
from PIL import Image
from pydantic import BaseModel

from corescore.mlflowregistry import MlflowRegistry

MODEL_NAME = 'corescore'


def load_model():
    """Load latest version of our model from MLFlow registry"""
    return registry.load_model(MODEL_NAME)


app = FastAPI()
registry = MlflowRegistry()


# Define classes for what Label-tool accepts

class InputBytes(BaseModel):
    b64: str


class Instance(BaseModel):
    input_bytes: InputBytes


class Instances(BaseModel):
    instances: List[Instance]


@app.post("/labels")
async def core_labels(images: Instances, model=Depends(load_model)):
    labels = []
    for instance in images:
        labels.append(segment_image(instance, model))
    return {"masks": labels}


def segment_image(instance: Instance, model):
    image_bytes = base64.decodebytes(instance[1][0].input_bytes.b64.encode())
    image_arr = np.array(Image.open(io.BytesIO(image_bytes)))
    # predict
    # TODO return labelled regions that LabelTool hopes for
    image_arr = fastaiImage(pil2tensor(image_arr, dtype=np.uint8))
    prediction = model.predict(image_arr)

    return prediction
