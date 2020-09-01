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

from fastapi import FastAPI
import numpy as np
from PIL import Image
from pydantic import BaseModel


# By default, models from corebreakout's assets.zip
def load_model():
    """Load the latest (best scoring) UNet from MLFlow registry"""
    pass


app = FastAPI()


# Define classes for what Label-tool accepts

class InputBytes(BaseModel):
    b64: str


class Instance(BaseModel):
    input_bytes: InputBytes


class Instances(BaseModel):
    instances: List[Instance]


@app.post("/labels")
def core_labels(images: Instances):
    labels = []
    for instance in images:
        labels.append(segment_image(instance))
    return {"masks": labels}


def segment_image(instance: Instance, model=load_model()):
    image_bytes = base64.decodebytes(instance[1][0].input_bytes.b64.encode())
    image_arr = np.array(Image.open(io.BytesIO(image_bytes)))
    # predict
    # TODO return labelled regions that LabelTool hopes for
    return {}
