"""API accepting requests sent by LabelTool,
for use with its ML Assist mode in image annotation

This shows the response formats for different image segmentation
https://github.com/Slava/label-tool/blob/master/client/src/admin/MLAssist.js
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
    try:
        model = registry.load_model(MODEL_NAME)
    except Exception as err:  # handle better
        raise err  # will fastapi be graceful
    return model


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
    return {"predictions": [labels]}


def segment_image(instance: Instance, model):
    image_bytes = base64.decodebytes(instance[1][0].input_bytes.b64.encode())
    image_arr = np.array(Image.open(io.BytesIO(image_bytes)))

    image_arr = fastaiImage(pil2tensor(image_arr, dtype=np.uint8))

    # The mask prediction will be the grayscale 2d array
    # LabelTool wants a list of {raw_image: []}
    _, mask, _ = model.predict(image_arr)
    return mask
