"""API accepting requests sent by LabelTool,
for use with its ML Assist mode in image annotation

This shows the response formats for different image segmentation
https://github.com/Slava/label-tool/blob/master/client/src/admin/MLAssist.js
"""

import base64
import io
from typing import List

from fastapi import FastAPI, Depends, HTTPException
from fastai.vision.image import Image
from pydantic import BaseModel
from skimage.io import imread
from torchvision.transforms import ToTensor
from corescore.mlflowregistry import MlflowRegistry, MlflowRegistryError

MODEL_NAME = 'corescore'


def load_model():
    """Load latest version of our model from MLFlow registry"""
    try:
        model = registry.load_model(MODEL_NAME)
    except MlflowRegistryError:  # handle better
        raise HTTPException(status_code=404, detail="Model not found")
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
    """Return a set of masks for input images
    Accepts JSON with base64 encoded images as per LabelTool MLAssist
    """
    labels = []
    for instance in images:
        labels.append(segment_image(instance, model))
    return {"predictions": labels}


def segment_image(instance: Instance, model):
    """Decode a base64 encoded image return Unet predictions for labels"""
    image_bytes = base64.decodebytes(instance[1][0].input_bytes.b64.encode())
    image_arr = imread(io.BytesIO(image_bytes))

    image_arr = Image(ToTensor()(image_arr))

    # The mask prediction will be the grayscale 2d array
    # LabelTool wants a list of {raw_image: []}
    _, mask, _ = model.predict(image_arr)
    mask_arr = mask.numpy()[0].astype('uint8').tolist()
    return {'raw_image': mask_arr}
