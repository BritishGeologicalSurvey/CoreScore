"""This is the same call as in the unit tests fori the API
It should show how to POST an image and get back a mask 
using the same interface that LabelTool expects
"""
import os
import requests
from base64 import b64encode
from corescore.api import app, load_model

API = 'http://localhost:5001'

def image_bytes():
    sample = 'S00128906.Cropped_Top_2.png'
    fix_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(fix_dir,
                           '../tests/fixtures',
                           'images',
                           'train',
                           sample), 'rb') as img_file:
        return b64encode(img_file.read()).decode()


def labels():
    body = {'instances': [
            {'input_bytes': {
                'b64': image_bytes()}}]}

    response = requests.post(f"{API}/labels", json=body)
    print(response.content)

if __name__ == '__main__':
    labels()
