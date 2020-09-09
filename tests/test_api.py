"""
API expects an input like this
{
  "instances": [
    {
      "input_bytes": {
        "b64": "<str>"
      }
    }
  ]
}
"""
import os
from base64 import b64encode
import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from corescore.api import app, load_model

client = TestClient(app)

class MockModel(MagicMock):
    # Could set a return_value of predict() here
    pass

async def load_test_model():
    return MockModel()

app.dependency_overrides[load_model] = load_test_model


@pytest.fixture
def image_bytes():
    sample = 'S00128906.Cropped_Top_2.png'
    fix_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(fix_dir,
                           'fixtures',
                           'images',
                           'train',
                           sample), 'rb') as img_file:
        return b64encode(img_file.read()).decode()


def test_labels(image_bytes):
    body = {'instances': [
            {'input_bytes': {
                'b64': image_bytes}}]}

    response = client.post("/labels", json=body)

    assert response.status_code == 200
    assert response.json()["masks"]
