import os
from corescore.models import CoreModel


def test_create_model():
    model = CoreModel(os.getcwd())

