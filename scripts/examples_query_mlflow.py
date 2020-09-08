import os

import argparse

from corescore.mlflowregistry import MlflowRegistry

def list_models(client):
    for rm in client.list_registered_models():
        print(rm)

def register_model(client, tag, search_str):
    client.register_latest(tag, search_str)

if __name__ == '__main__':
    URI = os.environ.get('MLFLOW_TRACKING_URI', '')
    client = MlflowRegistry(URI)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, help="model tag")
    parser.add_argument('--name', type=str, help="model name")
    args = parser.parse_args()
    
    if args.tag and args.name:
        register_model(client, tag=args.tag, search_str=args.name)
    else:
        list_models()

