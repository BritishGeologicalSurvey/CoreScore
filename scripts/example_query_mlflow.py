import os
import argparse
import mlflow

from corescore.mlflowregistry import MlflowRegistry


def list_models(client):
    for rm in client.list_registered_models():
        print(rm)


def load_latest(client, name):
    models = client.list_registered_models()
    latest = list(filter(lambda model: model.name == name, models))[0]
    model_path = os.path.join(latest.latest_versions[0].source, 'model')
    print(model_path)
    model = mlflow.fastai.load_model(model_path)


def register_model(client, tag, search_str):
    client.register_latest(tag, search_str)


if __name__ == '__main__':
    URI = os.environ.get('MLFLOW_TRACKING_URI', '')
    client = MlflowRegistry(URI)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, help="model tag")
    parser.add_argument('--name', type=str, help="model name")
    parser.add_argument('--load', action='store_true', help="load model")
    args = parser.parse_args()

    if args.tag and args.name:
        register_model(client, tag=args.tag, search_str=args.name)
    elif args.name and args.load:
        load_latest(client, name=args.name)
    else:
        list_models(client)
