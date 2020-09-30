import os
import argparse
import mlflow

from corescore.mlflowregistry import MlflowRegistry


if __name__ == '__main__':
    URI = os.environ.get('MLFLOW_TRACKING_URI', '')
    client = MlflowRegistry(URI)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, help="model's tag")
    parser.add_argument('--name', type=str, help="model name")
    parser.add_argument('--metric', type=str, help="model's metric")
    parser.add_argument('--load', action='store_true', help="load model")
    args = parser.parse_args()

    if args.tag and args.name and args.metric:
        client.register_model(query=f"tags.{args.tag} = '{args.name}'",
            metric=args.metric)
    elif args.name and args.load:
        client.load_model(name=args.name)
    else:
        print(client.list_experiments(query="tags.model = 'geo-ner-spacy'"))
   
