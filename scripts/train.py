import argparse
import os

from corescore.models import CoreModel
from corescore.mlflowregistry import MlflowRegistry
import mlflow

def train(epochs=10, lr=0.00001, resize=8, path=os.getcwd()):
    mlflow.fastai.autolog()
    mlflow.set_tag('model', 'corescore')
    coremodel = CoreModel(path, epochs=epochs)
    unet_learn = coremodel.learner(resize=resize)
    coremodel.fit(lr=lr, learner=unet_learn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        default=10,
                        help="Epochs to train, default 10")
    parser.add_argument('--lr',
                        default=0.00001,
                        help="Learning rate to perform training")
    parser.add_argument('--resize',
                        default=8,
                        help="Scale image input down by this proportion")

    args = parser.parse_args()

    # Run the training loop
    train(epochs=int(args.epochs), lr=float(args.lr), resize=int(args.resize))

    # Register the model
    #MlflowRegistry().register_model("tags.model = 'corescore'",
    #                                name="corescore")
