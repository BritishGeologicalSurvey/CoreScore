import argparse
import os

from corescore.models import CoreModel
from corescore.mlflowregistry import MlflowRegistry
import mlflow

def train(epochs=10, lr=0.00001, path=os.getcwd()):
    mlflow.fastai.autolog()
    coremodel = CoreModel(path, epochs=epochs)
    unet_learn = coremodel.learner()
    coremodel.fit(lr=lr, learner=unet_learn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        default=10,
                        help="Epochs to train, default 10")
    parser.add_argument('--lr',
                        default=0.00001
                        help="Learning rate to perform training")
    args = parser.parse_args()

    # Run the training loop
    train(epochs=int(args.epochs), lr=float(args.lr))

    # Register the model
    MlflowRegistry().register_model("tags.model = 'corescore'",
                                    name="corescore")
