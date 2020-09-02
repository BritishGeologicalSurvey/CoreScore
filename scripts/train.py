import argparse
import os

from corescore.models import CoreModel
import mlflow


def train(epochs=10, lr=0.00001, path=os.getcwd()):
    coremodel = CoreModel(path, epochs=epochs)
    coremodel.fit(lr=lr)
    coremodel.save()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help="Epochs to train, default 10")
    parser.add_argument('--lr', help="Learning rate to perform training")
    args = parser.parse_args()
    train(epochs=int(args.epochs), lr=float(args.lr))

