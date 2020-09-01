import argparse
import os
from corescore.models import CoreModel

def train(epochs=10, path=os.getcwd()):
    coremodel = CoreModel(path, epochs=epochs)
    coremodel.fit()
    coremodel.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help="Epochs to train, default 10")
    args = parser.parse_args()
    train(epochs=args.epochs)

