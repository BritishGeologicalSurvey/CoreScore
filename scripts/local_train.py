import argparse
import os
from corescore.models import CoreModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

test = os.getcwd()

def train(epochs=2, lr=0.00001, resize=8, batch_size=1, path=os.getcwd()):
    #    mlflow.fastai.autolog()
    #    mlflow.set_tag('model', 'corescore')

    # test = os.getcwd()
    coremodel = CoreModel(path, epochs=epochs, batch_size=batch_size)
    unet_learn = coremodel.learner(resize=resize)
    coremodel.fit(lr=lr, learner=unet_learn)
    unet_learn.save('tmp')


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
    parser.add_argument('--batch_size',
                        default=1,
                        help="Set the batch size")

    args = parser.parse_args()

    # Run the training loop
    train(epochs=int(args.epochs),
          lr=float(args.lr),
          resize=int(args.resize),
          batch_size=int(args.batch_size))

    # Register the model
    # Picks up MLFLOW_TRACKING_URI from environment.
    #    MlflowRegistry().register_model("tags.model = 'corescore'",
    #                                    name="corescore")
    
    # Long sleep to ensure model version is created
    # sleep(300)
