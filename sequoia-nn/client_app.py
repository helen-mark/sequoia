"""soup: A Flower / sklearn app."""

import warnings

from flwr.client import ClientApp, NumPyClient, Client
from flwr.common import Context
from .task import (
    load_model,
    load_data
)


class FlowerClient(NumPyClient):
    def __init__(self, model, train_ds, val_ds, epochs, batch_size, verbose):
        self.model = model
        self.train_ds, self.val_ds = train_ds, val_ds
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=self.val_ds
        )
        return self.model.get_weights(), len(self.train_ds), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc, recall, precision = self.model.evaluate(self.val_ds, verbose=0)
        print(loss, acc, recall, precision)
        return loss, len(self.val_ds), {"acc": acc}



def client_fn(context: Context):
    # Load model and data
    net = load_model()

    train_ds, val_ds = load_data(0, 0)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]

    # Return Client instance
    return FlowerClient(net, train_ds, val_ds, epochs, batch_size, True).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
