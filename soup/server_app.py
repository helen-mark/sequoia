"""soup: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from task import get_model, set_initial_params, get_model_params
import flwr as fl
import toml
import numpy as np

with open('pyproject.toml') as config_file:
    config = toml.load(config_file)
    print(config)
    tool_flwr_app_conf = config['tool']['flwr']['app']['config']
    num_rounds = tool_flwr_app_conf["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = tool_flwr_app_conf["penalty"]
    local_epochs = tool_flwr_app_conf["local-epochs"]
    model = get_model(penalty, local_epochs)
    print(model)
    set_initial_params(model)
    initial_parameters = get_model_params(model)
    print("Init parameters", initial_parameters)

    # Define strategy
    strategy = FedAvg(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    def server_fn(context: Context):
        return ServerAppComponents(strategy=strategy, config=config)
    app = ServerApp(server_fn=server_fn)

    fl.server.start_server(server_address="192.168.2.31:50057", config=config, strategy=strategy)
