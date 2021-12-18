"""Minimal example on how to start a simple Flower server."""


import argparse
from collections import OrderedDict
from os import truncate
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from src.data import load_fashionmnist
from src.utils import test_net, load_model
from src.strategy import FedAvg_c
from src.server import Server

import pickle

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--server_address",
    type=str,
    required=False,
    default="[::]:8080",
    help=f"gRPC server address",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=1,
    help="Number of rounds of federated learning (default: 1)",
)
parser.add_argument(
    "--load_params",
    dest="load_params",
    action="store_true",
    help="Load existing parameters for selected model",
)
parser.add_argument(
    "--no-load_params",
    dest="load_params",
    action="store_false",
    help="Load existing parameters for selected model",
)
parser.set_defaults(load_params=True)
parser.add_argument(
    "--save_file",
    type=str,
    default=None,
    help="File name to store the parameters in src/params/",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_sample_size",
    type=int,
    default=2,
    help="Minimum number of clients used for fit/evaluate (default: 2)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--log_host",
    type=str,
    help="Logserver address (no default)",
)
parser.add_argument(
    "--model",
    type=str,
    default="BasicCNN",
    choices=["FullyConnected", "BasicCNN"],
    help="model to train",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="training batch size",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of workers for dataset reading",
)
parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()


def main() -> None:
    """Start server and train five rounds."""

    print(args)

    assert (
        args.min_sample_size <= args.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Load evaluation data
    # _, testset = utils.load_cifar(download=True)
    _, testset = load_fashionmnist()

    # Load global parameters, if chosen or exist
    params, args.save_file = prepare_server(args)

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()

    strategy = FedAvg_c(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
        initial_parameters=params,
    )
    server = Server(
        client_manager=client_manager, strategy=strategy, save_file=args.save_file
    )

    # Run server
    fl.server.start_server(
        args.server_address, server, config={"num_rounds": args.rounds}
    )


def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(args.batch_size),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
    }
    return config


def set_weights(model, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the passed testset for evaluation."""

        model = load_model(args.model)
        set_weights(model, weights)

        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = test_net(model, testloader)
        return loss, {"accuracy": accuracy}

    return evaluate


def prepare_server(args) -> Tuple:
    "returns the model parameters and saves un args the file to store them after the global fit"

    m = {"BasicCNN": "fashion_basic_cnn.pk", "FullyConnected": "fashion_fully_conn.pk"}

    if args.save_file is None:
        args.save_file = m[args.model]

        if args.load_params is True:
            with open("src/params/" + args.save_file, "rb") as f:
                params = pickle.load(f)
        else:
            params = None

    else:
        if args.load_params is True:
            try:
                with open("src/params/" + args.save_file, "rb") as f:
                    params = pickle.load(f)
            except FileNotFoundError:
                print(
                    "There is no file named",
                    args.save_file,
                    "to get the parameters from",
                )
                print("Maybe you wanted the argument --no-load_params")

        else:
            params = None

    return params, args.save_file


if __name__ == "__main__":
    main()