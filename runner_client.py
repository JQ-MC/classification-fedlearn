import argparse
import flwr as fl

from src.client import FlClient
from src.utils import load_model


def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        required=False,
        default="[::]:8080",
        help=f"gRPC server address",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where the dataset lives",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BasicCNN",
        choices=["FullyConnected", "BasicCNN"],
        help="model to train",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")

    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # model
    model = load_model(args.model)

    # Start client
    client = FlClient(model, args)
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()