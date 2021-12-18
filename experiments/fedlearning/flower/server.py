import flwr as fl


if __name__ == "__main__":

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 3},
    )