# Federated Learning: Image Classification

Repository of the conceptual work around Federated Learning to train image classification models. 

Here you can find the developed code, documentation and concept experiments.

### venv

In order to activate `venv` environment, execute the following command from repository root:
```
source venv/bin/activate
```

### Repository folder structure:
- `.vscode`: vscode configuration.
- `documentation`: some documentation regarding DL and vision models.
- `experiments`: conceptual experiments regarding FL, Pytorch and more.
- `src`: source code.
    - `client`: implements the logic of the client in FL
    - `data`: functions to retrieve training data (Fashion MNIST)
    - `models`: classification models to be trained.
    - `server`: implements the logic of the server in FL
    - `strategy`: implements the strategies that guide the server and the parameters aggregation
    - `utils`: custom helper functions
- `runner_client.py`: sets up a client to train the net
- `runner_server.py`: sets up the server to handle the clients


### SetUp

#### Server

In order to start a Federated Learning training, it is necessary to start the server:

`python3 runner_server.py --server_address [::]:8080 --rounds 2 --loads_params --model BasicCNN`

```
- server_address: ip of the server
- rounds: number of federated rounds
- loads_params: controls if the server provides the initial parameters or it gets them from a random client (`--no_load_params`)
- save_file: path where the initials parameters are and where to store them.
- model: the model to be trained. FullyConnected or BasicCNN
```


#### Client

Clients must be created after the server. There must be at least `--min_num_clients` clients, a parameter from the server, to start the training. Defaults to 2.

`python3 runner_client.py --server_address [::]:8080 --cid 1 --epochs 100 --model BasicCNN --epochs 50`

```
- server_address: ip of the server. if executed in the same machine keep default. If not, check ip address of the server: XXX.XX.X.XX:8080
- cid: client id
- model: the model to be trained. FullyConnected or BasicCNN
- epochs: number of epochs per round
```


### Future work
