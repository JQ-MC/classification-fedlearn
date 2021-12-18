import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from src.models import BasicCNN, FullyConnected


def get_size_after_cnn(h: int, k: int, s: int, p: int):
    "calculates the size of the images after 2 convolutions"

    size = -(-(h - k + s + p)) / s
    size = -(-size / 2)
    size = -(-(size - k + s + p)) / s
    size = -(-size / 2)

    return size


def train_net(model, dataloader: DataLoader, epochs: int) -> None:
    "training of the models"
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    model.train()

    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(model.device), y.to(model.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        print()
        print(f"Epoch: {epoch+1} / {epochs}")


def test_net(model, dataloader):
    "testing of the models"
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(model.device), y.to(model.device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss, correct * 100


def load_model(model_str):
    "returns selected model"

    if model_str == "BasicCNN":
        return BasicCNN(in_chn=1, n_out=10)
    elif model_str == "FullyConnected":
        return FullyConnected(n_inp=1, n_out=10)
