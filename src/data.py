import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import torchvision.transforms as transforms


def FashionMNIST_train(batch_size, transformation) -> DataLoader:
    """
    More:
    -------------
    more attr in DataLoader to explore
    """
    # Loading FashionMNIST using torchvision.datasets
    training_data = datasets.FashionMNIST(
        root="./datasets",
        train=True,
        download=True,
        transform=transformation,
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    return train_dataloader


def FashionMNIST_test(batch_size, transformation) -> DataLoader:
    """
    More:
    -------------
    more attr in DataLoader to explore
    """
    test_data = datasets.FashionMNIST(
        root="./datasets",
        train=False,
        download=True,
        transform=transformation,
    )

    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return test_dataloader


def load_fashionmnist():

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Loading FashionMNIST using torchvision.datasets
    training_data = datasets.FashionMNIST(
        root="./datasets",
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.FashionMNIST(
        root="./datasets",
        train=False,
        download=True,
        transform=transform,
    )

    return training_data, test_data