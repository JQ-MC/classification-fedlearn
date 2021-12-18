import torch
from torch import nn
from torch.nn.modules.container import Sequential
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from typing import Optional


class FullyConnected(nn.Module):
    def __init__(self, n_inp: int, n_out: int, device: Optional[str] = None):
        """
        Parameters:
        ---------------
        n_inp: int. h * w of the images shape
        n_out: int. number of labels to classify
        device: str
        """
        super(FullyConnected, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inp, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_out),
        )

        if device != None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class BasicCNN(nn.Module):
    def __init__(self, in_chn: int, n_out: int, device: Optional[int] = None):
        """
        Parameters:
        ---------------
        in_chnn: input channels of the image (e.g. RGB=3)
        n_out: int. number of labels to classify
        device: str
        """
        super(BasicCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_chn, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.connected_ly = Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_out),
        )

        if device != None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.connected_ly(x)
        return x
