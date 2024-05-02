"""Implimentation of LeNet Network.
Note the original implimentation used sigmoid instead of ReLU,
and Average Pooling instead of Max Pooling."""

from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(out_features=120),
            nn.ReLU(),
            nn.LazyLinear(out_features=84),
            nn.ReLU(),
            nn.LazyLinear(out_features=10),
        )

    def forward(self, x):
        return self.net(x)
