"""Linear ReLU network with dropout layers."""

from torch import nn


class dropout_relu(nn.Module):
    def __init__(self, layer1=0.25, layer2=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(p=layer1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=layer2),
            nn.Linear(512, 10),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits
