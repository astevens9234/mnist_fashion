"""Dense Convolution Network"""

import torch
from torch import nn


class DenseNet(nn.Module):
    def __init__(self, arch: tuple = (4, 4, 4, 4), growth_rate: int = 32, num_classes: int = 10, num_channels: int = 64):
        """
        Args:
            arch (tuple, optional): Architecture of Dense Blocks. Defaults to (4, 4, 4, 4).
            growth_rate (int, optional): Number of Convolution Block Channels. Defaults to 32.
            num_classes (int, optional): Output classes. Defaults to 10.
            num_channels (int, optional): Number of Channels. Defaults to 64.
        """
        super().__init__()
        self.net = nn.Sequential(self.b1())

        for i, num_conv in enumerate(arch):
            self.net.add_module(f"dense_blk{i+1}", DenseBlock(num_conv, growth_rate))
            num_channels += num_conv * growth_rate

            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f"tran_blk{i+1}", transition_block(num_channels))

        self.net.add_module(
            name="last",
            module=nn.Sequential(
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.LazyLinear(num_classes),
            ),
        )

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    
    def forward(self, X):
        return self.net(X)


class DenseBlock(nn.Module):
    def __init__(self, num_conv, num_channels):
        super().__init__()
        layer = []
        for _ in range(num_conv):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1),
    )


def transition_block(num_channels):
    """Reduce channels by applying a 1x1 convolution layer
    & half dimensions with stride=2."""
    return nn.Sequential(
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )
