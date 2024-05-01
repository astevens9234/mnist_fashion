"""VGG Network implimentation."""

from torch import nn


class VGG(nn.Module):
    def __init__(self, arch: list = [(1, 64)]):
        super().__init__()
        conv_blocks = []
        for num_convs, out_channels in arch:
            conv_blocks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blocks,
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(10)
        )

    def forward(self, x):
        return self.net(x)


def vgg_block(num_convs: int = 3, out_channels: int = 2):
    """Sequence of Convolutions, followed by Pooling."""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)
