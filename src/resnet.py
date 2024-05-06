"""Implimentation of ResNet.

This architecture is pretty complicated. See docs at:
https://arxiv.org/abs/1512.03385 """

from torch import nn
from torch.nn import functional as F


class ResNet18(nn.Module):
    def __init__(self, arch, num_classes=10):
        """18 Layer ResNet Model.

        Args:
            arch (tuple): Architecture of Residual Blocks, i.e. ((2, 64), (2, 128) ... )
            num_classes (int): Number of classes to output. Defaults to 10.
        """
        super().__init__()
        self.net = nn.Sequential(
            self.b1(),
        )
        for i, b in enumerate(arch):
            self.net.add_module(
                f"b{i+2}", self.residual_block(*b, first_block=(i == 0))
            )
        self.net.add_module(
            name="last",
            module=nn.Sequential(
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten(),
                nn.LazyLinear(num_classes),
            ),
        )

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def residual_block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

    def forward(self, x):
        return self.net(x)


class Residual(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        """Residual Block Class.

        Args:
            num_channels (int): Number of Channels
            use_1x1conv (bool, optional): Optional transformation to input, to preserve the input operation across convolutions. Defaults to True.
            strides (int, optional): Distance covered by Convolution Kernel. Defaults to 1.
        """
        super().__init__()
        self.conv1 = nn.LazyConv2d(
            num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
