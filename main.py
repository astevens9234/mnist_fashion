"""Pytorch implimentation of the Fashion MNIST Classification Problem.

For GPU enabled computation, install pytorch in the venv from:
https://pytorch.org/get-started/locally/
"""

import logging
import pdb
import warnings

import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


warnings.simplefilter("ignore")
logging.basicConfig(
    filename="logging.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class NN(nn.Module):
    def __init__(self, resize: tuple = (28, 28)) -> None:
        super(NN, self).__init__()
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(resize), torchvision.transforms.ToTensor()]
        )
        # One hot encode the targets
        target_transform = torchvision.transforms.Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                0, torch.tensor(y), value=1
            )
        )
        self.training_data = torchvision.datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
        self.test_data = torchvision.datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
        self.flatten = nn.Flatten()

    def get_dataloader(self, train: bool = True, batch_size: int = 64):
        data_ = self.training_data if train else self.test_data
        return DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)

    def text_labels(self, idx):
        labels = [
            "t-shirt",
            "trouser",
            "pullover",
            "dress",
            "coat",
            "sandal",
            "shirt",
            "sneaker",
            "bag",
            "ankle boot",
        ]
        return [labels[int(i)] for i in idx]


class linear_relu(NN):
    def __init__(self):
        super(linear_relu, self).__init__()
        self.linear_reLU_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_reLU_stack(x)
        return logits
    

class dropout_relu(NN):
    def __init__(self, layer1=.25, layer2=.25):
        super(dropout_relu, self).__init__()
        self.dropout_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(p=layer1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=layer2),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.dropout_relu_stack(x)
        return logits


class MLP(NN):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits


class Classifier(nn.Module):
    """Base class for Classification Models"""

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])  # * operator to unpack elements of batch iterable
        self.plot("loss", self.loss(Y_hat, batch[-1]), train=False)
        self.plot("accuracy", self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged: bool = True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        comparison = (preds == Y.reshape(-1)).type(torch.float32)
        return comparison.mean() if averaged else comparison

    def configure_optimizer(self):
        raise NotImplementedError

    def configure_lossfx(self):
        raise NotImplementedError


def training_loop(dataloader, model, lossfx, optimizer, device):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = lossfx(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, lossfx, device):
    model.eval()
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():  # Don't compute gradients during test eval
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += lossfx(pred, y).item()
            # pdb.set_trace()
            # NOTE: Because of the one-hot encoding, pay special mind to the dimensions.
            #       Cast the target to the same shape as the prediction.
            correct += (
                (pred.argmax(1) == torch.argmax(y, dim=1))
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= n_batches
    correct /= size
    logging.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main(
    batch_size: int = 64,
    num_workers: int = 1,
    learning_rate: float = 1e-3,
    epochs: int = 100,
):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info(
        f"\n Using device: {device} \n batch_size: {batch_size} \n num_workers: {num_workers} \n learning_rate: {learning_rate} \n epochs: {epochs} \n",
    )

    data = NN(resize=(28, 28))
    # print(len(data.training_data))  # 60000
    # print(len(data.test_data))      # 10000

    train_dataloader = NN.get_dataloader(data, train=True, batch_size=batch_size)
    test_dataloader = NN.get_dataloader(data, train=False, batch_size=batch_size)
    # print(len(train_dataloader))    # 938 * 64 batches == 60,032
    # print(len(test_dataloader))     # 157 * 64 batches == 10,048

    # TODO: cleaner way to specify model class
    model = dropout_relu(layer1=.2, layer2=.2).to(device)
    lossfx = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    logging.info(
        f"\n model: {model} \n lossfx: {lossfx} \n optimizer: {optimizer} \n",
    )

    for i in tqdm(range(epochs)):
        logging.info(f"Epoch {i+1}\n-------------------------------")
        training_loop(train_dataloader, model, lossfx, optimizer, device)
        test_loop(test_dataloader, model, lossfx, device)

    torch.save(model.state_dict(), "model.pth")

    logging.info("Finished!")


if __name__ == "__main__":
    main()
