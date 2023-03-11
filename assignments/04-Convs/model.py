import torch.nn as nn
import torch


class Model(nn.Module):
    """
    Convolutional neural network model.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize the model.
        """
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # self.fc1 = nn.Linear(128, 128)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1568, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        x = self.fc2(x)
        return x
