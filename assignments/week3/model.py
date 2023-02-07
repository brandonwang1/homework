import torch
from typing import Callable
import torch


class MLP(torch.nn.Module):
    """
    Trains a multi-layer perceptron on the MNIST dataset.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = activation()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        initializer(self.layers[0].weight)
        self.layers.append(self.activation)
        for _ in range(hidden_count):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            initializer(self.layers[-1].weight)
            self.layers.append(self.activation)
        self.layers.append(torch.nn.Linear(hidden_size, num_classes))
        initializer(self.layers[-1].weight)
        # self.layers.append(self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = torch.flatten(x, 1)
        for layer in self.layers:
            x = layer(x)
        return x
