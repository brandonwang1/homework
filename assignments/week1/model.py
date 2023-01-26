import numpy as np


class LinearRegression:
    """
    Class for a linear regression model fitted using least squares.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.array([0, 0])
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the given input and output data.
        X: (n, d) array of input data
        y: (n,) array of output data
        Ref https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
        """
        A = np.hstack((X, np.ones((X.shape[0], 1))))
        m = np.linalg.lstsq(A, y, rcond=None)[0]
        self.w = m[:-1]
        self.b = m[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.
        X: (n, d) array of input data
        Returns: (n,) array of output data
        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model to the given input and output data.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
            lr (float): The learning rate.
            epochs (int): The number of epochs to train for.

        Returns:
            None
        """
        A = np.hstack((X, np.ones((X.shape[0], 1))))
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(epochs):
            y_hat = A @ np.hstack((self.w, self.b))
            grad = A.T @ (y_hat - y)
            self.w -= lr * grad[:-1]
            self.b -= lr * grad[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b
