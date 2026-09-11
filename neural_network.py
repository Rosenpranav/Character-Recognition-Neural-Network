import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation, squashes input to (-1, 1)."""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of tanh w.r.t. its pre-activation input z."""
    return 1.0 - np.tanh(z) ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically-stable softmax over the last axis.

    Subtracting the row-wise max before exponentiating prevents overflow
    without changing the result, since softmax is shift-invariant.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
}


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Categorical cross-entropy loss between predicted probabilities and
    one-hot true labels, averaged over the batch.

    Parameters
    ----------
    y_pred : np.ndarray of shape (batch_size, num_classes)
        Softmax output probabilities.
    y_true : np.ndarray of shape (batch_size, num_classes)
        One-hot encoded ground-truth labels.

    Returns
    -------
    float
        Mean cross-entropy loss over the batch.
    """
    eps = 1e-9  # avoids log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


# ---------------------------------------------------------------------------
# The Neural Network
# ---------------------------------------------------------------------------
class NeuralNetwork:
    """
    A configurable multi-layer perceptron trained with plain NumPy.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of every layer, including input and output, e.g.
        [1024, 128, 64, 35] describes a network with a 1024-dim input,
        two hidden layers (128 and 64 units), and a 35-class output.
    hidden_activation : str
        Either 'relu' or 'tanh'; used for every hidden layer.
    seed : int
        Random seed for reproducible weight initialization.
    """

    def __init__(self, layer_sizes, hidden_activation="relu", seed=42):
        assert hidden_activation in ACTIVATIONS, "activation must be 'relu' or 'tanh'"
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # number of weight matrices
        self.activation_name = hidden_activation
        self.activation_fn, self.activation_deriv_fn = ACTIVATIONS[hidden_activation]

        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            if hidden_activation == "relu":
                # He initialization: suited to ReLU, keeps activation
                # variance stable across layers.
                std = np.sqrt(2.0 / fan_in)
            else:
                # Xavier/Glorot initialization: suited to tanh.
                std = np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0, std, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

        # Momentum buffers (used if momentum > 0 during training).
        self._vW = [np.zeros_like(W) for W in self.weights]
        self._vb = [np.zeros_like(b) for b in self.biases]

    # -----------------------------------------------------------------
    # Forward propagation
    # -----------------------------------------------------------------
    def forward(self, X: np.ndarray):
        """
        Run forward propagation through the whole network.

        Parameters
        ----------
        X : np.ndarray of shape (batch_size, input_dim)
            Input feature batch.

        Returns
        -------
        activations : list[np.ndarray]
            activations[0] is the input X; activations[i] for i>=1 is the
            output of layer i (post-activation for hidden layers,
            post-softmax for the final layer).
        pre_activations : list[np.ndarray]
            Linear (pre-activation) outputs z = XW + b for every layer,
            needed for computing derivatives during backpropagation.
        """
        activations = [X]
        pre_activations = []

        A = X
        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]
            pre_activations.append(Z)
            if i < self.num_layers - 1:
                A = self.activation_fn(Z)          # hidden layer activation
            else:
                A = softmax(Z)                      # output layer
            activations.append(A)

        return activations, pre_activations

    # -----------------------------------------------------------------
    # Backward propagation
    # -----------------------------------------------------------------
    def backward(self, activations, pre_activations, y_true):
        """
        Compute gradients of the cross-entropy loss with respect to every
        weight and bias, via backpropagation (reverse-mode chain rule).

        Parameters
        ----------
        activations : list[np.ndarray]
            Output of `forward`.
        pre_activations : list[np.ndarray]
            Output of `forward`.
        y_true : np.ndarray of shape (batch_size, num_classes)
            One-hot ground-truth labels.

        Returns
        -------
        grads_W, grads_b : list[np.ndarray], list[np.ndarray]
            Gradients for every weight matrix / bias vector, in the same
            order as self.weights / self.biases.
        """
        m = y_true.shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # --- Output layer ---
        # For softmax + cross-entropy combined, the gradient of the loss
        # w.r.t. the pre-activation z of the output layer simplifies
        # beautifully to (y_pred - y_true). This avoids ever having to
        # compute the full softmax Jacobian explicitly.
        y_pred = activations[-1]
        dZ = (y_pred - y_true) / m                      # (batch, out_dim)
        grads_W[-1] = activations[-2].T @ dZ
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

        # --- Hidden layers, propagating error backwards ---
        dA_prev = dZ @ self.weights[-1].T
        for i in reversed(range(self.num_layers - 1)):
            dZ = dA_prev * self.activation_deriv_fn(pre_activations[i])
            grads_W[i] = activations[i].T @ dZ
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dA_prev = dZ @ self.weights[i].T

        return grads_W, grads_b

    # -----------------------------------------------------------------
    # Parameter update (Gradient Descent, optionally with momentum)
    # -----------------------------------------------------------------
    def update_params(self, grads_W, grads_b, learning_rate, momentum=0.0):
        """
        Apply one gradient-descent step to all weights and biases.

        Parameters
        ----------
        grads_W, grads_b : list[np.ndarray]
            Gradients computed by `backward`.
        learning_rate : float
            Step size for the update.
        momentum : float
            Momentum coefficient in [0, 1). 0 = plain gradient descent.
        """
        for i in range(self.num_layers):
            self._vW[i] = momentum * self._vW[i] - learning_rate * grads_W[i]
            self._vb[i] = momentum * self._vb[i] - learning_rate * grads_b[i]
            self.weights[i] += self._vW[i]
            self.biases[i] += self._vb[i]

    # -----------------------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax class probabilities for input batch X."""
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted integer class label for each input row."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X: np.ndarray, y_true_int: np.ndarray) -> float:
        """Compute classification accuracy on a dataset."""
        preds = self.predict(X)
        return float(np.mean(preds == y_true_int))
