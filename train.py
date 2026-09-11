import json
import os
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_and_prepare
from neural_network import NeuralNetwork, cross_entropy_loss

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_LAYERS = [256, 128]     # two hidden layers
HIDDEN_ACTIVATION = "relu"     # 'relu' or 'tanh'
LEARNING_RATE = 0.02
MOMENTUM = 0.9
BATCH_SIZE = 64
EPOCHS = 120
SEED = 42


def iterate_minibatches(X, y, batch_size, rng):
    """
    Yield shuffled mini-batches of (X, y) for one epoch.

    Parameters
    ----------
    X, y : np.ndarray
        Full training features / one-hot labels.
    batch_size : int
    rng : np.random.Generator

    Yields
    ------
    (X_batch, y_batch) : tuple of np.ndarray
    """
    n = X.shape[0]
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train():
    """
    Full training pipeline: load data, build the network, run the
    training loop with periodic validation, and save the trained model
    plus metric history and plots to disk.
    """
    data = load_and_prepare()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]

    os.makedirs("artifacts", exist_ok=True)

    layer_sizes = [input_dim] + HIDDEN_LAYERS + [num_classes]
    net = NeuralNetwork(layer_sizes, hidden_activation=HIDDEN_ACTIVATION, seed=SEED)

    rng = np.random.default_rng(SEED)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Architecture: {layer_sizes} | activation={HIDDEN_ACTIVATION} | "
          f"lr={LEARNING_RATE} | momentum={MOMENTUM} | batch={BATCH_SIZE}")

    for epoch in range(1, EPOCHS + 1):
        # ---------------- Training loop (one epoch) ----------------
        epoch_losses = []
        for X_batch, y_batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, rng):
            # 1. Forward propagation
            activations, pre_activations = net.forward(X_batch)
            # 2. Loss computation
            loss = cross_entropy_loss(activations[-1], y_batch)
            epoch_losses.append(loss)
            # 3. Backpropagation
            grads_W, grads_b = net.backward(activations, pre_activations, y_batch)
            # 4. Gradient descent parameter update
            net.update_params(grads_W, grads_b, LEARNING_RATE, MOMENTUM)

        train_loss = float(np.mean(epoch_losses))
        train_acc = net.accuracy(X_train, data["y_train_int"])

        # ---------------- Validation loop (forward-only) ----------------
        val_probs = net.predict_proba(X_val)
        val_loss = cross_entropy_loss(val_probs, y_val)
        val_acc = net.accuracy(X_val, data["y_val_int"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # ---------------- Test set evaluation ----------------
    test_acc = net.accuracy(data["X_test"], data["y_test_int"])
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # ---------------- Save model weights ----------------
    np.savez("artifacts/model_weights.npz",
              **{f"W{i}": W for i, W in enumerate(net.weights)},
              **{f"b{i}": b for i, b in enumerate(net.biases)},
              layer_sizes=np.array(layer_sizes),
              activation=HIDDEN_ACTIVATION)

    with open("artifacts/history.json", "w") as f:
        json.dump({**history, "test_acc": test_acc,
                    "hyperparameters": {
                        "layer_sizes": layer_sizes,
                        "activation": HIDDEN_ACTIVATION,
                        "learning_rate": LEARNING_RATE,
                        "momentum": MOMENTUM,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                    }}, f, indent=2)

    # ---------------- Plot loss & accuracy curves ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss over Training"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over Training"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("artifacts/training_curves.png", dpi=130)
    print("Saved model weights, history, and training curves to artifacts/")

    return net, data, history


if __name__ == "__main__":
    train()
