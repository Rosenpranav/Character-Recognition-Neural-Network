
import json
import numpy as np

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer class labels into one-hot encoded vectors.

    Parameters
    ----------
    labels : np.ndarray of shape (N,)
        Integer class indices in [0, num_classes).
    num_classes : int
        Total number of distinct classes.

    Returns
    -------
    np.ndarray of shape (N, num_classes)
        One-hot encoded label matrix.
    """
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def stratified_split(labels: np.ndarray, seed: int = SEED):
    """
    Compute stratified train/validation/test indices so that every class
    is represented in the same proportion in each split.

    Parameters
    ----------
    labels : np.ndarray of shape (N,)
        Integer class labels for the full dataset.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    (train_idx, val_idx, test_idx) : tuple of np.ndarray
        Index arrays into the original dataset for each split.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def load_and_prepare(data_dir: str = "data"):
    """
    Load raw images/labels from disk and produce normalized, flattened,
    one-hot encoded, and split datasets ready for training.

    Returns
    -------
    dict
        Dictionary with keys:
        'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        (y_* are one-hot encoded), plus 'y_train_int', 'y_val_int',
        'y_test_int' (integer labels, useful for confusion matrices),
        and 'classes' (list of class label strings).
    """
    images = np.load(f"{data_dir}/raw_images.npy")   # (N, H, W) uint8
    labels = np.load(f"{data_dir}/raw_labels.npy")   # (N,) int64
    classes = json.load(open(f"{data_dir}/classes.json"))
    num_classes = len(classes)

    # Flatten each HxW image into a single feature vector, normalize pixel
    # intensities from [0, 255] to [0, 1], then zero-center by subtracting
    # 0.5 (mapping to [-0.5, 0.5]). Zero-centering matters a lot here: our
    # images are mostly white background (~1.0) with dark strokes (~0.0),
    # so uncentered inputs are heavily biased toward large positive values.
    # That bias was found (empirically, during tuning) to push many ReLU
    # units into a saturated/dead regime immediately, stalling learning at
    # the "predict everything as one class" plateau. Centering keeps the
    # pre-activation distribution balanced around zero and lets gradients
    # flow properly from the very first epoch.
    N, H, W = images.shape
    X = images.reshape(N, H * W).astype(np.float32) / 255.0
    X = X - 0.5

    train_idx, val_idx, test_idx = stratified_split(labels)

    data = {
        "X_train": X[train_idx], "y_train_int": labels[train_idx],
        "X_val": X[val_idx],     "y_val_int": labels[val_idx],
        "X_test": X[test_idx],   "y_test_int": labels[test_idx],
        "classes": classes,
        "img_shape": (H, W),
    }
    data["y_train"] = one_hot(data["y_train_int"], num_classes)
    data["y_val"] = one_hot(data["y_val_int"], num_classes)
    data["y_test"] = one_hot(data["y_test_int"], num_classes)

    print(f"Train: {data['X_train'].shape[0]} | "
          f"Val: {data['X_val'].shape[0]} | "
          f"Test: {data['X_test'].shape[0]}")
    return data


if __name__ == "__main__":
    d = load_and_prepare()
    print("Feature vector length:", d["X_train"].shape[1])
    print("Number of classes:", d["y_train"].shape[1])
