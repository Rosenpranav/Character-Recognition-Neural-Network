import json
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_and_prepare
from neural_network import NeuralNetwork, ACTIVATIONS, softmax


def load_model(path="artifacts/model_weights.npz"):
   
    npz = np.load(path, allow_pickle=True)
    layer_sizes = list(npz["layer_sizes"])
    activation = str(npz["activation"])

    net = NeuralNetwork(layer_sizes, hidden_activation=activation)
    n_layers = len(layer_sizes) - 1
    net.weights = [npz[f"W{i}"] for i in range(n_layers)]
    net.biases = [npz[f"b{i}"] for i in range(n_layers)]
    return net


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
   
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray, classes: list) -> dict:
    
    metrics = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        metrics[cls] = {"precision": precision, "recall": recall,
                          "f1": f1, "support": int(support)}
    return metrics


def plot_confusion_matrix(cm: np.ndarray, classes: list, path: str):
    """Render and save the confusion matrix as a heatmap image."""
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def analyze_misclassifications(net, data, n_examples=8, seed=0):
    """
    Find misclassified test samples and save a labeled image grid plus a
    text summary for qualitative analysis.

    Parameters
    ----------
    net : NeuralNetwork
    data : dict
        Output of preprocessing.load_and_prepare().
    n_examples : int
        Number of misclassified examples to visualize/discuss.
    seed : int

    Returns
    -------
    list[dict]
        Each dict has true_label, pred_label, confidence, index.
    """
    X_test, y_true = data["X_test"], data["y_test_int"]
    classes = data["classes"]
    H, W = data["img_shape"]

    probs = net.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    wrong_idx = np.where(preds != y_true)[0]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(wrong_idx, size=min(n_examples, len(wrong_idx)), replace=False)

    records = []
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for ax, idx in zip(axes.flat, chosen):
        img = ((X_test[idx] + 0.5) * 255).reshape(H, W).astype(np.uint8)
        true_c, pred_c = classes[y_true[idx]], classes[preds[idx]]
        conf = confidences[idx]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {true_c}  Pred: {pred_c}\nconf={conf:.2f}", fontsize=9)
        ax.axis("off")
        records.append({"index": int(idx), "true_label": true_c,
                         "pred_label": pred_c, "confidence": float(conf)})
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("artifacts/misclassified_examples.png", dpi=130)
    plt.close(fig)

    return records


def main():
    data = load_and_prepare()
    net = load_model()
    classes = data["classes"]
    num_classes = len(classes)

    X_test, y_true = data["X_test"], data["y_test_int"]
    y_pred = net.predict(X_test)

    test_acc = float(np.mean(y_pred == y_true))
    print(f"Test Accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, num_classes)
    plot_confusion_matrix(cm, classes, "artifacts/confusion_matrix.png")

    metrics = per_class_metrics(cm, classes)
    macro_precision = np.mean([m["precision"] for m in metrics.values()])
    macro_recall = np.mean([m["recall"] for m in metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in metrics.values()])
    print(f"Macro Precision: {macro_precision:.4f} | "
          f"Macro Recall: {macro_recall:.4f} | Macro F1: {macro_f1:.4f}")

    records = analyze_misclassifications(net, data, n_examples=8)
    print(f"\nSaved {len(records)} misclassified examples for analysis:")
    for r in records:
        print(f"  idx={r['index']:4d}  true={r['true_label']:>2}  "
              f"pred={r['pred_label']:>2}  confidence={r['confidence']:.3f}")

    report = {
        "test_accuracy": test_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_metrics": metrics,
        "misclassified_examples": records,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    with open("artifacts/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved confusion matrix, misclassified examples, and full "
          "evaluation report to artifacts/")


if __name__ == "__main__":
    main()
