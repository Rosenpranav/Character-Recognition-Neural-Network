import sys 
import json 
import numpy as np 
from PIL import Image, ImageOps 
 
from neural_network import NeuralNetwork 
 
 
def load_model(path="artifacts/model_weights.npz"): 
    npz = np.load(path, allow_pickle=True) 
    layer_sizes = list(npz["layer_sizes"]) 
    activation = str(npz["activation"]) 
    net = NeuralNetwork(layer_sizes, hidden_activation=activation) 
    n_layers = len(layer_sizes) - 1 
    net.weights = [npz[f"W{i}"] for i in range(n_layers)] 
    net.biases = [npz[f"b{i}"] for i in range(n_layers)] 
    return net 
 
 
def preprocess_image(path, img_size=32): 
    """ 
    Load an arbitrary image file and convert it into the same 32x32, 
    zero-centered format the network was trained on. 
 
    Parameters 
    ---------- 
    path : str 
        Path to an image file (png/jpg/etc.) containing one character. 
    img_size : int 
        Target square size (must match training, default 32). 
 
    Returns 
    ------- 
    np.ndarray of shape (1, img_size*img_size) 
        Flattened, normalized, zero-centered feature vector ready for 
        the network's forward() / predict() methods. 
    PIL.Image.Image 
        The processed grayscale image, for optional visual inspection. 
    """ 
    img = Image.open(path).convert("L")  # grayscale 
 
    if np.array(img).mean() < 127: 
        img = ImageOps.invert(img) 
 
    arr = np.array(img) 
    mask = arr < 200 
    if mask.any(): 
        ys, xs = np.where(mask) 
        top, bottom = ys.min(), ys.max() 
        left, right = xs.min(), xs.max() 
        h, w = bottom - top + 1, right - left + 1 
        pad = int(0.55 * max(h, w)) 
        top = max(0, top - pad) 
        bottom = min(arr.shape[0] - 1, bottom + pad) 
        left = max(0, left - pad) 
        right = min(arr.shape[1] - 1, right + pad) 
        img = img.crop((left, top, right + 1, bottom + 1)) 
 
    w, h = img.size 
    side = max(w, h) 
    square = Image.new("L", (side, side), color=255) 
    square.paste(img, ((side - w) // 2, (side - h) // 2)) 
    small = square.resize((img_size, img_size), Image.BICUBIC) 
 
    x = np.array(small, dtype=np.float32).reshape(1, -1) / 255.0 
    x = x - 0.5 
    return x, small 
 
 
def predict_image(path, top_k=3): 
    """ 
    Predict the character class for a single image. 
    """ 
    classes = json.load(open("data/classes.json")) 
    net = load_model() 
    x, processed_img = preprocess_image(path) 
 
    probs = net.predict_proba(x)[0] 
    top_idx = np.argmax(probs) 
 
    print(f"\nPrediction for '{path}':") 
    print(f"  {classes[top_idx]}") 
 
    processed_img.save("last_prediction_input.png") 
    print("\n(Saved the model's 32x32 processed view of your image as " 
          "'last_prediction_input.png' — check this if the prediction " 
          "looks wrong; it shows exactly what the network 'saw'.)") 
 
    return classes[top_idx], probs[top_idx] 
 
 
if __name__ == "__main__": 
    if len(sys.argv) != 2: 
        print("Usage: python predict.py path/to/image.png") 
        sys.exit(1) 
    predict_image(sys.argv[1])