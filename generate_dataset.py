import os
import json
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = 32                 # final square image side length (pixels)
CANVAS_SIZE = 64              # oversized canvas used before augment+resize
SAMPLES_PER_CLASS = 120       # > 50 required; extra headroom for a clean split
SEED = 42

CLASSES = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [str(d) for d in range(1, 10)]
# 26 letters + 9 digits = 35 classes, matches the assignment spec.

def _find_fonts() -> list:
    """
    Locate usable TrueType font files on whichever OS this script runs on
    (Linux, Windows, or macOS), so the dataset generator works out of the
    box without hard-coding one platform's font paths.

    Returns
    -------
    list[str]
        Paths to existing, loadable .ttf/.ttc font files.
    """
    candidates = []

    # Linux (common distro font locations)
    linux_dir = "/usr/share/fonts"
    if os.path.isdir(linux_dir):
        candidates += glob.glob(f"{linux_dir}/**/*.ttf", recursive=True)

    # Windows — check both the WINDIR env var and the hardcoded default,
    # since some locked-down environments don't expose WINDIR reliably.
    win_font_dirs = []
    windir = os.environ.get("WINDIR")
    if windir:
        win_font_dirs.append(os.path.join(windir, "Fonts"))
    win_font_dirs.append(r"C:\Windows\Fonts")
    for win_fonts_dir in win_font_dirs:
        if not os.path.isdir(win_fonts_dir):
            continue
        preferred = [
            "arial.ttf", "arialbd.ttf", "ariali.ttf",
            "times.ttf", "timesbd.ttf", "timesi.ttf",
            "cour.ttf", "courbd.ttf",
            "calibri.ttf", "calibrib.ttf",
            "verdana.ttf", "verdanab.ttf",
            "tahoma.ttf", "tahomabd.ttf",
            "georgia.ttf", "georgiab.ttf",
            "comic.ttf", "comicbd.ttf",
            "consola.ttf", "consolab.ttf",
        ]
        for name in preferred:
            fp = os.path.join(win_fonts_dir, name)
            if os.path.exists(fp):
                candidates.append(fp)
        candidates += glob.glob(f"{win_fonts_dir}/*.ttf")
        candidates += glob.glob(f"{win_fonts_dir}/*.ttc")

    # macOS
    for mac_dir in ["/System/Library/Fonts", "/Library/Fonts",
                     os.path.expanduser("~/Library/Fonts")]:
        if os.path.isdir(mac_dir):
            candidates += glob.glob(f"{mac_dir}/*.ttf") + glob.glob(f"{mac_dir}/*.ttc")

    # Reliable cross-platform fallback: matplotlib ships its own copy of
    # the DejaVu font family (used for plot text) inside its package data.
    # Since matplotlib is already a required dependency of this project,
    # this guarantees at least a handful of usable, varied fonts even if
    # no OS fonts could be found or accessed (e.g. locked-down font
    # folders, unusual Windows configurations, minimal Linux containers).
    try:
        import matplotlib
        mpl_font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        if os.path.isdir(mpl_font_dir):
            candidates += glob.glob(f"{mpl_font_dir}/*.ttf")
    except Exception:
        pass

    # De-duplicate while preserving order, and verify each font actually loads.
    seen, usable = set(), []
    for fp in candidates:
        key = os.path.normcase(os.path.abspath(fp))
        if key in seen:
            continue
        seen.add(key)
        try:
            ImageFont.truetype(fp, 20)
            usable.append(fp)
        except Exception:
            continue
        if len(usable) >= 24:   # plenty of variety; keep dataset generation fast
            break

    return usable


FONT_PATHS = _find_fonts()
if not FONT_PATHS:
    raise RuntimeError(
        "No usable TrueType fonts were found on this system (checked "
        "system font folders and matplotlib's bundled fonts). Please "
        "ensure matplotlib is installed correctly (pip install matplotlib "
        "--upgrade) or install at least one .ttf font, then try again."
    )


def render_base_glyph(char: str, font_path: str, font_size: int) -> Image.Image:
    """
    Render a single character onto a blank square canvas using a given font.

    Parameters
    ----------
    char : str
        The character to render (e.g. 'A', '7').
    font_path : str
        Path to a .ttf font file used to draw the glyph.
    font_size : int
        Font size in points.

    Returns
    -------
    PIL.Image.Image
        A grayscale (mode 'L') image of size (CANVAS_SIZE, CANVAS_SIZE)
        with a white background and the character drawn in black,
        centered using its bounding box.
    """
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CANVAS_SIZE - w) / 2 - bbox[0]
    y = (CANVAS_SIZE - h) / 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return img


def augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """
    Apply a random combination of realistic augmentations to a glyph image.

    This introduces the kind of variability seen in real handwritten or
    printed characters: slight rotation, translation (off-center placement),
    stroke-width change (via slight blur + re-threshold, simulating thicker/
    thinner pens), and additive pixel noise.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale glyph image to augment.
    rng : numpy.random.Generator
        Random number generator (for reproducibility).

    Returns
    -------
    PIL.Image.Image
        The augmented grayscale image, same size as input.
    """
    # Random rotation between -18 and +18 degrees.
    angle = rng.uniform(-18, 18)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

    # Random translation (shift) up to ~10% of canvas size.
    max_shift = int(CANVAS_SIZE * 0.10)
    dx = rng.integers(-max_shift, max_shift + 1)
    dy = rng.integers(-max_shift, max_shift + 1)
    img = Image.fromarray(np.roll(np.array(img), (dy, dx), axis=(0, 1)))

    # Random slight blur to simulate stroke-width / anti-aliasing variation.
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))

    # Random mild scaling (zoom in/out) then crop/pad back to CANVAS_SIZE.
    scale = rng.uniform(0.85, 1.15)
    new_size = max(8, int(CANVAS_SIZE * scale))
    img = img.resize((new_size, new_size), Image.BICUBIC)
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
    off = ((CANVAS_SIZE - new_size) // 2, (CANVAS_SIZE - new_size) // 2)
    canvas.paste(img, off)
    img = canvas

    # Additive Gaussian pixel noise.
    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_dataset():
    """
    Generate the full synthetic dataset and save it to disk as .npy files.

    For every one of the 35 classes, this renders SAMPLES_PER_CLASS
    augmented glyph images (cycling through all available fonts and font
    sizes for base variety, then applying independent random augmentation
    to each sample), resizes each to IMG_SIZE x IMG_SIZE, and stores the
    raw pixel arrays and integer labels.
    """
    rng = np.random.default_rng(SEED)
    os.makedirs("data", exist_ok=True)

    images, labels = [], []
    font_sizes = [34, 38, 42, 46]

    for class_idx, char in enumerate(CLASSES):
        for i in range(SAMPLES_PER_CLASS):
            font_path = FONT_PATHS[i % len(FONT_PATHS)]
            font_size = font_sizes[i % len(font_sizes)]
            base = render_base_glyph(char, font_path, font_size)
            aug = augment(base, rng)
            small = aug.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            images.append(np.array(small, dtype=np.uint8))
            labels.append(class_idx)

    images = np.stack(images)          # (N, IMG_SIZE, IMG_SIZE)
    labels = np.array(labels, dtype=np.int64)

    np.save("data/raw_images.npy", images)
    np.save("data/raw_labels.npy", labels)
    with open("data/classes.json", "w") as f:
        json.dump(CLASSES, f)

    print(f"Generated dataset: {images.shape[0]} samples, "
          f"{len(CLASSES)} classes, image size {IMG_SIZE}x{IMG_SIZE}")
    print(f"Fonts used: {len(FONT_PATHS)}")
    return images, labels


if __name__ == "__main__":
    build_dataset()
