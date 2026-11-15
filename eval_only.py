"""
eval_only.py
======================================================================
Runs the ENTIRE evaluation half of the project on models that are
ALREADY TRAINED and saved to disk. No training happens here.

Requires these weight files to exist (either naming scheme works):
    models/resnet50_baseline.weights.h5        (or .h5)
    models/resnet50_noise_augmented.weights.h5 (or .h5)
    models/resnet50_adversarial.weights.h5     (or .h5)

Produces:
    results/robustness_comparison.csv
    results/denoising_defense.csv
    results/explainability_gallery.png
    results/explanation_metrics.csv
    results/RESULTS_SUMMARY.md

Install deps first:   pip install pandas scipy tqdm matplotlib
======================================================================
"""

import os, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import gaussian_filter, median_filter
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIG (matches final_project.py)
# ----------------------------------------------------------------------
DATA_DIR      = "./tiny-imagenet-200"
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_CLASSES   = 200
LEARNING_RATE = 1e-4

EPS           = 8 / 255.0
PGD_STEP_SIZE = 2 / 255.0
PGD_STEPS     = 7

EVAL_BATCHES  = 30     # ~960 val images; set None for the full 10k
SEED          = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
os.makedirs("results", exist_ok=True)

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass


def free_memory(*names):
    g = globals()
    for n in names:
        g.pop(n, None)
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


# ----------------------------------------------------------------------
# MODEL + WEIGHTS-ONLY LOADER
# ----------------------------------------------------------------------
def build_model(trainable_base=False, fine_tune_at=None):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    base = ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    if fine_tune_at is not None:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        base.trainable = trainable_base
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs)


def resolve_weights(name):
    for ext in (".weights.h5", ".h5"):
        p = f"models/resnet50_{name}{ext}"
        if os.path.exists(p):
            return p
    return None


def load_for_eval(name_or_path):
    path = name_or_path if os.path.exists(name_or_path) else resolve_weights(name_or_path)
    if path is None:
        raise FileNotFoundError(f"weights not found for '{name_or_path}'")
    m = build_model(fine_tune_at=140)
    m.load_weights(path)
    m.compile(optimizer=optimizers.Adam(LEARNING_RATE),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m


BASELINE_PATH    = resolve_weights("baseline")
NOISE_MODEL_PATH = resolve_weights("noise_augmented")
MODEL_SAVE_PATH  = resolve_weights("adversarial")
MODEL_PATHS = {"Baseline": BASELINE_PATH,
               "Noise-Augmented": NOISE_MODEL_PATH,
               "Adversarial": MODEL_SAVE_PATH}

# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

# ----------------------------------------------------------------------
# PERTURBATIONS
# ----------------------------------------------------------------------
def add_gaussian_noise_batch(images, sigma, max_val=1.0):
    noise = np.random.normal(0, sigma * max_val, images.shape).astype(np.float32)
    return np.clip(images + noise, 0.0, max_val).astype(np.float32)


@tf.function
def fgsm_attack(model, images, labels, eps=EPS):
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images, training=False)
        loss = tf.keras.losses.categorical_crossentropy(labels, preds)
    grads = tape.gradient(loss, images)
    return tf.clip_by_value(images + eps * tf.sign(grads), 0.0, 1.0)


@tf.function
def pgd_attack(model, images, labels, eps=EPS, alpha=PGD_STEP_SIZE, steps=PGD_STEPS):
    adv = images + tf.random.uniform(tf.shape(images), -eps, eps)
    adv = tf.clip_by_value(adv, 0.0, 1.0)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            preds = model(adv, training=False)
            loss = tf.keras.losses.categorical_crossentropy(labels, preds)
        grads = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grads)
        adv = tf.clip_by_value(adv, images - eps, images + eps)
        adv = tf.clip_by_value(adv, 0.0, 1.0)
    return adv


# ----------------------------------------------------------------------
# ACCURACY METER + TRANSFORMS
# ----------------------------------------------------------------------
def _to_np(x):
    return x.numpy() if hasattr(x, "numpy") else x


def measure_accuracy(model, val_gen, transform=None, max_batches=None, desc="eval"):
    correct, total = 0, 0
    n = len(val_gen) if max_batches is None else min(max_batches, len(val_gen))
    for i in tqdm(range(n), desc=desc, leave=False):
        images, labels = val_gen[i]
        if transform is not None:
            images = _to_np(transform(images, labels))
        logits = model(tf.convert_to_tensor(images, tf.float32), training=False).numpy()
        correct += int(np.sum(np.argmax(logits, 1) == np.argmax(labels, 1)))
        total += len(labels)
    return correct / total


def make_noise_transform(sigma):
    return lambda images, labels: add_gaussian_noise_batch(images, sigma)

def make_fgsm_transform(model, eps=EPS):
    return lambda images, labels: fgsm_attack(
        model, tf.convert_to_tensor(images, tf.float32),
        tf.convert_to_tensor(labels, tf.float32), eps=eps)

def make_pgd_transform(model, eps=EPS, alpha=PGD_STEP_SIZE, steps=PGD_STEPS):
    return lambda images, labels: pgd_attack(
        model, tf.convert_to_tensor(images, tf.float32),
        tf.convert_to_tensor(labels, tf.float32), eps=eps, alpha=alpha, steps=steps)


def evaluate_all_conditions(model, val_gen, max_batches=EVAL_BATCHES):
    r = {}
    r["clean"]      = measure_accuracy(model, val_gen, None, max_batches, "clean")
    r["gauss_0.05"] = measure_accuracy(model, val_gen, make_noise_transform(0.05), max_batches, "g0.05")
    r["gauss_0.10"] = measure_accuracy(model, val_gen, make_noise_transform(0.10), max_batches, "g0.10")
    r["gauss_0.20"] = measure_accuracy(model, val_gen, make_noise_transform(0.20), max_batches, "g0.20")
    r["fgsm"]       = measure_accuracy(model, val_gen, make_fgsm_transform(model), max_batches, "fgsm")
    r["pgd"]        = measure_accuracy(model, val_gen, make_pgd_transform(model),  max_batches, "pgd")
    return r


# ----------------------------------------------------------------------
# GRAD-CAM / SCORE-CAM
# ----------------------------------------------------------------------
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("no 4D conv layer")

def compute_gradcam(model, image, class_index=None, last_conv=None):
    if last_conv is None:
        last_conv = get_last_conv_layer_name(model)
    gm = models.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
    img = tf.convert_to_tensor(image[None], tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = gm(img, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        channel = preds[:, class_index]
    grads = tape.gradient(channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    cam = tf.maximum(cam, 0); cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), class_index

def compute_scorecam(model, image, class_index=None, last_conv=None, top_k=256, batch_size=64):
    if last_conv is None:
        last_conv = get_last_conv_layer_name(model)
    conv_model = models.Model(model.inputs, model.get_layer(last_conv).output)
    img = image[None].astype(np.float32)
    acts = conv_model(tf.convert_to_tensor(img), training=False).numpy()[0]
    H, W = image.shape[:2]
    acts_up = tf.image.resize(acts[None], (H, W)).numpy()[0]
    C = acts_up.shape[-1]
    if class_index is None:
        class_index = int(np.argmax(model(tf.convert_to_tensor(img), training=False).numpy()[0]))
    chans = np.argsort(acts_up.reshape(-1, C).max(0))[-top_k:] if (top_k and top_k < C) else np.arange(C)
    maps, masked = [], []
    for c in chans:
        m = acts_up[..., c]; rng = m.max() - m.min()
        if rng < 1e-8:
            continue
        mn = (m - m.min()) / rng
        maps.append(mn); masked.append(image * mn[..., None])
    maps, masked = np.array(maps, np.float32), np.array(masked, np.float32)
    if masked.shape[0] == 0:            # degenerate activations (e.g. noisy input) -> flat cam
        return np.zeros((H, W), np.float32), class_index
    scores = []
    for i in range(0, len(masked), batch_size):
        p = model(tf.convert_to_tensor(masked[i:i+batch_size]), training=False).numpy()
        scores.append(p[:, class_index])
    weights = tf.nn.softmax(np.concatenate(scores)).numpy()
    cam = np.tensordot(weights, maps, axes=(0, 0))
    cam = np.maximum(cam, 0); cam = cam / (cam.max() + 1e-8)
    return cam, class_index

def overlay_cam(image, cam, alpha=0.45):
    cam_rs = tf.image.resize(cam[..., None], image.shape[:2]).numpy().squeeze()
    heat = matplotlib.colormaps["jet"](cam_rs)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)

def collect_images(val_gen, n):
    imgs, lbls, i = [], [], 0
    while sum(len(x) for x in imgs) < n and i < len(val_gen):
        im, lb = val_gen[i]; imgs.append(im); lbls.append(lb); i += 1
    return np.concatenate(imgs)[:n], np.concatenate(lbls)[:n]


# ----------------------------------------------------------------------
# 1) ROBUSTNESS TABLE
# ----------------------------------------------------------------------
def run_robustness():
    rows = {}
    for name, path in MODEL_PATHS.items():
        if path is None:
            print(f"[skip] {name}: weights not found"); continue
        print(f"\nEvaluating: {name}")
        m = load_for_eval(path)
        rows[name] = evaluate_all_conditions(m, val_gen, EVAL_BATCHES)
        free_memory("m")
    df = pd.DataFrame(rows).T
    df.to_csv("results/robustness_comparison.csv")
    print("\n===== ROBUSTNESS COMPARISON (accuracy) =====")
    print(df.round(4))
    return df


# ----------------------------------------------------------------------
# 2) DENOISING DEFENSE
# ----------------------------------------------------------------------
def denoise_batch(images, method="gaussian", param=1.0):
    out = np.empty_like(images, dtype=np.float32)
    for i in range(images.shape[0]):
        for c in range(images.shape[-1]):
            if method == "gaussian":
                out[i, ..., c] = gaussian_filter(images[i, ..., c], sigma=param)
            elif method == "median":
                out[i, ..., c] = median_filter(images[i, ..., c], size=int(param))
            else:
                out[i, ..., c] = images[i, ..., c]
    return np.clip(out, 0.0, 1.0)

def make_pgd_then_denoise_transform(model, method="gaussian", param=1.0, **pgd_kwargs):
    pgd_t = make_pgd_transform(model, **pgd_kwargs)
    def _t(images, labels):
        return denoise_batch(_to_np(pgd_t(images, labels)), method, param)
    return _t

def run_denoise(method="gaussian", param=1.0):
    rows = {}
    for name, path in MODEL_PATHS.items():
        if path is None:
            continue
        m = load_for_eval(path)
        pgd_only = measure_accuracy(m, val_gen, make_pgd_transform(m), EVAL_BATCHES, f"{name}:pgd")
        pgd_den  = measure_accuracy(m, val_gen,
                    make_pgd_then_denoise_transform(m, method=method, param=param),
                    EVAL_BATCHES, f"{name}:pgd+den")
        rows[name] = {"pgd": pgd_only, "pgd+denoise": pgd_den, "recovered": pgd_den - pgd_only}
        free_memory("m")
    df = pd.DataFrame(rows).T
    df.to_csv("results/denoising_defense.csv")
    print(f"\n===== DENOISING DEFENSE ({method}, param={param}) =====")
    print(df.round(4))
    return df


# ----------------------------------------------------------------------
# 3) GALLERY
# ----------------------------------------------------------------------
def build_gallery(model, val_gen, n_images=3, eps=EPS, noise_sigma=0.10,
                  save_path="results/explainability_gallery.png"):
    imgs, lbls = collect_images(val_gen, 64)
    preds = model(tf.convert_to_tensor(imgs, tf.float32), training=False).numpy()
    ok = np.where(np.argmax(preds, 1) == np.argmax(lbls, 1))[0][:n_images]
    if len(ok) == 0:
        print("  [gallery] no correctly-classified samples found."); return
    titles = ["Clean", "Grad-CAM\n(clean)", "Score-CAM\n(clean)",
              "Gaussian", "Grad-CAM\n(gauss)", "FGSM", "Grad-CAM\n(FGSM)",
              "PGD", "Grad-CAM\n(PGD)"]
    ncols = len(titles)
    fig, axes = plt.subplots(len(ok), ncols, figsize=(2.1*ncols, 2.3*len(ok)))
    if len(ok) == 1:
        axes = axes[None, :]
    for r, idx in enumerate(ok):
        img = imgs[idx]; lab = lbls[idx][None]; c = int(np.argmax(lbls[idx]))
        gauss = add_gaussian_noise_batch(img[None], noise_sigma)[0]
        fgsm  = _to_np(fgsm_attack(model, tf.convert_to_tensor(img[None], tf.float32),
                                   tf.convert_to_tensor(lab, tf.float32), eps=eps))[0]
        pgd   = _to_np(pgd_attack(model, tf.convert_to_tensor(img[None], tf.float32),
                                  tf.convert_to_tensor(lab, tf.float32)))[0]
        gc_clean, _ = compute_gradcam(model, img,   class_index=c)
        sc_clean, _ = compute_scorecam(model, img,  class_index=c, top_k=256)
        gc_gauss, _ = compute_gradcam(model, gauss, class_index=c)
        gc_fgsm,  _ = compute_gradcam(model, fgsm,  class_index=c)
        gc_pgd,   _ = compute_gradcam(model, pgd,   class_index=c)
        panels = [(img, None), (img, gc_clean), (img, sc_clean),
                  (gauss, None), (gauss, gc_gauss), (fgsm, None), (fgsm, gc_fgsm),
                  (pgd, None), (pgd, gc_pgd)]
        for col, (base, heat) in enumerate(panels):
            ax = axes[r, col]
            ax.imshow(base if heat is None else overlay_cam(base, heat)); ax.axis("off")
            if r == 0:
                ax.set_title(titles[col], fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved gallery -> {save_path}")

def run_gallery():
    path = BASELINE_PATH or MODEL_SAVE_PATH     # gallery on the real classifier
    m = load_for_eval(path)
    build_gallery(m, val_gen, n_images=3)
    free_memory("m")


# ----------------------------------------------------------------------
# 4) EXPLANATION FAITHFULNESS & STABILITY
# ----------------------------------------------------------------------
def explanation_metrics(model, val_gen, cam_fn=compute_gradcam, n_images=40, stab_sigma=0.05):
    imgs, lbls = collect_images(val_gen, n_images)
    drops, incs, stabs = [], [], []
    for k in tqdm(range(len(imgs)), desc="faithfulness", leave=False):
        img = imgs[k]
        base = model(tf.convert_to_tensor(img[None], tf.float32), training=False).numpy()[0]
        c = int(np.argmax(base)); Yc = float(base[c])
        cam, _ = cam_fn(model, img, class_index=c)
        cam_rs = tf.image.resize(cam[..., None], img.shape[:2]).numpy().squeeze()
        Oc = float(model(tf.convert_to_tensor((img * cam_rs[..., None])[None], tf.float32),
                         training=False).numpy()[0][c])
        drops.append(max(0.0, Yc - Oc) / (Yc + 1e-8))
        incs.append(1.0 if Oc > Yc else 0.0)
        noisy = add_gaussian_noise_batch(img[None], stab_sigma)[0]
        cam_n, _ = cam_fn(model, noisy, class_index=c)
        stabs.append(np.corrcoef(cam.flatten(), cam_n.flatten())[0, 1])
    return {"avg_drop_%": 100*float(np.mean(drops)),
            "increase_in_confidence_%": 100*float(np.mean(incs)),
            "stability_corr": float(np.nanmean(stabs))}

def run_explanations():
    path = BASELINE_PATH or MODEL_SAVE_PATH     # explain the real classifier, not the weak model
    m = load_for_eval(path)
    res = {"Grad-CAM":  explanation_metrics(m, val_gen, compute_gradcam,  n_images=40),
           "Score-CAM": explanation_metrics(m, val_gen, compute_scorecam, n_images=15)}
    df = pd.DataFrame(res).T
    df.to_csv("results/explanation_metrics.csv")
    print("\n===== EXPLANATION FAITHFULNESS & STABILITY =====")
    print(df.round(3))
    free_memory("m")
    return df


# ----------------------------------------------------------------------
# 5) ABLATION
# ----------------------------------------------------------------------
def run_ablation(batches=15):
    if MODEL_SAVE_PATH is None:
        print("[ablation] adversarial weights not found."); return
    m = load_for_eval(MODEL_SAVE_PATH)
    print("\nPGD epsilon sweep (steps=7):")
    for e in [2/255., 4/255., 8/255., 16/255.]:
        acc = measure_accuracy(m, val_gen, make_pgd_transform(m, eps=e, steps=7),
                               batches, f"eps={e*255:.0f}/255")
        print(f"  eps={e*255:.0f}/255 : {acc:.4f}")
    print("PGD steps sweep (eps=8/255):")
    for s in [1, 7, 10]:
        acc = measure_accuracy(m, val_gen, make_pgd_transform(m, steps=s),
                               batches, f"steps={s}")
        print(f"  steps={s} : {acc:.4f}")
    free_memory("m")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Models found:", {k: v for k, v in MODEL_PATHS.items()})
    rob = run_robustness()
    den = run_denoise(method="gaussian", param=1.0)
    run_gallery()
    expl = run_explanations()
    run_ablation(batches=15)

    with open("results/RESULTS_SUMMARY.md", "w") as f:
        f.write("# Results Summary\n\n## Robustness\n\n")
        f.write(rob.round(4).to_markdown() + "\n\n## Denoising\n\n")
        f.write(den.round(4).to_markdown() + "\n\n## Explanation metrics\n\n")
        f.write(expl.round(3).to_markdown() + "\n")
    print("\nAll evaluation outputs written to results/.")
