"""
qualitative_results.py
======================================================================
Qualitative "defense recovery" visualisations for the adversarial-
robustness project.

For each perturbation it finds validation images where:
    1. the BASELINE classifies the CLEAN image correctly,
    2. the BASELINE classifies the PERTURBED image INCORRECTLY,
    3. the DEFENDED model classifies the PERTURBED image correctly,
    4. the DEFENDED model classifies the CLEAN image correctly too,
i.e. the full "it broke, then the defense fixed it (both clean & perturbed)"
story. Each sample is plotted across the four scenarios, with Grad-CAM
overlays so you can see the attention collapse and recovery.

Perturbation -> Defense pairings:
    Gaussian noise  ->  noise-augmented model
    FGSM            ->  adversarially-trained model
    PGD             ->  adversarially-trained model

Run AFTER training (the model weight files must exist). Works whether the
weights were saved as full models (.h5) or weights-only (.weights.h5) —
we rebuild the architecture and only load weights, so nothing has to be
deserialized from disk.
======================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; PNGs are written to disk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------------------------------------------------
# CONFIG  (matches the main project)
# ----------------------------------------------------------------------
DATA_DIR     = "./tiny-imagenet-200"
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_CLASSES  = 200

EPS          = 8 / 255.0
PGD_STEP_SIZE = 2 / 255.0
PGD_STEPS    = 7
GAUSS_SIGMA  = 0.10

OUT_DIR      = "results/qualitative"
N_SAMPLES    = 3          # samples shown per perturbation
SCAN_BATCHES = 60         # how many val batches to search for story samples
SHOW_GRADCAM = True       # add a Grad-CAM row under each image row
SEED         = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# MODEL  (identical architecture to the project, so load_weights matches)
# ----------------------------------------------------------------------
def build_model(trainable_base=False, fine_tune_at=None):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))     # model expects [0,1]
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
    """Find the weight file for a model, accepting either naming scheme."""
    for ext in (".weights.h5", ".h5"):
        p = f"models/resnet50_{name}{ext}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No weights found for '{name}'. Looked for "
        f"models/resnet50_{name}.weights.h5 and .h5 — train it first.")


def load_model_weights(name):
    """Rebuild architecture + load weights only (avoids .h5 graph issues)."""
    path = resolve_weights(name)
    m = build_model(fine_tune_at=140)
    m.load_weights(path)
    print(f"  loaded {name:16s} <- {path}")
    return m


# ----------------------------------------------------------------------
# PERTURBATIONS  (all operate in the [0,1] image domain)
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
    adv = images + eps * tf.sign(grads)
    return tf.clip_by_value(adv, 0.0, 1.0)


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


def perturb(tag, images, labels, attacker):
    if tag == "gaussian":
        return add_gaussian_noise_batch(images, GAUSS_SIGMA)
    t_img = tf.convert_to_tensor(images, tf.float32)
    t_lab = tf.convert_to_tensor(labels, tf.float32)
    if tag == "fgsm":
        return fgsm_attack(attacker, t_img, t_lab).numpy()
    if tag == "pgd":
        return pgd_attack(attacker, t_img, t_lab).numpy()
    raise ValueError(tag)


# ----------------------------------------------------------------------
# GRAD-CAM
# ----------------------------------------------------------------------
def last_conv_name(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("no 4D conv layer found")


def grad_cam(model, image, class_index, last_conv=None):
    if last_conv is None:
        last_conv = last_conv_name(model)
    grad_model = models.Model(model.inputs,
                              [model.get_layer(last_conv).output, model.output])
    img = tf.convert_to_tensor(image[None], tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img, training=False)
        channel = preds[:, class_index]
    grads = tape.gradient(channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay(image, cam, alpha=0.45):
    cam_rs = tf.image.resize(cam[..., None], image.shape[:2]).numpy().squeeze()
    heat = cm.get_cmap("jet")(cam_rs)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


# ----------------------------------------------------------------------
# DATA + LABEL NAMES
# ----------------------------------------------------------------------
def build_val_gen():
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    if val_gen.samples == 0:
        raise RuntimeError(
            f"No images found under {os.path.join(DATA_DIR, 'val')}. "
            "Make sure the validation set has been reorganised into class "
            "sub-folders (the main training script does this).")
    return val_gen


def build_label_names(val_gen):
    """index -> human-readable name, via Tiny ImageNet words.txt if present."""
    idx_to_wnid = {v: k for k, v in val_gen.class_indices.items()}
    wnid_to_name = {}
    words = os.path.join(DATA_DIR, "words.txt")
    if os.path.exists(words):
        with open(words) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    wnid_to_name[parts[0]] = parts[1].split(",")[0]

    def name(idx):
        wnid = idx_to_wnid.get(idx, str(idx))
        label = wnid_to_name.get(wnid, wnid)
        return label[:18]
    return name


# ----------------------------------------------------------------------
# SAMPLE MINING
# ----------------------------------------------------------------------
def mine_story_samples(tag, baseline, defense, val_gen, n_samples, scan_batches):
    """Return story samples + recovery statistics for one perturbation."""
    collected = []
    n_broken = 0      # baseline: clean right, perturbed wrong
    n_recovered = 0   # of those, defense gets perturbed right

    n = min(scan_batches, len(val_gen))
    for bi in range(n):
        images, labels = val_gen[bi]
        true = np.argmax(labels, 1)

        base_clean = baseline(tf.convert_to_tensor(images, tf.float32),
                              training=False).numpy()
        bc_pred = np.argmax(base_clean, 1)

        pert = perturb(tag, images, labels, baseline)
        base_pert = baseline(tf.convert_to_tensor(pert, tf.float32),
                             training=False).numpy()
        bp_pred = np.argmax(base_pert, 1)

        def_pert = defense(tf.convert_to_tensor(pert, tf.float32),
                           training=False).numpy()
        dp_pred = np.argmax(def_pert, 1)

        def_clean = defense(tf.convert_to_tensor(images, tf.float32),
                            training=False).numpy()
        dc_pred = np.argmax(def_clean, 1)

        broke = (bc_pred == true) & (bp_pred != true)
        n_broken += int(broke.sum())
        n_recovered += int((broke & (dp_pred == true)).sum())

        story = broke & (dp_pred == true) & (dc_pred == true)
        for i in np.where(story)[0]:
            collected.append(dict(
                clean=images[i], pert=pert[i], true=int(true[i]),
                base_clean=(int(bc_pred[i]), float(base_clean[i].max())),
                base_pert=(int(bp_pred[i]), float(base_pert[i].max())),
                def_clean=(int(dc_pred[i]), float(def_clean[i].max())),
                def_pert=(int(dp_pred[i]), float(def_pert[i].max())),
            ))
            if len(collected) >= n_samples:
                break
        if len(collected) >= n_samples:
            break

    recovery = (n_recovered / n_broken) if n_broken else float("nan")
    return collected, dict(broken=n_broken, recovered=n_recovered, rate=recovery)


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------
def style_axis(ax, scenario, pred_idx, conf, true_idx, name_fn):
    correct = (pred_idx == true_idx)
    color = "#1a9641" if correct else "#d7191c"
    mark = "\u2713" if correct else "\u2717"
    ax.set_title(f"{scenario}\n{mark} {name_fn(pred_idx)} ({conf:.0%})",
                 fontsize=8, color=color)
    for s in ax.spines.values():
        s.set_color(color)
        s.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])


def plot_condition(title, samples, stats, baseline, defense, name_fn, save_path):
    if not samples:
        print(f"  [no story samples found for '{title}'] — "
              "try raising SCAN_BATCHES.")
        return

    scenarios = ["Clean\nBaseline", "Perturbed\nBaseline",
                 "Clean\nDefended", "Perturbed\nDefended"]
    rows_per_sample = 2 if SHOW_GRADCAM else 1
    nrows = len(samples) * rows_per_sample
    ncols = 4

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.3 * ncols, 2.5 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    lc_base = last_conv_name(baseline)
    lc_def = last_conv_name(defense)

    for s, smp in enumerate(samples):
        true = smp["true"]
        # which model + image + heatmap target per scenario column
        cols = [
            (baseline, smp["clean"], smp["base_clean"], lc_base),
            (baseline, smp["pert"],  smp["base_pert"],  lc_base),
            (defense,  smp["clean"], smp["def_clean"],  lc_def),
            (defense,  smp["pert"],  smp["def_pert"],   lc_def),
        ]
        img_row = s * rows_per_sample
        for c, (mdl, img, (pred, conf), lc) in enumerate(cols):
            ax = axes[img_row, c]
            ax.imshow(np.clip(img, 0, 1))
            style_axis(ax, scenarios[c], pred, conf, true, name_fn)
            if c == 0:
                ax.set_ylabel(f"sample {s+1}\ntrue: {name_fn(true)}",
                              fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=38)
            if SHOW_GRADCAM:
                axg = axes[img_row + 1, c]
                cam = grad_cam(mdl, img, pred, last_conv=lc)
                axg.imshow(overlay(img, cam))
                axg.set_xticks([]); axg.set_yticks([])
                if c == 0:
                    axg.set_ylabel("Grad-CAM", fontsize=8, rotation=0,
                                   ha="right", va="center", labelpad=38)

    rate = stats["rate"]
    supt = (f"{title}   |   baseline broke {stats['broken']} imgs, "
            f"defense recovered {stats['recovered']} "
            f"({rate:.0%})" if stats["broken"] else title)
    fig.suptitle(supt, fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    print("Loading data ...")
    val_gen = build_val_gen()
    name_fn = build_label_names(val_gen)

    print("Loading models ...")
    baseline = load_model_weights("baseline")
    noise_aug = load_model_weights("noise_augmented")
    adversarial = load_model_weights("adversarial")

    conditions = [
        ("Gaussian noise (sigma=0.10)  ->  Noise-Augmented defense",
         "gaussian", noise_aug,  "gaussian_recovery.png"),
        ("FGSM (eps=8/255)  ->  Adversarially-Trained defense",
         "fgsm",     adversarial, "fgsm_recovery.png"),
        ("PGD (eps=8/255, 7 steps)  ->  Adversarially-Trained defense",
         "pgd",      adversarial, "pgd_recovery.png"),
    ]

    summary = []
    for title, tag, defense, fname in conditions:
        print(f"\n=== {title} ===")
        samples, stats = mine_story_samples(
            tag, baseline, defense, val_gen, N_SAMPLES, SCAN_BATCHES)
        plot_condition(title, samples, stats, baseline, defense,
                       name_fn, os.path.join(OUT_DIR, fname))
        summary.append((title, stats))

    print("\n==================== RECOVERY SUMMARY ====================")
    for title, st in summary:
        if st["broken"]:
            print(f"{title}\n    baseline misclassified {st['broken']} "
                  f"(of scanned) images it had gotten right when clean;\n"
                  f"    the defense recovered {st['recovered']} of them "
                  f"= {st['rate']:.1%} recovery rate.")
        else:
            print(f"{title}\n    baseline was not broken on the scanned "
                  "subset (try a stronger perturbation or more batches).")
    print("\nFigures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
