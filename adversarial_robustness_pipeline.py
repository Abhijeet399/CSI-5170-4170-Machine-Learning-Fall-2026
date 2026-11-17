"""
================================================================================
 adversarial_robustness_pipeline.py
 CSI 5170/4170 — Machine Learning, Fall 2026 — Final Project
 Adversarial Robustness of CNNs: Gaussian Noise vs Adversarial Attacks,
 Defenses, Explainability (Grad-CAM / Score-CAM), and K-Fold Cross-Validation.

 Author: Abhijeet Bhattacharya <bhattacharya@oakland.edu>

 ONE self-contained pipeline containing everything built for this project:
   - Data prep (Tiny ImageNet download check + validation reorganization)
   - ResNet-50 transfer model (ImageNet preprocessing baked in; [0,1] domain)
   - Training: baseline (2-stage), noise-augmented, adversarial (v3 recipe)
   - Attacks: FGSM, PGD (white-box)
   - Defenses: noise augmentation, adversarial training, test-time denoising
   - Evaluation: robustness table, denoising defense, PGD eps/steps ablation
   - Explainability: Grad-CAM, Score-CAM + faithfulness/stability metrics
   - Qualitative "attack breaks baseline -> defense recovers" galleries
   - *** NEW: Stratified K-Fold cross-validation (feature-cached, efficient) ***

 USAGE
   python adversarial_robustness_pipeline.py --stage all
   python adversarial_robustness_pipeline.py --stage cv --folds 10
   python adversarial_robustness_pipeline.py --stage baseline,eval
   python adversarial_robustness_pipeline.py --stage cv --folds 3 --cv-per-class 25   # quick demo

   Stages: data, baseline, noise, adversarial, eval, qualitative, cv, all
   (eval/qualitative/cv load saved weights; train the models first or use --stage all)

 NOTE on cross-validation
   Full k-fold of deep CNN *training* is prohibitively expensive, so CV here
   freezes the ImageNet backbone, caches its 2048-d features ONCE, and runs
   k-fold on the classifier head. This measures the generalization of the
   transfer-learned representation + head, reported as mean +/- std and 95% CI.

 Requires: tensorflow<2.16, numpy, scipy, scikit-learn, pandas, matplotlib, tqdm, Pillow
================================================================================
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR      = "./tiny-imagenet-200"
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_CLASSES   = 200
LEARNING_RATE = 1e-4
EPOCHS        = 12          # baseline / noise-aug stages (split across 2 stages for baseline)

# Attacks / perturbations (all in [0,1] image space)
EPS           = 8 / 255.0
PGD_STEP_SIZE = 2 / 255.0
PGD_STEPS     = 7
GAUSS_SIGMA   = 0.10

# Adversarial training (v3 recipe)
ADV_EPOCHS        = 10
ADV_LR            = 2e-5
ADV_WEIGHT        = 0.5       # loss = (1-w)*clean + w*adv
ADV_DROPOUT       = 0.2
ADV_RAMP_EPOCHS   = 3         # epsilon curriculum 0 -> EPS
ADV_FINE_TUNE_AT  = 0         # 0 = unfreeze whole backbone (BN still frozen)
ROBUST_VAL_BATCHES  = 20
ADV_EARLY_PATIENCE  = 3

# Evaluation
EVAL_BATCHES  = 30           # None = full 10k validation set

# Cross-validation
CV_FOLDS          = 10
CV_PER_CLASS      = 50       # (head mode) images/class sampled into the CV pool (<=500 avail in train)
CV_HEAD_EPOCHS    = 30
CV_HEAD_LR        = 1e-3
# Full end-to-end CV (trains the whole ResNet-50 per fold — very expensive)
CV_FULL_PER_CLASS = -1       # -1 = use ALL ~500 images/class; or cap per class for a cheaper run
CV_FULL_EPOCHS    = 12       # total per fold, split across 2 stages (head, then fine-tune) like the baseline

# Paths
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/qualitative", exist_ok=True)
BASELINE_PATH    = "models/resnet50_baseline.weights.h5"
NOISE_MODEL_PATH = "models/resnet50_noise_augmented.weights.h5"
MODEL_SAVE_PATH  = "models/resnet50_adversarial.weights.h5"

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

# current PGD budget (tf.Variable so the curriculum doesn't retrace the graph)
eps_var = tf.Variable(EPS, trainable=False, dtype=tf.float32)


def free_memory(*names):
    """Drop named globals, clear the TF/Keras graph, collect garbage (reclaim VRAM)."""
    g = globals()
    for n in names:
        g.pop(n, None)
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
    print(f"[free_memory] cleared session; dropped: {', '.join(names) if names else '(session only)'}")


# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_data():
    """Check dataset presence and reorganize the Tiny ImageNet validation split
    into per-class folders so ImageDataGenerator can read it."""
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"{DATA_DIR} not found. Download Tiny ImageNet:\n"
            "  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip")

    val_dir = os.path.join(DATA_DIR, "val")
    ann = os.path.join(val_dir, "val_annotations.txt")
    images_subdir = os.path.join(val_dir, "images")
    if os.path.exists(ann) and os.path.isdir(images_subdir):
        print("Reorganizing validation set into class folders ...")
        import shutil
        with open(ann) as f:
            mapping = {ln.split("\t")[0]: ln.split("\t")[1] for ln in f}
        for img, cls in mapping.items():
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(images_subdir, img)
            if os.path.exists(src):
                shutil.move(src, os.path.join(cls_dir, img))
        shutil.rmtree(images_subdir, ignore_errors=True)
        try:
            os.remove(ann)
        except OSError:
            pass
        print("Validation set reorganized.")
    else:
        print("Validation set already reorganized (or already in class-folder layout).")


def make_train_gen(noise_augment=False):
    kwargs = dict(rescale=1./255, horizontal_flip=True)
    datagen = ImageDataGenerator(preprocessing_function=_noise_preproc if noise_augment else None, **kwargs)
    return datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)


def make_val_gen():
    return ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "val"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)


def _noise_preproc(x):
    # preprocessing_function runs BEFORE rescale -> x is in [0,255]
    return add_gaussian_noise_batch(x[None], GAUSS_SIGMA, max_val=255.0)[0]


# =============================================================================
# MODEL
# =============================================================================
def build_model(trainable_base=False, fine_tune_at=None, dropout=0.5):
    """ResNet-50 with ImageNet preprocessing baked in; consumes [0,1] images."""
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
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs)


def build_feature_extractor():
    """[0,1] image -> 2048-d ResNet-50 GAP features (frozen). For CV feature caching."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    base = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_tensor=x)
    base.trainable = False
    return models.Model(inputs, base.output)


def freeze_batchnorm(model):
    n = 0
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            n += 1
    return n


def load_for_eval(path):
    m = build_model(fine_tune_at=140)
    m.load_weights(path)
    m.compile(optimizer=optimizers.Adam(LEARNING_RATE),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m


# =============================================================================
# PERTURBATIONS
# =============================================================================
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


@tf.function
def pgd_attack_curriculum(model, images, labels):
    """PGD using the global eps_var (for adversarial training's epsilon curriculum)."""
    eps = eps_var; alpha = eps / 4.0
    adv = images + tf.random.uniform(tf.shape(images), -eps, eps)
    adv = tf.clip_by_value(adv, 0.0, 1.0)
    for _ in range(PGD_STEPS):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            preds = model(adv, training=False)
            loss = tf.keras.losses.categorical_crossentropy(labels, preds)
        grads = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grads)
        adv = tf.clip_by_value(adv, images - eps, images + eps)
        adv = tf.clip_by_value(adv, 0.0, 1.0)
    return adv


# =============================================================================
# DEFENSE: TEST-TIME INPUT DENOISING
# =============================================================================
def denoise_batch(images, method="gaussian", param=1.0):
    from scipy.ndimage import gaussian_filter, median_filter
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


# =============================================================================
# ACCURACY METER + TRANSFORMS
# =============================================================================
def _to_np(x):
    return x.numpy() if hasattr(x, "numpy") else x


def measure_accuracy(model, val_gen, transform=None, max_batches=None, desc="eval"):
    correct = total = 0
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
    return lambda images, labels: fgsm_attack(model, tf.convert_to_tensor(images, tf.float32),
                                              tf.convert_to_tensor(labels, tf.float32), eps=eps)

def make_pgd_transform(model, eps=EPS, alpha=PGD_STEP_SIZE, steps=PGD_STEPS):
    return lambda images, labels: pgd_attack(model, tf.convert_to_tensor(images, tf.float32),
                                             tf.convert_to_tensor(labels, tf.float32),
                                             eps=eps, alpha=alpha, steps=steps)

def make_pgd_then_denoise_transform(model, method="gaussian", param=1.0, **pgd_kwargs):
    pgd_t = make_pgd_transform(model, **pgd_kwargs)
    return lambda images, labels: denoise_batch(_to_np(pgd_t(images, labels)), method, param)


# =============================================================================
# TRAINING: BASELINE (two-stage transfer)
# =============================================================================
def train_baseline():
    print("\n" + "="*60 + "\nTraining BASELINE ResNet-50 (Clean)\n" + "="*60)
    train_gen = make_train_gen(noise_augment=False)
    val_gen   = make_val_gen()

    # Stage 1: train the head
    print("Stage 1: training classification head ...")
    model = build_model(trainable_base=False)
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss="categorical_crossentropy",
                  metrics=["accuracy", "top_k_categorical_accuracy"])
    ckpt1 = callbacks.ModelCheckpoint(BASELINE_PATH, monitor="val_accuracy",
                                      save_best_only=True, save_weights_only=True, verbose=1)
    rlr1 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    h1 = model.fit(train_gen, epochs=EPOCHS // 2, validation_data=val_gen,
                   callbacks=[ckpt1, rlr1], verbose=1)

    # Stage 2: fine-tune upper backbone
    print("\nStage 2: fine-tuning later layers of ResNet ...")
    stage1_best = max(h1.history["val_accuracy"])
    model = build_model(fine_tune_at=140)
    model.load_weights(BASELINE_PATH)
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE / 10), loss="categorical_crossentropy",
                  metrics=["accuracy", "top_k_categorical_accuracy"])
    ckpt2 = callbacks.ModelCheckpoint(BASELINE_PATH, monitor="val_accuracy", save_best_only=True,
                                      save_weights_only=True, initial_value_threshold=stage1_best, verbose=1)
    rlr2 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    model.fit(train_gen, epochs=EPOCHS // 2, validation_data=val_gen,
              callbacks=[ckpt2, rlr2], verbose=1)
    print(f"Baseline saved to {BASELINE_PATH}")
    free_memory("model", "train_gen", "val_gen")


# =============================================================================
# TRAINING: NOISE-AUGMENTED
# =============================================================================
def train_noise_augmented():
    print("\n" + "="*60 + f"\nTraining Noise-Augmented Model (sigma={GAUSS_SIGMA})\n" + "="*60)
    train_gen = make_train_gen(noise_augment=True)
    val_gen   = make_val_gen()
    model = build_model(fine_tune_at=140)
    if os.path.exists(BASELINE_PATH):
        model.load_weights(BASELINE_PATH)      # warm-start from clean baseline
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE / 10), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    ckpt = callbacks.ModelCheckpoint(NOISE_MODEL_PATH, monitor="val_accuracy",
                                     save_best_only=True, save_weights_only=True, verbose=1)
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=[ckpt], verbose=1)
    print(f"Noise-augmented model saved to {NOISE_MODEL_PATH}")
    free_memory("model", "train_gen", "val_gen")


# =============================================================================
# TRAINING: ADVERSARIAL (v3 recipe — frozen BN + full backbone + eps curriculum)
# =============================================================================
@tf.function
def adv_train_step(model, optimizer, images, labels):
    adv_images = pgd_attack_curriculum(model, images, labels)
    with tf.GradientTape() as tape:
        clean_preds = model(images, training=True)
        adv_preds   = model(adv_images, training=True)
        clean_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, clean_preds))
        adv_loss   = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, adv_preds))
        loss = (1.0 - ADV_WEIGHT) * clean_loss + ADV_WEIGHT * adv_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    clean_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(clean_preds,1), tf.argmax(labels,1)), tf.float32))
    adv_acc   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(adv_preds,1),   tf.argmax(labels,1)), tf.float32))
    return loss, clean_acc, adv_acc


def _robust_val(model, val_gen, max_batches):
    saved = float(eps_var.numpy()); eps_var.assign(EPS)
    n = min(max_batches, len(val_gen)); c = r = tot = 0
    for i in tqdm(range(n), desc="val", leave=False):
        images, labels = val_gen[i]
        t_img = tf.convert_to_tensor(images, tf.float32); t_lab = tf.convert_to_tensor(labels, tf.float32)
        clean_pred = model(t_img, training=False).numpy()
        adv = pgd_attack_curriculum(model, t_img, t_lab).numpy()
        adv_pred = model(tf.convert_to_tensor(adv, tf.float32), training=False).numpy()
        y = np.argmax(labels, 1)
        c += int(np.sum(np.argmax(clean_pred,1)==y)); r += int(np.sum(np.argmax(adv_pred,1)==y)); tot += len(labels)
    eps_var.assign(saved)
    return c/tot, r/tot


def train_adversarial():
    print("\n" + "="*60 + "\nTraining ADVERSARIAL Model (v3: frozen BN + full backbone + eps curriculum)\n" + "="*60)
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError("Baseline weights required; run --stage baseline first.")
    model = build_model(fine_tune_at=ADV_FINE_TUNE_AT, dropout=ADV_DROPOUT)
    model.load_weights(BASELINE_PATH)
    n_bn = freeze_batchnorm(model)
    print(f"  froze {n_bn} BatchNorm layers; {len(model.trainable_variables)} trainable tensors")
    optimizer = optimizers.Adam(ADV_LR)

    train_gen = make_train_gen(noise_augment=False)
    val_gen   = make_val_gen()
    steps = len(train_gen); best_robust = -1.0; patience = 0

    for epoch in range(ADV_EPOCHS):
        cur_eps = EPS * min(1.0, (epoch + 1) / ADV_RAMP_EPOCHS)
        eps_var.assign(cur_eps)
        print(f"\n=== Epoch {epoch+1}/{ADV_EPOCHS}  (train eps={cur_eps*255:.1f}/255) ===")
        run = dict(loss=0., clean=0., adv=0.)
        pbar = tqdm(range(steps), desc="train")
        for step in pbar:
            images, labels = next(train_gen)
            loss, c_acc, a_acc = adv_train_step(model, optimizer,
                tf.convert_to_tensor(images, tf.float32), tf.convert_to_tensor(labels, tf.float32))
            run["loss"] += float(loss); run["clean"] += float(c_acc); run["adv"] += float(a_acc)
            if (step+1) % 50 == 0:
                d = step+1
                pbar.set_postfix(loss=f"{run['loss']/d:.3f}", clean=f"{run['clean']/d:.3f}", adv=f"{run['adv']/d:.3f}")
        clean_val, robust_val = _robust_val(model, val_gen, ROBUST_VAL_BATCHES)
        print(f"  clean_val={clean_val:.4f} | robust_val(PGD@{EPS*255:.0f}/255)={robust_val:.4f}")
        if epoch + 1 >= ADV_RAMP_EPOCHS:
            if robust_val > best_robust:
                best_robust = robust_val; patience = 0
                model.save_weights(MODEL_SAVE_PATH); print(f"  -> saved best (robust_val={robust_val:.4f})")
            else:
                patience += 1; print(f"  no robust improvement ({patience}/{ADV_EARLY_PATIENCE})")
                if patience >= ADV_EARLY_PATIENCE:
                    print("  early stopping."); break
    if best_robust < 0:
        model.save_weights(MODEL_SAVE_PATH)
    print(f"\nAdversarial model saved to {MODEL_SAVE_PATH} (best robust val={max(best_robust,0):.4f})")
    free_memory("model", "optimizer", "train_gen", "val_gen")


# =============================================================================
# EXPLAINABILITY: Grad-CAM / Score-CAM
# =============================================================================
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("no 4D conv layer found")


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
    if masked.shape[0] == 0:
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
        drops.append(max(0.0, Yc - Oc) / (Yc + 1e-8)); incs.append(1.0 if Oc > Yc else 0.0)
        noisy = add_gaussian_noise_batch(img[None], stab_sigma)[0]
        cam_n, _ = cam_fn(model, noisy, class_index=c)
        stabs.append(np.corrcoef(cam.flatten(), cam_n.flatten())[0, 1])
    return {"avg_drop_%": 100*float(np.mean(drops)),
            "increase_in_confidence_%": 100*float(np.mean(incs)),
            "stability_corr": float(np.nanmean(stabs))}


# =============================================================================
# EVALUATION
# =============================================================================
def run_evaluation():
    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    val_gen = make_val_gen()
    model_paths = {"Baseline": BASELINE_PATH, "Noise-Augmented": NOISE_MODEL_PATH, "Adversarial": MODEL_SAVE_PATH}

    # Robustness table
    rows = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing"); continue
        print(f"\nEvaluating: {name}")
        m = load_for_eval(path)
        r = {}
        r["clean"]      = measure_accuracy(m, val_gen, None, EVAL_BATCHES, "clean")
        r["gauss_0.05"] = measure_accuracy(m, val_gen, make_noise_transform(0.05), EVAL_BATCHES, "g05")
        r["gauss_0.10"] = measure_accuracy(m, val_gen, make_noise_transform(0.10), EVAL_BATCHES, "g10")
        r["gauss_0.20"] = measure_accuracy(m, val_gen, make_noise_transform(0.20), EVAL_BATCHES, "g20")
        r["fgsm"]       = measure_accuracy(m, val_gen, make_fgsm_transform(m), EVAL_BATCHES, "fgsm")
        r["pgd"]        = measure_accuracy(m, val_gen, make_pgd_transform(m), EVAL_BATCHES, "pgd")
        rows[name] = r
        free_memory("m")
    robustness_df = pd.DataFrame(rows).T
    robustness_df.to_csv("results/robustness_comparison.csv")
    print("\n===== ROBUSTNESS COMPARISON =====\n", robustness_df.round(4))

    # Denoising defense
    drows = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            continue
        m = load_for_eval(path)
        pgd_only = measure_accuracy(m, val_gen, make_pgd_transform(m), EVAL_BATCHES, f"{name}:pgd")
        pgd_den  = measure_accuracy(m, val_gen, make_pgd_then_denoise_transform(m), EVAL_BATCHES, f"{name}:pgd+den")
        drows[name] = {"pgd": pgd_only, "pgd+denoise": pgd_den, "recovered": pgd_den - pgd_only}
        free_memory("m")
    denoise_df = pd.DataFrame(drows).T
    denoise_df.to_csv("results/denoising_defense.csv")
    print("\n===== DENOISING DEFENSE =====\n", denoise_df.round(4))

    # Explanation metrics (on baseline)
    if os.path.exists(BASELINE_PATH):
        m = load_for_eval(BASELINE_PATH)
        expl = {"Grad-CAM":  explanation_metrics(m, val_gen, compute_gradcam,  n_images=40),
                "Score-CAM": explanation_metrics(m, val_gen, compute_scorecam, n_images=15)}
        expl_df = pd.DataFrame(expl).T
        expl_df.to_csv("results/explanation_metrics.csv")
        print("\n===== EXPLANATION METRICS =====\n", expl_df.round(3))
        free_memory("m")

    # PGD ablation (adversarial model)
    if os.path.exists(MODEL_SAVE_PATH):
        m = load_for_eval(MODEL_SAVE_PATH)
        print("\nPGD epsilon sweep (steps=7):")
        for e in [2/255., 4/255., 8/255., 16/255.]:
            acc = measure_accuracy(m, val_gen, make_pgd_transform(m, eps=e, steps=7), 15, f"eps={e*255:.0f}")
            print(f"  eps={e*255:.0f}/255 : {acc:.4f}")
        print("PGD steps sweep (eps=8/255):")
        for s in [1, 7, 10]:
            acc = measure_accuracy(m, val_gen, make_pgd_transform(m, steps=s), 15, f"steps={s}")
            print(f"  steps={s} : {acc:.4f}")
        free_memory("m")
    free_memory("val_gen")


# =============================================================================
# QUALITATIVE GALLERIES
# =============================================================================
def run_qualitative(n_samples=3, scan_batches=60):
    print("\n" + "="*60 + "\nQUALITATIVE RESULTS\n" + "="*60)
    for p in (BASELINE_PATH, NOISE_MODEL_PATH, MODEL_SAVE_PATH):
        if not os.path.exists(p):
            print(f"[skip] qualitative: missing {p}"); return
    val_gen = make_val_gen()
    baseline = load_for_eval(BASELINE_PATH)
    noise = load_for_eval(NOISE_MODEL_PATH)
    adv = load_for_eval(MODEL_SAVE_PATH)

    def perturb(tag, images, labels, attacker):
        if tag == "gaussian":
            return add_gaussian_noise_batch(images, GAUSS_SIGMA)
        ti = tf.convert_to_tensor(images, tf.float32); tl = tf.convert_to_tensor(labels, tf.float32)
        return (fgsm_attack(attacker, ti, tl) if tag == "fgsm" else pgd_attack(attacker, ti, tl)).numpy()

    conditions = [("gaussian", noise, "Gaussian σ=0.10 → Noise-Augmented", "gaussian_recovery.png"),
                  ("fgsm", adv, "FGSM → Adversarial", "fgsm_recovery.png"),
                  ("pgd", adv, "PGD → Adversarial", "pgd_recovery.png")]
    for tag, defense, title, fname in conditions:
        collected, broken, recovered = [], 0, 0
        for bi in range(min(scan_batches, len(val_gen))):
            images, labels = val_gen[bi]; true = np.argmax(labels, 1)
            bc = np.argmax(baseline(tf.convert_to_tensor(images, tf.float32), training=False).numpy(), 1)
            pert = perturb(tag, images, labels, baseline)
            bp = np.argmax(baseline(tf.convert_to_tensor(pert, tf.float32), training=False).numpy(), 1)
            dp = np.argmax(defense(tf.convert_to_tensor(pert, tf.float32), training=False).numpy(), 1)
            dc = np.argmax(defense(tf.convert_to_tensor(images, tf.float32), training=False).numpy(), 1)
            broke = (bc == true) & (bp != true); broken += int(broke.sum())
            recovered += int((broke & (dp == true)).sum())
            story = broke & (dp == true) & (dc == true)
            for i in np.where(story)[0]:
                collected.append((images[i], pert[i], int(true[i])))
                if len(collected) >= n_samples:
                    break
            if len(collected) >= n_samples:
                break
        rate = (recovered / broken) if broken else float("nan")
        print(f"{title}: baseline broke {broken}, defense recovered {recovered} ({rate:.1%}) [transfer, subset]")
        if collected:
            _plot_recovery(collected, baseline, defense, title, f"results/qualitative/{fname}")
    free_memory("baseline", "noise", "adv", "val_gen")


def _plot_recovery(samples, baseline, defense, title, save_path):
    cols = ["Clean / Baseline", "Perturbed / Baseline", "Clean / Defended", "Perturbed / Defended"]
    fig, axes = plt.subplots(len(samples), 4, figsize=(9, 2.4 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]
    for r, (clean, pert, true) in enumerate(samples):
        panels = [(baseline, clean), (baseline, pert), (defense, clean), (defense, pert)]
        for c, (mdl, img) in enumerate(panels):
            p = mdl(tf.convert_to_tensor(img[None], tf.float32), training=False).numpy()[0]
            pred = int(np.argmax(p)); ok = (pred == true)
            ax = axes[r, c]; ax.imshow(np.clip(img, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{cols[c]}\n{'✓' if ok else '✗'} pred={pred} ({p[pred]:.0%})",
                         fontsize=8, color=("#15803d" if ok else "#b91c1c"))
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {save_path}")


# =============================================================================
# *** K-FOLD CROSS-VALIDATION (feature-cached head CV) ***
# =============================================================================
def _gather_cv_pool(per_class):
    """Return (paths, labels) sampled per class from the training directory."""
    train_dir = os.path.join(DATA_DIR, "train")
    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    rng = np.random.default_rng(SEED)
    for c in classes:
        # Tiny ImageNet train images live under <class>/images/*.JPEG
        cdir = os.path.join(train_dir, c, "images")
        if not os.path.isdir(cdir):
            cdir = os.path.join(train_dir, c)
        files = [f for f in os.listdir(cdir) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
        if not files:
            continue
        pick = rng.choice(files, size=min(per_class, len(files)), replace=False)
        for f in pick:
            paths.append(os.path.join(cdir, f)); labels.append(class_to_idx[c])
    return np.array(paths), np.array(labels), classes


def _precompute_features(paths):
    """Forward every image through the frozen backbone ONCE -> (N, 2048) features."""
    extractor = build_feature_extractor()
    feats = np.zeros((len(paths), 2048), dtype=np.float32)
    bs = 64
    for i in tqdm(range(0, len(paths), bs), desc="caching features"):
        batch_paths = paths[i:i+bs]
        imgs = np.stack([
            img_to_array(load_img(p, target_size=(IMG_SIZE, IMG_SIZE))) / 255.0
            for p in batch_paths
        ]).astype(np.float32)
        feats[i:i+bs] = extractor(tf.convert_to_tensor(imgs), training=False).numpy()
    free_memory("extractor")
    return feats


def run_cross_validation(folds=CV_FOLDS, per_class=CV_PER_CLASS, head_epochs=CV_HEAD_EPOCHS):
    """Stratified K-fold CV on the classifier head over cached backbone features.
    Reports per-fold top-1 accuracy, mean, std, and a 95% confidence interval."""
    from sklearn.model_selection import StratifiedKFold
    from scipy import stats
    print("\n" + "="*60 + f"\n{folds}-FOLD CROSS-VALIDATION (feature-cached head)\n" + "="*60)

    paths, labels, classes = _gather_cv_pool(per_class)
    print(f"CV pool: {len(paths)} images across {len(classes)} classes "
          f"({per_class}/class requested). Caching backbone features ...")
    feats = _precompute_features(paths)                         # (N, 2048), computed once
    y = labels.astype(np.int64)
    y_oh = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_acc = []
    for fold, (tr, va) in enumerate(skf.split(feats, y), 1):
        head = models.Sequential([
            keras.Input(shape=(2048,)),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ])
        head.compile(optimizer=optimizers.Adam(CV_HEAD_LR),
                     loss="categorical_crossentropy", metrics=["accuracy"])
        es = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
        head.fit(feats[tr], y_oh[tr], validation_data=(feats[va], y_oh[va]),
                 epochs=head_epochs, batch_size=128, verbose=0, callbacks=[es])
        acc = head.evaluate(feats[va], y_oh[va], verbose=0)[1]
        fold_acc.append(acc)
        print(f"  Fold {fold:2d}/{folds}:  val accuracy = {acc:.4f}")
        tf.keras.backend.clear_session()

    fold_acc = np.array(fold_acc)
    mean, std = fold_acc.mean(), fold_acc.std(ddof=1)
    # 95% CI via Student's t (small sample)
    ci = stats.t.interval(0.95, len(fold_acc)-1, loc=mean, scale=std/np.sqrt(len(fold_acc)))
    print("\n----- CROSS-VALIDATION SUMMARY -----")
    print(f"  folds            : {folds}")
    print(f"  per-fold accuracy: {', '.join(f'{a:.4f}' for a in fold_acc)}")
    print(f"  mean +/- std     : {mean:.4f} +/- {std:.4f}")
    print(f"  95% CI           : [{ci[0]:.4f}, {ci[1]:.4f}]")

    df = pd.DataFrame({"fold": np.arange(1, folds+1), "val_accuracy": fold_acc})
    df.loc[len(df)] = ["mean", mean]; df.loc[len(df)] = ["std", std]
    df.loc[len(df)] = ["ci95_low", ci[0]]; df.loc[len(df)] = ["ci95_high", ci[1]]
    df.to_csv("results/cross_validation.csv", index=False)

    # Bar chart with error bar
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(np.arange(1, folds+1), fold_acc, color="#0f766e", alpha=0.85)
    ax.axhline(mean, color="#b91c1c", ls="--", lw=1.5, label=f"mean={mean:.3f}")
    ax.fill_between([0.5, folds+0.5], ci[0], ci[1], color="#b91c1c", alpha=0.12, label="95% CI")
    ax.set_xlabel("Fold"); ax.set_ylabel("Top-1 accuracy"); ax.set_xticks(np.arange(1, folds+1))
    ax.set_ylim(0, max(0.8, fold_acc.max()+0.1)); ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"{folds}-fold cross-validation (classifier head on frozen features)",
                 fontsize=10, color="#0f172a", weight="bold")
    plt.tight_layout(); plt.savefig("results/cross_validation.png", dpi=200); plt.close()
    print("  saved -> results/cross_validation.csv and results/cross_validation.png")
    free_memory()


# ----- FULL end-to-end K-fold CV (trains the whole ResNet-50 per fold) -----
def _make_fold_dataset(paths, labels, training):
    """tf.data pipeline: file path -> [0,1] 224x224x3 image, one-hot label.
    The model does ImageNet preprocessing internally, so images stay in [0,1]."""
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)          # Tiny ImageNet is JPEG; forces 3 channels
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = tf.image.random_flip_left_right(img)
        return img, tf.one_hot(label, NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def run_cross_validation_full(folds=CV_FOLDS, per_class=CV_FULL_PER_CLASS, total_epochs=CV_FULL_EPOCHS):
    """FULL end-to-end stratified K-fold CV: trains a complete ResNet-50 (two-stage
    transfer, fresh from ImageNet each fold) on the fold's training split and
    evaluates top-1 on the held-out fold. VERY expensive (~10x a full training)."""
    from sklearn.model_selection import StratifiedKFold
    from scipy import stats
    print("\n" + "="*60 + f"\n{folds}-FOLD CROSS-VALIDATION (FULL end-to-end ResNet-50)\n" + "="*60)
    pc = None if (per_class is None or per_class < 0) else per_class
    paths, labels, classes = _gather_cv_pool(pc if pc is not None else 10**9)
    s1 = max(1, total_epochs // 2); s2 = max(1, total_epochs - s1)
    print(f"CV pool: {len(paths)} images / {len(classes)} classes "
          f"({'ALL' if pc is None else pc}/class). {folds} folds x ({s1}+{s2}) epochs.")
    print("WARNING: this trains the full network 10 times and can take many GPU-hours.")

    y = labels.astype(np.int64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_acc = []
    for fold, (tr, va) in enumerate(skf.split(paths, y), 1):
        print(f"\n--- Fold {fold}/{folds}  (train={len(tr)}, val={len(va)}) ---")
        train_ds = _make_fold_dataset(paths[tr], y[tr], training=True)
        val_ds   = _make_fold_dataset(paths[va], y[va], training=False)

        # Stage 1: train head
        model = build_model(trainable_base=False)
        model.compile(optimizer=optimizers.Adam(LEARNING_RATE),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=s1, verbose=2)
        w = model.get_weights()

        # Stage 2: fine-tune upper backbone (rebuild with fine_tune_at, carry weights)
        tf.keras.backend.clear_session()
        model = build_model(fine_tune_at=140)
        model.set_weights(w)
        model.compile(optimizer=optimizers.Adam(LEARNING_RATE / 10),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=s2, verbose=2)

        acc = model.evaluate(val_ds, verbose=0)[1]
        fold_acc.append(acc)
        print(f"  Fold {fold}/{folds}: held-out top-1 = {acc:.4f}")
        free_memory("model", "train_ds", "val_ds", "w")

    fold_acc = np.array(fold_acc)
    mean, std = fold_acc.mean(), fold_acc.std(ddof=1)
    ci = stats.t.interval(0.95, len(fold_acc)-1, loc=mean, scale=std/np.sqrt(len(fold_acc)))
    print("\n----- FULL CROSS-VALIDATION SUMMARY -----")
    print(f"  per-fold accuracy: {', '.join(f'{a:.4f}' for a in fold_acc)}")
    print(f"  mean +/- std     : {mean:.4f} +/- {std:.4f}")
    print(f"  95% CI           : [{ci[0]:.4f}, {ci[1]:.4f}]")

    df = pd.DataFrame({"fold": np.arange(1, folds+1), "val_accuracy": fold_acc})
    df.loc[len(df)] = ["mean", mean]; df.loc[len(df)] = ["std", std]
    df.loc[len(df)] = ["ci95_low", ci[0]]; df.loc[len(df)] = ["ci95_high", ci[1]]
    df.to_csv("results/cross_validation_full.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(np.arange(1, folds+1), fold_acc, color="#6d28d9", alpha=0.85)
    ax.axhline(mean, color="#b91c1c", ls="--", lw=1.5, label=f"mean={mean:.3f}")
    ax.fill_between([0.5, folds+0.5], ci[0], ci[1], color="#b91c1c", alpha=0.12, label="95% CI")
    ax.set_xlabel("Fold"); ax.set_ylabel("Held-out top-1 accuracy"); ax.set_xticks(np.arange(1, folds+1))
    ax.set_ylim(0, max(0.8, fold_acc.max()+0.1)); ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"{folds}-fold cross-validation (full end-to-end ResNet-50)",
                 fontsize=10, color="#0f172a", weight="bold")
    plt.tight_layout(); plt.savefig("results/cross_validation_full.png", dpi=200); plt.close()
    print("  saved -> results/cross_validation_full.csv and results/cross_validation_full.png")
    free_memory()


# =============================================================================
# MAIN
# =============================================================================
STAGES = ["data", "baseline", "noise", "adversarial", "eval", "qualitative", "cv"]

def main():
    global EVAL_BATCHES
    ap = argparse.ArgumentParser(description="Adversarial robustness pipeline (CSI 5170 final project).")
    ap.add_argument("--stage", default="all",
                    help="comma-separated: " + ",".join(STAGES) + ",all")
    ap.add_argument("--folds", type=int, default=CV_FOLDS, help="K for cross-validation")
    ap.add_argument("--cv-mode", choices=["head", "full"], default="head",
                    help="'head' = fast feature-cached head CV; 'full' = train the whole ResNet-50 per fold (very slow)")
    ap.add_argument("--cv-per-class", type=int, default=CV_PER_CLASS,
                    help="images/class in the CV pool (head mode). For full mode, -1 uses all ~500/class")
    ap.add_argument("--cv-epochs", type=int, default=CV_FULL_EPOCHS,
                    help="full-mode: total epochs per fold, split across 2 stages")
    ap.add_argument("--eval-batches", type=int,
                    default=(-1 if EVAL_BATCHES is None else EVAL_BATCHES),
                    help="val batches per eval metric (use -1 for the full set)")
    args = ap.parse_args()

    EVAL_BATCHES = None if args.eval_batches == -1 else args.eval_batches

    stages = STAGES if args.stage.strip() == "all" else [s.strip() for s in args.stage.split(",")]
    print("Running stages:", stages)

    if "data" in stages or "all" in stages or any(s in stages for s in STAGES):
        prepare_data()
    if "baseline" in stages:     train_baseline()
    if "noise" in stages:        train_noise_augmented()
    if "adversarial" in stages:  train_adversarial()
    if "eval" in stages:         run_evaluation()
    if "qualitative" in stages:  run_qualitative()
    if "cv" in stages:
        if args.cv_mode == "full":
            cap = -1 if args.cv_per_class == CV_PER_CLASS else args.cv_per_class  # default -> all classes
            run_cross_validation_full(args.folds, cap, args.cv_epochs)
        else:
            run_cross_validation(args.folds, args.cv_per_class)
    print("\nDone.")


if __name__ == "__main__":
    main()
