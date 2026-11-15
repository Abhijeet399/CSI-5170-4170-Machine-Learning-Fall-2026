"""
retrain_adversarial_v3.py
======================================================================
Third-attempt adversarial training. Regenerates ONLY
    models/resnet50_adversarial.weights.h5

Diagnosis behind the changes
----------------------------
Run 1 (BN trainable):  robustness crept up but CLEAN accuracy collapsed
                       (BN running stats drifted onto adversarial images).
Run 2 (BN frozen, top block only): clean preserved, but NO robustness and
                       the model got hyper-brittle to tiny noise -- because
                       with BN frozen AND only the last block trainable there
                       was no capacity to learn robust features, so it just
                       sharpened the boundary around clean points.

v3 keeps the good half of run 2 and fixes the bad half:
  * BN stays FROZEN  -> clean accuracy is protected.
  * The WHOLE backbone is unfrozen (fine_tune_at=0) -> real capacity to
    learn robust features that survive the attack after clean normalization.
  * Dropout lowered (0.2) -> doesn't fight the adversarial signal.
  * EPSILON CURRICULUM -> the PGD budget ramps 0 -> EPS over the first
    RAMP_EPOCHS, so early epochs are learnable instead of hopeless.
  * Robust validation + checkpoint are always measured at FULL EPS.

Warm-starts from the clean baseline. Run it, then re-run eval_only.py.
Heavier than before (full backbone + PGD): budget ~25-30 min/epoch.
======================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_DIR      = "./tiny-imagenet-200"
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_CLASSES   = 200

EPOCHS            = 10
RAMP_EPOCHS       = 3          # epsilon ramps 0 -> EPS over this many epochs
LEARNING_RATE     = 2e-5
ADV_WEIGHT        = 0.5        # loss = (1-w)*clean + w*adv
DROPOUT           = 0.2
EPS               = 8 / 255.0  # full attack budget (train target + eval)
PGD_STEPS         = 7
FINE_TUNE_AT      = 0          # 0 = unfreeze the ENTIRE backbone (BN still frozen)
ROBUST_VAL_BATCHES = 20
EARLY_STOP_PATIENCE = 3
# If robustness still won't move after a few epochs, drop EPS to 4/255 and
# re-run (also change EPS in eval_only.py so you evaluate at the same budget).

BASELINE_PATH = next((f"models/resnet50_baseline{e}"
                      for e in (".weights.h5", ".h5")
                      if os.path.exists(f"models/resnet50_baseline{e}")), None)
MODEL_SAVE_PATH = "models/resnet50_adversarial.weights.h5"

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
os.makedirs("models", exist_ok=True)
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

# current attack budget, updated each epoch (tf.Variable -> no retracing)
eps_var = tf.Variable(EPS, trainable=False, dtype=tf.float32)


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------
def build_model(fine_tune_at=0, dropout=DROPOUT):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    base = ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:        # fine_tune_at=0 freezes nothing
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs)


def freeze_batchnorm(model):
    n = 0
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            n += 1
    return n


# ----------------------------------------------------------------------
# PGD (uses eps_var; alpha scales with the current budget)
# ----------------------------------------------------------------------
@tf.function
def pgd_attack(model, images, labels):
    eps = eps_var
    alpha = eps / 4.0
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


@tf.function
def train_step(model, optimizer, images, labels):
    adv_images = pgd_attack(model, images, labels)
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


def val_accuracies(model, val_gen, max_batches):
    """Clean accuracy + robust accuracy measured at FULL eps."""
    saved = float(eps_var.numpy())
    eps_var.assign(EPS)                      # always validate at the real budget
    n = min(max_batches, len(val_gen))
    c_ok = r_ok = total = 0
    for i in tqdm(range(n), desc="val", leave=False):
        images, labels = val_gen[i]
        t_img = tf.convert_to_tensor(images, tf.float32)
        t_lab = tf.convert_to_tensor(labels, tf.float32)
        clean_pred = model(t_img, training=False).numpy()
        adv = pgd_attack(model, t_img, t_lab).numpy()
        adv_pred = model(tf.convert_to_tensor(adv, tf.float32), training=False).numpy()
        y = np.argmax(labels, 1)
        c_ok += int(np.sum(np.argmax(clean_pred, 1) == y))
        r_ok += int(np.sum(np.argmax(adv_pred, 1) == y))
        total += len(labels)
    eps_var.assign(saved)
    return c_ok / total, r_ok / total


# ----------------------------------------------------------------------
def main():
    if BASELINE_PATH is None:
        raise FileNotFoundError("baseline weights not found; train the baseline first.")

    print("Building adversarial model (full backbone trainable, BN frozen) ...")
    model = build_model(fine_tune_at=FINE_TUNE_AT, dropout=DROPOUT)
    model.load_weights(BASELINE_PATH)        # dropout has no weights -> loads fine
    n_bn = freeze_batchnorm(model)
    print(f"  froze {n_bn} BatchNorm layers; {len(model.trainable_variables)} trainable weight tensors")

    optimizer = optimizers.Adam(LEARNING_RATE)

    train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True).flow_from_directory(
        os.path.join(DATA_DIR, "train"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "val"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

    steps = len(train_gen)
    best_robust, patience = -1.0, 0

    for epoch in range(EPOCHS):
        # epsilon curriculum: ramp 0 -> EPS across the first RAMP_EPOCHS
        cur_eps = EPS * min(1.0, (epoch + 1) / RAMP_EPOCHS)
        eps_var.assign(cur_eps)
        print(f"\n=== Epoch {epoch+1}/{EPOCHS}  (train eps = {cur_eps*255:.1f}/255) ===")

        run = dict(loss=0.0, clean=0.0, adv=0.0)
        pbar = tqdm(range(steps), desc="train")
        for step in pbar:
            images, labels = next(train_gen)
            loss, c_acc, a_acc = train_step(
                model, optimizer,
                tf.convert_to_tensor(images, tf.float32),
                tf.convert_to_tensor(labels, tf.float32))
            run["loss"] += float(loss); run["clean"] += float(c_acc); run["adv"] += float(a_acc)
            if (step + 1) % 50 == 0:
                d = step + 1
                pbar.set_postfix(loss=f"{run['loss']/d:.3f}",
                                 clean=f"{run['clean']/d:.3f}", adv=f"{run['adv']/d:.3f}")

        clean_val, robust_val = val_accuracies(model, val_gen, ROBUST_VAL_BATCHES)
        print(f"  clean_val={clean_val:.4f} | robust_val(PGD@{EPS*255:.0f}/255)={robust_val:.4f}")

        # only start checkpointing once we are training at (near) full strength
        if epoch + 1 >= RAMP_EPOCHS:
            if robust_val > best_robust:
                best_robust = robust_val; patience = 0
                model.save_weights(MODEL_SAVE_PATH)
                print(f"  -> saved best (robust_val={robust_val:.4f})")
            else:
                patience += 1
                print(f"  no robust improvement ({patience}/{EARLY_STOP_PATIENCE})")
                if patience >= EARLY_STOP_PATIENCE:
                    print("  early stopping."); break

    if best_robust < 0:      # never saved (e.g. ramp not finished) -> save final state
        model.save_weights(MODEL_SAVE_PATH)
        print("  saved final model (no full-strength checkpoint beat the init).")
    print(f"\nDone. Best robust val accuracy @ {EPS*255:.0f}/255 = {max(best_robust,0):.4f}")
    print(f"Saved to {MODEL_SAVE_PATH}. Re-run eval_only.py.")


if __name__ == "__main__":
    main()
