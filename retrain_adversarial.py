"""
retrain_adversarial.py
======================================================================
Re-trains ONLY the adversarially-robust model and overwrites
    models/resnet50_adversarial.weights.h5
Baseline and noise-augmented models are not touched.

Fixes applied vs the diverged run:
  1. BatchNorm FROZEN during adversarial fine-tuning. A frozen BN layer
     (trainable=False) stays in inference mode even under training=True,
     so its running mean/var keep the CLEAN statistics from the baseline
     instead of drifting toward adversarial-image statistics. This is the
     main cause of the clean-accuracy collapse you saw.
  2. MIXED clean + adversarial loss (not adversarial-only), so clean
     accuracy is preserved while robustness is learned.
  3. Checkpoint on ROBUST (PGD) validation accuracy, with early stopping,
     instead of clean accuracy — so the saved model is the most robust one.

Warm-starts from the clean baseline weights. Run, then re-run eval_only.py.
Deps: pip install tqdm numpy tensorflow
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

# adversarial-training hyper-parameters
EPOCHS            = 6          # with early stopping this rarely runs to the end
LEARNING_RATE     = 1e-5       # low LR for stable fine-tuning
ADV_WEIGHT        = 0.5        # loss = (1-w)*clean + w*adv
EPS               = 8 / 255.0
PGD_STEP_SIZE     = 2 / 255.0
PGD_STEPS         = 7
FINE_TUNE_AT      = 140        # unfreeze conv layers from here (BN still frozen)
ROBUST_VAL_BATCHES = 20        # PGD-eval this many val batches each epoch (~640 imgs)
EARLY_STOP_PATIENCE = 2        # stop if robust val acc doesn't improve for N epochs

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


# ----------------------------------------------------------------------
# MODEL
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


def freeze_batchnorm(model):
    """Keep every BatchNorm in inference mode so its running stats stay clean."""
    n = 0
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            n += 1
    return n


# ----------------------------------------------------------------------
# PGD ATTACK  (operates in [0,1])
# ----------------------------------------------------------------------
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
# ONE TRAINING STEP  (mixed clean + adversarial loss)
# ----------------------------------------------------------------------
@tf.function
def train_step(model, optimizer, images, labels):
    adv_images = pgd_attack(model, images, labels)          # BN frozen -> inference mode
    with tf.GradientTape() as tape:
        clean_preds = model(images, training=True)          # dropout on; BN frozen
        adv_preds   = model(adv_images, training=True)
        clean_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(labels, clean_preds))
        adv_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(labels, adv_preds))
        loss = (1.0 - ADV_WEIGHT) * clean_loss + ADV_WEIGHT * adv_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    clean_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(clean_preds, 1), tf.argmax(labels, 1)), tf.float32))
    adv_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(adv_preds, 1), tf.argmax(labels, 1)), tf.float32))
    return loss, clean_acc, adv_acc


# ----------------------------------------------------------------------
# VALIDATION (clean + robust)
# ----------------------------------------------------------------------
def val_accuracies(model, val_gen, max_batches):
    n = min(max_batches, len(val_gen))
    c_correct = r_correct = total = 0
    for i in tqdm(range(n), desc="val", leave=False):
        images, labels = val_gen[i]
        t_img = tf.convert_to_tensor(images, tf.float32)
        t_lab = tf.convert_to_tensor(labels, tf.float32)
        clean_pred = model(t_img, training=False).numpy()
        adv = pgd_attack(model, t_img, t_lab).numpy()
        adv_pred = model(tf.convert_to_tensor(adv, tf.float32), training=False).numpy()
        y = np.argmax(labels, 1)
        c_correct += int(np.sum(np.argmax(clean_pred, 1) == y))
        r_correct += int(np.sum(np.argmax(adv_pred, 1) == y))
        total += len(labels)
    return c_correct / total, r_correct / total


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    if BASELINE_PATH is None:
        raise FileNotFoundError(
            "Baseline weights not found (models/resnet50_baseline.weights.h5). "
            "Train the baseline first.")

    print("Building adversarial model, warm-starting from baseline ...")
    model = build_model(fine_tune_at=FINE_TUNE_AT)
    model.load_weights(BASELINE_PATH)
    n_bn = freeze_batchnorm(model)
    n_train = len(model.trainable_variables)
    print(f"  froze {n_bn} BatchNorm layers; {n_train} trainable weight tensors remain")

    optimizer = optimizers.Adam(LEARNING_RATE)

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

    steps = len(train_gen)
    best_robust = -1.0
    patience = 0

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        running = dict(loss=0.0, clean=0.0, adv=0.0)
        pbar = tqdm(range(steps), desc="train")
        for step in pbar:
            images, labels = next(train_gen)
            loss, c_acc, a_acc = train_step(
                model, optimizer,
                tf.convert_to_tensor(images, tf.float32),
                tf.convert_to_tensor(labels, tf.float32))
            running["loss"]  += float(loss)
            running["clean"] += float(c_acc)
            running["adv"]   += float(a_acc)
            if (step + 1) % 50 == 0:
                d = step + 1
                pbar.set_postfix(loss=f"{running['loss']/d:.3f}",
                                 clean=f"{running['clean']/d:.3f}",
                                 adv=f"{running['adv']/d:.3f}")

        clean_val, robust_val = val_accuracies(model, val_gen, ROBUST_VAL_BATCHES)
        print(f"  clean_val={clean_val:.4f} | robust_val(PGD)={robust_val:.4f}")

        if robust_val > best_robust:
            best_robust = robust_val
            patience = 0
            model.save_weights(MODEL_SAVE_PATH)
            print(f"  -> saved best (robust_val={robust_val:.4f}) to {MODEL_SAVE_PATH}")
        else:
            patience += 1
            print(f"  no robust improvement ({patience}/{EARLY_STOP_PATIENCE})")
            if patience >= EARLY_STOP_PATIENCE:
                print("  early stopping.")
                break

    print(f"\nDone. Best robust val accuracy = {best_robust:.4f}")
    print(f"Saved to {MODEL_SAVE_PATH}. Now re-run eval_only.py.")


if __name__ == "__main__":
    main()
