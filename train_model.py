"""Train the ResNet on the SIPaKMeD split and evaluate it on the held-out test set."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from cervnet import config
from cervnet.resnet import build_resnet


def load_split(path, shuffle):
    """Load one split, cache the decoded images in memory and batch them.

    SIPaKMeD's cropped cells are small, so caching after the first epoch removes
    disk reads and image decoding from every later epoch.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        path, image_size=config.IMAGE_SIZE, batch_size=None,
        label_mode="categorical", shuffle=False)
    class_names = ds.class_names
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(4096, seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(config.BATCH_SIZE)
    ds.class_names = class_names
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=int, default=config.NUM_EPOCHS)
    args = ap.parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_ds = load_split(config.TRAIN_PATH, shuffle=True)
    val_ds = load_split(config.VAL_PATH, shuffle=False)
    test_ds = load_split(config.TEST_PATH, shuffle=False)
    class_names = train_ds.class_names

    # class weights so smaller classes are not drowned out
    y_train = np.concatenate([np.argmax(y, axis=1) for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y_train)
    class_weight = dict(enumerate(weights))

    train_ds, val_ds, test_ds = (d.prefetch(tf.data.AUTOTUNE) for d in (train_ds, val_ds, test_ds))

    # cells have no canonical orientation, so flips/rotations are safe augmentations
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

    model = build_resnet((*config.IMAGE_SIZE, 3), len(class_names), augmentation=augmentation)

    # polynomial (linear) learning-rate decay, as in the original tutorial
    steps = args.epochs * len(train_ds)
    lr = tf.keras.optimizers.schedules.PolynomialDecay(config.INIT_LR, steps, 0.0, power=1.0)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(config.MODEL_PATH, monitor="val_accuracy",
                                                    save_best_only=True)]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=class_weight, callbacks=callbacks)

    # evaluate the best checkpoint on the test set
    model = tf.keras.models.load_model(config.MODEL_PATH)
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(config.REPORT_PATH, "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45,
                                                                cmap="Blues", colorbar=False)
    ax.set_title("SIPaKMeD test set — confusion matrix")
    fig.tight_layout()
    fig.savefig(config.CM_PATH, dpi=150)

    h = history.history
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
        ax.plot(h[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Accuracy")
    ax.set_title("Training loss and accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOT_PATH, dpi=150)
    print(f"[INFO] model saved to {config.MODEL_PATH}; plots and report in {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
