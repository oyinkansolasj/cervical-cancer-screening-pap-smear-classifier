"""Classify one or more Pap smear cell images with a trained model."""
import argparse
import os

import numpy as np
import tensorflow as tf

from cervnet import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="image file(s) to classify")
    ap.add_argument("-m", "--model", default=config.MODEL_PATH)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    labels = sorted(c.replace("im_", "") for c in config.CLASSES)
    for path in args.images:
        img = tf.keras.utils.load_img(path, target_size=config.IMAGE_SIZE)
        x = np.expand_dims(tf.keras.utils.img_to_array(img), 0)
        probs = model.predict(x, verbose=0)[0]
        i = int(np.argmax(probs))
        print(f"{os.path.basename(path)}: {labels[i]} ({probs[i]:.2%})")


if __name__ == "__main__":
    main()
