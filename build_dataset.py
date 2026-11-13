"""Split the raw SIPaKMeD download into training / validation / testing folders.

SIPaKMeD ships each class as  im_<Class>/[im_<Class>/]CROPPED/*.bmp  (single cells).
We use the CROPPED single-cell images (~4,049 total) for classification.
"""
import argparse
import os
import random
import shutil
import zipfile

from cervnet import config

IMAGE_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def find_class_images(root, class_name):
    """Return all cropped cell images for a class, wherever they sit under its folder."""
    class_dir = os.path.join(root, class_name)
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Class folder not found: {class_dir}")
    paths = []
    for dirpath, _, files in os.walk(class_dir):
        if os.path.basename(dirpath).upper() != "CROPPED":
            continue
        paths += [os.path.join(dirpath, f) for f in files if f.lower().endswith(IMAGE_EXTS)]
    if not paths:  # fall back to any image if there is no CROPPED folder
        for dirpath, _, files in os.walk(class_dir):
            paths += [os.path.join(dirpath, f) for f in files if f.lower().endswith(IMAGE_EXTS)]
    return sorted(paths)


def resolve_dataset_root(path):
    """Accept the downloaded .zip or a folder; return the folder that holds the class folders."""
    if path.lower().endswith(".zip"):
        extract_to = os.path.join("datasets", "sipakmed")
        print(f"[INFO] extracting cropped cell images from {path} -> {extract_to} ...")
        with zipfile.ZipFile(path) as zf:
            # only pull the cropped single-cell images; the full slides are several GB
            members = [m for m in zf.namelist()
                       if "/CROPPED/" in m.upper() and m.lower().endswith(IMAGE_EXTS)]
            zf.extractall(extract_to, members=members or None)
        path = extract_to
    for dirpath, dirnames, _ in os.walk(path):
        if config.CLASSES[0] in dirnames:
            return dirpath
    raise FileNotFoundError(f"Could not find the SIPaKMeD class folders (e.g. {config.CLASSES[0]}) "
                            f"under {path}")


def copy_split(paths, split_dir, label):
    out = os.path.join(split_dir, label)
    os.makedirs(out, exist_ok=True)
    for p in paths:
        shutil.copy2(p, os.path.join(out, os.path.basename(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default=config.ORIG_INPUT_DATASET,
                    help="path to the SIPaKMeD .zip file or extracted folder")
    args = ap.parse_args()
    root = resolve_dataset_root(args.dataset)

    if os.path.exists(config.BASE_PATH):
        shutil.rmtree(config.BASE_PATH)

    rng = random.Random(config.SEED)
    for class_name in config.CLASSES:
        paths = find_class_images(root, class_name)
        rng.shuffle(paths)  # stratified: each class is split on its own
        n_train = int(len(paths) * config.TRAIN_SPLIT)
        train, test = paths[:n_train], paths[n_train:]
        n_val = int(len(train) * config.VAL_SPLIT)
        val, train = train[:n_val], train[n_val:]

        label = class_name.replace("im_", "")
        for split_dir, split in ((config.TRAIN_PATH, train), (config.VAL_PATH, val),
                                 (config.TEST_PATH, test)):
            copy_split(split, split_dir, label)
        print(f"[INFO] {label:<26} train={len(train):4d} val={len(val):4d} test={len(test):4d}")

    print(f"[INFO] split written to {config.BASE_PATH}")


if __name__ == "__main__":
    main()
