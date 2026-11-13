"""Central configuration for the SIPaKMeD cervical cell classification project."""
import os

# Where the raw SIPaKMeD download lives (the folder containing im_Dyskeratotic, ...)
ORIG_INPUT_DATASET = os.environ.get("SIPAKMED_DIR", os.path.join("datasets", "sipakmed"))

# Where build_dataset.py writes the train/val/test split
BASE_PATH = os.path.join("datasets", "split")
TRAIN_PATH = os.path.join(BASE_PATH, "training")
VAL_PATH = os.path.join(BASE_PATH, "validation")
TEST_PATH = os.path.join(BASE_PATH, "testing")

# The five SIPaKMeD classes (folder names in the original download)
CLASSES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]

# Split ratios: 80% train / 20% test, then 10% of train held out for validation
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
SEED = 42

# Training hyper-parameters
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_EPOCHS = 50
INIT_LR = 1e-1

# Outputs
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "cervnet.keras")
PLOT_PATH = os.path.join(OUTPUT_DIR, "training_plot.png")
CM_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
