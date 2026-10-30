"""Central configuration. NO numpy/torch/cv2 imports here (safe to import before thread env vars)."""
import os

# ======================  >>> EDIT THESE PATHS <<<  ======================
# Root folder that contains DUTS-TR and DUTS-TE (as extracted from the DUTS zip files).
# Can also be set without editing code:  export DUTS_ROOT=/path/to/DUTS
DUTS_ROOT = os.environ.get("DUTS_ROOT", "./data/DUTS")

# Optional: your own captured photos + manual masks (Prompt 2):
#   PHOTOS_DIR/images/*.jpg   and   PHOTOS_DIR/masks/*.png   (same file stems)
PHOTOS_DIR = os.environ.get("PHOTOS_DIR", "./data/captured_photos")

# Optional explicit overrides if auto-discovery of folders fails (leave None normally).
DUTS_TR_IMAGE_DIR = None   # e.g. ".../DUTS-TR/DUTS-TR-Image"
DUTS_TR_MASK_DIR = None    # e.g. ".../DUTS-TR/DUTS-TR-Mask"
DUTS_TE_IMAGE_DIR = None
DUTS_TE_MASK_DIR = None
# ========================================================================

# Assignment-mandated constants
IMG_SIZE = 256
MASK_THRESHOLD = 128                 # mask >= 128 -> foreground
TIME_LIMIT_MS = 50.0
SIZE_REF_BYTES = 4 * 1024 * 1024

# Implementation choices (documented in README)
SEED = 42
VAL_FRACTION = 0.10
NORM_MEAN = (0.485, 0.456, 0.406)    # plain RGB statistics used as constants (NOT pretrained weights)
NORM_STD = (0.229, 0.224, 0.225)

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "best.pt")            # float32, with BN, training state
PARAM_FILE = os.path.join(CHECKPOINT_DIR, "model_fp16.pt")     # the ONLY file counted for Sdisk
SPLIT_FILE = os.path.join(RESULTS_DIR, "split.json")
CACHE_DIR = "cache"                                            # optional .npy cache of 256x256 data

def single_thread_env():
    """Call at the very top of timing scripts, BEFORE importing numpy/cv2/torch."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
