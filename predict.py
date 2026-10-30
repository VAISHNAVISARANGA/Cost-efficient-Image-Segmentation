import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"                                 # must precede numpy/cv2/torch imports
import argparse, sys, time
import numpy as np
import cv2, torch
cv2.setNumThreads(1); torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
"""python predict.py --image photo.jpg --out mask.png   (optional: --params path)"""
import config
from model import load_inference_model, Segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=config.PARAM_FILE)
    a = ap.parse_args()
    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None: sys.exit(f"cannot read {a.image}")
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)   # untimed
    seg = Segmenter(load_inference_model(a.params))                                          # untimed
    mask = seg(img)                                                                          # warm-up (untimed)
    t0 = time.process_time()
    for _ in range(20): mask = seg(img)
    t_ms = (time.process_time() - t0) / 20 * 1000.0
    cv2.imwrite(a.out, (mask * 255).astype(np.uint8))
    print(f"info: saved mask {a.out} (foreground=255), shape={mask.shape}", file=sys.stderr)
    print(f"T CPU ms: {t_ms:.2f}")
    print(f"S disk bytes: {os.path.getsize(a.params)}")


if __name__ == "__main__":
    main()
