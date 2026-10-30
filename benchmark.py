import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"                                 # must precede numpy/cv2/torch imports
import argparse, time
import numpy as np
import cv2, torch
cv2.setNumThreads(1); torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
"""Official CPU benchmark: all DUTS-TE images, one by one, single thread, time.process_time().
Run on YOUR laptop. Image loading/resizing happens BEFORE the timed region."""
import config, dataset
from model import load_inference_model, Segmenter
from metrics import MetricAccumulator
from utils import compute_ces, env_report, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duts_root", default=config.DUTS_ROOT); ap.add_argument("--params", default=config.PARAM_FILE)
    ap.add_argument("--limit", type=int, default=0, help="debug only; official run uses all 5019")
    ap.add_argument("--jit", action="store_true", help="try TorchScript (trace+freeze); keep only if it measures faster")
    ap.add_argument("--q", type=float, default=None, help="DUTS-TE Q; default: read results/test_metrics.json if present")
    a = ap.parse_args()

    pairs = dataset.get_pairs(a.duts_root, "TE")
    if a.limit: pairs = pairs[:a.limit]
    imgs = [dataset.load_pair(ip, mp)[0] for ip, mp in pairs]           # untimed: 256x256 BGR images in memory
    seg = Segmenter(load_inference_model(a.params, jit=a.jit))
    seg(imgs[0])                                                         # warm-up (untimed)
    t0 = time.process_time()
    for im in imgs: seg(im)
    total = time.process_time() - t0
    t_ms = total / len(imgs) * 1000.0
    s_bytes = os.path.getsize(a.params)

    q = a.q
    qf = os.path.join(config.RESULTS_DIR, "test_metrics.json")
    if q is None and os.path.exists(qf):
        import json; q = json.load(open(qf))["Q"]
    res = {"n_images": len(imgs), "total_process_time_s": total, "T_CPU_ms": t_ms, "S_disk_bytes": s_bytes,
           "within_50ms": t_ms <= config.TIME_LIMIT_MS, "jit": a.jit, "Q_used": q,
           "CES": compute_ces(q, t_ms, s_bytes) if q is not None else "pending (run evaluate.py --split test --final first)",
           "counted_in_Sdisk": a.params, "env": env_report()}
    save_json(res, os.path.join(config.RESULTS_DIR, "benchmark.json"))
    print(f"images={len(imgs)}  total_process_time={total:.3f}s  T_CPU={t_ms:.2f} ms  S_disk={s_bytes} bytes  CES={res['CES']}")


if __name__ == "__main__":
    main()
