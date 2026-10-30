"""Quality evaluation (mIoU / BF1 / Q) of the saved parameter file.
  --split val     DUTS-TR validation split (development use)
  --split test    DUTS-TE: FINAL, once. Needs --final; refuses to re-run without --force.
  --split photos  your captured photos (PHOTOS_DIR/images + masks); also saves per-image predictions.
Uses the exact timed pipeline (model.Segmenter). Multi-threaded is fine here (timing is in benchmark.py)."""
import argparse, os
import numpy as np, cv2
import config, dataset
from model import load_inference_model, Segmenter
from metrics import MetricAccumulator, boundary_f1, confusion_2x2, miou_from_cm
from utils import save_json, env_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "photos"], required=True)
    ap.add_argument("--duts_root", default=config.DUTS_ROOT); ap.add_argument("--photos_dir", default=config.PHOTOS_DIR)
    ap.add_argument("--params", default=config.PARAM_FILE); ap.add_argument("--cache_dir", default=config.CACHE_DIR)
    ap.add_argument("--final", action="store_true", help="confirm that settings are frozen (required for --split test)")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out_json = os.path.join(config.RESULTS_DIR, f"{a.split}_metrics.json" if a.split != "val" else "val_final_metrics.json")

    if a.split == "test":
        if not a.final: raise SystemExit("DUTS-TE is reserved for the final run. Re-run with --final once settings are frozen.")
        if os.path.exists(out_json) and not a.force: raise SystemExit(f"{out_json} exists: DUTS-TE already evaluated (use --force only for a re-run you will disclose).")
        pairs = dataset.get_pairs(a.duts_root, "TE")
    elif a.split == "val":
        pairs = dataset.make_split(dataset.get_pairs(a.duts_root, "TR"))[1]
    else:
        pairs = dataset.get_photo_pairs(a.photos_dir)
    if a.limit: pairs = pairs[:a.limit]
    X, Y = dataset.cached_arrays(pairs, f"eval_{a.split}_{len(pairs)}", a.cache_dir or None) if a.split != "photos" else dataset.build_arrays(pairs, "photos")

    seg = Segmenter(load_inference_model(a.params)); acc = MetricAccumulator(); per = []
    pred_dir = os.path.join(config.RESULTS_DIR, "photo_preds")
    if a.split == "photos": os.makedirs(pred_dir, exist_ok=True)
    for i in range(len(X)):
        img = np.ascontiguousarray(X[i]); gt = np.asarray(Y[i]); pred = seg(img); acc.update(pred, gt)
        if a.split == "photos":
            stem = os.path.splitext(os.path.basename(pairs[i][0]))[0]
            cm = confusion_2x2(pred, gt); per.append({"image": stem, "mIoU": miou_from_cm(cm)[0], "BF1": acc.bf1s[-1]})
            cv2.imwrite(os.path.join(pred_dir, stem + "_pred.png"), pred * 255)
    s = acc.summary(); s.update({"split": a.split, "params_file": a.params, "env": env_report()})
    if per: s["per_image"] = per
    save_json(s, out_json)
    print(f"[{a.split}] n={s['n_images']}  mIoU={s['mIoU']:.4f}  BF1={s['BF1']:.4f}  Q={s['Q']:.4f}   -> {out_json}")


if __name__ == "__main__":
    main()
