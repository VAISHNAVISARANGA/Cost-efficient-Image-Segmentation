"""Assemble results/report.md from the JSON results that exist; anything missing is marked PENDING with its command.
Also builds a figure per captured photo: original | ground truth | prediction | overlay (Matplotlib, plotting only)."""
import json, os
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config, dataset

R = config.RESULTS_DIR
def load(n):
    p = os.path.join(R, n); return json.load(open(p)) if os.path.exists(p) else None

def row(name, d, cmd):
    if not d: return f"| {name} | PENDING | PENDING | PENDING | `{cmd}` |"
    return f"| {name} | {d['mIoU']:.4f} | {d['BF1']:.4f} | {d['Q']:.4f} | n={d['n_images']} |"

def photo_figures(photos_dir, per):
    pairs = {os.path.splitext(os.path.basename(p[0]))[0]: p for p in dataset.get_photo_pairs(photos_dir)}
    figdir = os.path.join(R, "photo_figs"); os.makedirs(figdir, exist_ok=True); out = []
    for e in per:
        ip, mp = pairs[e["image"]]; img, gt = dataset.load_pair(ip, mp)
        pr = cv2.imread(os.path.join(R, "photo_preds", e["image"] + "_pred.png"), 0) > 127
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); ov = rgb.copy(); ov[pr] = (0.5 * ov[pr] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        ov[(pr ^ gt.astype(bool))] = (0.5 * ov[(pr ^ gt.astype(bool))] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
        for a, im, t in zip(ax, (rgb, gt * 255, pr * 255, ov), ("original", "ground truth", "prediction", "overlay (red=pred, yellow=error)")):
            a.imshow(im, cmap=None if im.ndim == 3 else "gray"); a.set_title(t, fontsize=8); a.axis("off")
        fig.suptitle(f"{e['image']}  mIoU={e['mIoU']:.3f}  BF1={e['BF1']:.3f}", fontsize=9); fp = os.path.join(figdir, e["image"] + ".png")
        fig.savefig(fp, dpi=110, bbox_inches="tight"); plt.close(fig); out.append(fp)
    return out

def main(photos_dir=config.PHOTOS_DIR):
    val, te, ph, bm = load("val_metrics.json"), load("test_metrics.json"), load("photos_metrics.json"), load("benchmark.json")
    L = ["# Report (auto-generated from results/*.json; nothing here is hand-typed)\n", "## Results\n",
         "| Set | mIoU | BF1 | Q | note / command to produce |", "|---|---|---|---|---|",
         row("DUTS-TR validation (best ckpt)", val, "python train.py ..."),
         row("DUTS-TE", te, "python evaluate.py --split test --final"),
         row("Captured photos", ph, "python evaluate.py --split photos")]
    L += ["\n## Cost\n"]
    if bm:
        L += [f"- T_CPU = {bm['T_CPU_ms']:.2f} ms/image over {bm['n_images']} images (time.process_time, 1 thread, jit={bm['jit']})",
              f"- S_disk = {bm['S_disk_bytes']} bytes (`os.path.getsize` of `{bm['counted_in_Sdisk']}`; only this file counted)",
              f"- CES = {bm['CES']}", f"- CPU/software: {json.dumps(bm['env'])}"]
    else:
        L += ["- PENDING: `python benchmark.py --duts_root <DUTS>` (run on your laptop)"]
    if ph and ph.get("per_image"):
        per = sorted(ph["per_image"], key=lambda e: e["mIoU"]); figs = photo_figures(photos_dir, per)
        L += ["\n## Captured photos (worst first)\n"] + [f"![{os.path.basename(f)}]({os.path.relpath(f, R)})" for f in figs]
        L += ["\n## Failure cases\n", "The two lowest-mIoU photos above are the candidates. **Write the likely cause yourself after looking at them** "
              "(e.g. low contrast, cluttered background, multiple objects, thin structures); do not leave this as boilerplate."]
    open(os.path.join(R, "report.md"), "w").write("\n".join(L)); print("wrote", os.path.join(R, "report.md"))

if __name__ == "__main__":
    main()
