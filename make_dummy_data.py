"""Create a tiny synthetic DUTS-shaped dataset for smoke-testing the pipeline (NOT real data)."""
import argparse, os
import numpy as np, cv2

def make(root, split, n, rng):
    idir = os.path.join(root, f"DUTS-{split}", f"DUTS-{split}-Image"); mdir = os.path.join(root, f"DUTS-{split}", f"DUTS-{split}-Mask")
    os.makedirs(idir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    for i in range(n):
        h, w = int(rng.integers(200, 420)), int(rng.integers(200, 420))
        bg = (rng.integers(0, 255, 3)).astype(np.uint8)
        img = np.tile(bg, (h, w, 1)).astype(np.uint8)
        img = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
        msk = np.zeros((h, w), np.uint8)
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        c = (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4)))
        if rng.random() < 0.5:
            ax = (int(rng.integers(w // 8, w // 3)), int(rng.integers(h // 8, h // 3)))
            cv2.ellipse(img, c, ax, 0, 0, 360, col, -1); cv2.ellipse(msk, c, ax, 0, 0, 360, 255, -1)
        else:
            hw, hh = int(rng.integers(w // 8, w // 3)), int(rng.integers(h // 8, h // 3))
            p1, p2 = (c[0] - hw, c[1] - hh), (c[0] + hw, c[1] + hh)
            cv2.rectangle(img, p1, p2, col, -1); cv2.rectangle(msk, p1, p2, 255, -1)
        cv2.imwrite(os.path.join(idir, f"img_{i:05d}.jpg"), img)
        cv2.imwrite(os.path.join(mdir, f"img_{i:05d}.png"), msk)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/DUMMY_DUTS"); ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--n_test", type=int, default=40); ap.add_argument("--n_photos", type=int, default=6)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    make(a.root, "TR", a.n_train, rng); make(a.root, "TE", a.n_test, rng)
    # dummy "captured photos" in the images/ + masks/ layout used by evaluate.py
    ph = os.path.join(a.root, "..", "DUMMY_PHOTOS"); os.makedirs(os.path.join(ph, "images"), exist_ok=True); os.makedirs(os.path.join(ph, "masks"), exist_ok=True)
    make(os.path.join(a.root, "_tmp"), "TE", a.n_photos, rng)
    import shutil
    t = os.path.join(a.root, "_tmp", "DUTS-TE")
    for f in os.listdir(os.path.join(t, "DUTS-TE-Image")): shutil.move(os.path.join(t, "DUTS-TE-Image", f), os.path.join(ph, "images", f))
    for f in os.listdir(os.path.join(t, "DUTS-TE-Mask")): shutil.move(os.path.join(t, "DUTS-TE-Mask", f), os.path.join(ph, "masks", f))
    shutil.rmtree(os.path.join(a.root, "_tmp"))
    print("dummy data written to", a.root, "and", os.path.abspath(ph))
