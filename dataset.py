"""DUTS discovery/pairing, assignment preprocessing, deterministic split, augmentation, cached Dataset."""
import json, os, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------- layout inspection / pairing ----------
def print_tree(root, max_depth=3, max_files=3):
    print(f"[layout] {root}")
    if not os.path.isdir(root):
        print("   (does not exist)"); return
    for dp, dn, fn in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth > max_depth:
            dn[:] = []; continue
        print("   " * depth + f"{os.path.basename(dp) or dp}/  ({len(fn)} files; e.g. {sorted(fn)[:max_files]})")


def discover_dirs(root, split):
    """split in {'TR','TE'}; returns (image_dir, mask_dir). Looks for dirs under root/DUTS-<split> whose
    name contains 'image' / 'mask' (case-insensitive). Honour explicit overrides in config.py."""
    ov = {"TR": (config.DUTS_TR_IMAGE_DIR, config.DUTS_TR_MASK_DIR),
          "TE": (config.DUTS_TE_IMAGE_DIR, config.DUTS_TE_MASK_DIR)}[split]
    if ov[0] and ov[1]:
        return ov
    base = None
    for dp, dn, _ in os.walk(root):
        for d in dn:
            if d.upper() == f"DUTS-{split}":
                base = os.path.join(dp, d); break
        if base: break
    if base is None:
        print_tree(root)
        raise FileNotFoundError(f"Could not find a 'DUTS-{split}' folder under {root}. "
                                f"Set DUTS_ROOT or the explicit *_DIR overrides in config.py.")
    img = msk = None
    for dp, dn, fn in os.walk(base):
        name = os.path.basename(dp).lower()
        if any(os.path.splitext(f)[1].lower() in IMG_EXT for f in fn):
            if "mask" in name or "gt" in name: msk = msk or dp
            elif "image" in name or "img" in name: img = img or dp
    if not (img and msk):
        print_tree(base)
        raise FileNotFoundError(f"Could not identify image/mask folders under {base}; set overrides in config.py.")
    return img, msk


def pair_files(img_dir, mask_dir, max_unmatched_frac=0.01):
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMG_EXT}
    msks = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) if os.path.splitext(f)[1].lower() in IMG_EXT}
    common = sorted(set(imgs) & set(msks))
    miss_m, miss_i = sorted(set(imgs) - set(msks)), sorted(set(msks) - set(imgs))
    print(f"[pairing] images={len(imgs)} masks={len(msks)} matched={len(common)} "
          f"images_without_mask={len(miss_m)} masks_without_image={len(miss_i)}")
    if miss_m: print("   e.g. image without mask:", miss_m[:5])
    if miss_i: print("   e.g. mask without image:", miss_i[:5])
    tot = max(len(imgs), len(msks), 1)
    if (len(miss_m) + len(miss_i)) / tot > max_unmatched_frac:
        raise RuntimeError("Too many unmatched files - check directory layout / naming.")
    return [(os.path.join(img_dir, imgs[k]), os.path.join(mask_dir, msks[k])) for k in common]


def get_pairs(root, split):
    return pair_files(*discover_dirs(root, split))


def get_photo_pairs(photos_dir):
    return pair_files(os.path.join(photos_dir, "images"), os.path.join(photos_dir, "masks"), 0.0)


def make_split(pairs, val_frac=config.VAL_FRACTION, seed=config.SEED, path=config.SPLIT_FILE):
    """Fixed-seed split of DUTS-TR; the stems are saved to disk so the split is auditable."""
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(idx) * val_frac))
    val_idx, tr_idx = sorted(idx[:n_val]), sorted(idx[n_val:])
    tr = [pairs[i] for i in tr_idx]; va = [pairs[i] for i in val_idx]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"seed": seed, "val_frac": val_frac, "n_train": len(tr), "n_val": len(va),
               "val_stems": [os.path.splitext(os.path.basename(p[0]))[0] for p in va]}, open(path, "w"), indent=1)
    print(f"[split] seed={seed} train={len(tr)} val={len(va)}")
    return tr, va


# ---------- assignment preprocessing ----------
def load_pair(img_path, mask_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)                       # BGR uint8
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        raise IOError(f"unreadable pair: {img_path} / {mask_path}")
    S = config.IMG_SIZE
    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (S, S), interpolation=cv2.INTER_NEAREST)
    msk = (msk >= config.MASK_THRESHOLD).astype(np.uint8)              # >=128 -> 1
    return img, msk                                                    # BGR uint8 256x256, {0,1} uint8


def build_arrays(pairs, desc=""):
    n = len(pairs)
    X = np.empty((n, config.IMG_SIZE, config.IMG_SIZE, 3), np.uint8)
    Y = np.empty((n, config.IMG_SIZE, config.IMG_SIZE), np.uint8)
    for i, (ip, mp) in enumerate(pairs):
        X[i], Y[i] = load_pair(ip, mp)
        if (i + 1) % 1000 == 0: print(f"[load {desc}] {i+1}/{n}", flush=True)
    return X, Y


def cached_arrays(pairs, name, cache_dir=config.CACHE_DIR):
    """256x256 arrays cached as .npy (stored BGR exactly as produced by load_pair). Delete cache/ if data changes."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fx, fy = os.path.join(cache_dir, f"{name}_X.npy"), os.path.join(cache_dir, f"{name}_Y.npy")
        if os.path.exists(fx) and os.path.exists(fy):
            X, Y = np.load(fx, mmap_mode="r"), np.load(fy, mmap_mode="r")
            if len(X) == len(pairs): return X, Y
        X, Y = build_arrays(pairs, name); np.save(fx, X); np.save(fy, Y); return X, Y
    return build_arrays(pairs, name)


# ---------- augmentation (identical geometry for image and mask) ----------
def augment(img, msk, rng):
    S = config.IMG_SIZE
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.7:
        ang = rng.uniform(-15, 15); sc = rng.uniform(0.8, 1.25)
        tx, ty = rng.uniform(-0.1, 0.1) * S, rng.uniform(-0.1, 0.1) * S
        M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, sc); M[:, 2] += (tx, ty)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        msk = cv2.warpAffine(np.ascontiguousarray(msk), M, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.8:   # photometric: image only
        img = np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


class SegDataset(Dataset):
    """Yields (RGB float tensor 3x256x256 in 0..255, mask float tensor 1x256x256). Normalisation is inside the model."""
    def __init__(self, X, Y, train=False, seed=config.SEED):
        self.X, self.Y, self.train, self.seed = X, Y, train, seed

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        img, msk = np.asarray(self.X[i]), np.asarray(self.Y[i])
        if self.train:
            rng = random.Random(self.seed * 1000003 + i * 9176 + torch.initial_seed() % (2 ** 31))  # varies per epoch, reproducible given seed
            img, msk = augment(img, msk, rng)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (torch.from_numpy(rgb).permute(2, 0, 1).float(), torch.from_numpy(msk.astype(np.float32)).unsqueeze(0))
