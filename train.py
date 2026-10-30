"""Train on the DUTS-TR train split, validate on the DUTS-TR val split. DUTS-TE is never touched here.
Example: python train.py --duts_root /data/DUTS --epochs 60 --batch_size 32 --width 1.0"""
import argparse, csv, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import config, dataset, export
from model import SegNet, count_params
from metrics import MetricAccumulator
from utils import seed_everything, save_json


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits); inter = (p * y).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


def run_val(model, loader, device, crit):
    model.eval(); acc = MetricAccumulator(); tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device); lg = model(x)
            tot += crit(lg, y).item() * len(x); n += len(x)
            pred = (lg[:, 0] > 0).cpu().numpy().astype(np.uint8); gt = y[:, 0].cpu().numpy().astype(np.uint8)
            for p, g in zip(pred, gt): acc.update(p, g)
    return tot / n, acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duts_root", default=config.DUTS_ROOT)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--width", type=float, default=1.0, help="channel multiplier")
    ap.add_argument("--dice_weight", type=float, default=0.0, help="0 = pure BCE; >0 adds soft-Dice")
    ap.add_argument("--patience", type=int, default=12, help="early stop on val Q")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--val_frac", type=float, default=config.VAL_FRACTION)
    ap.add_argument("--cache_dir", default=config.CACHE_DIR, help="'' disables the .npy cache")
    ap.add_argument("--no_aug", action="store_true"); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max_train", type=int, default=0, help="debug: limit #train images")
    a = ap.parse_args()

    seed_everything(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device:", dev)
    pairs = dataset.get_pairs(a.duts_root, "TR")
    tr_pairs, va_pairs = dataset.make_split(pairs, a.val_frac, a.seed)
    if a.max_train: tr_pairs = tr_pairs[:a.max_train]
    Xtr, Ytr = dataset.cached_arrays(tr_pairs, f"tr_{a.seed}_{len(tr_pairs)}", a.cache_dir or None)
    Xva, Yva = dataset.cached_arrays(va_pairs, f"va_{a.seed}_{len(va_pairs)}", a.cache_dir or None)
    g = torch.Generator(); g.manual_seed(a.seed)
    tl = DataLoader(dataset.SegDataset(Xtr, Ytr, train=not a.no_aug, seed=a.seed), a.batch_size, shuffle=True,
                    num_workers=a.workers, drop_last=True, generator=g, persistent_workers=a.workers > 0)
    vl = DataLoader(dataset.SegDataset(Xva, Yva), a.batch_size, shuffle=False, num_workers=0)

    model = SegNet(width=a.width).to(dev)
    print(f"params: {count_params(model):,}")
    bce = nn.BCEWithLogitsLoss()
    crit = lambda lg, y: bce(lg, y) + (a.dice_weight * dice_loss(lg, y) if a.dice_weight > 0 else 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev.type == "cuda")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True); os.makedirs(config.RESULTS_DIR, exist_ok=True)
    logf = os.path.join(config.RESULTS_DIR, "train_log.csv")
    with open(logf, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_BF1", "val_Q", "lr", "epoch_sec"])
    best_q, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=a.amp and dev.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); n += len(x)
        vloss, vs = run_val(model, vl, dev, crit); dt = time.time() - t0
        print(f"ep {ep:03d} | train {tot/n:.4f} | val {vloss:.4f} | mIoU {vs['mIoU']:.4f} BF1 {vs['BF1']:.4f} Q {vs['Q']:.4f} | {dt:.0f}s", flush=True)
        with open(logf, "a", newline="") as f:
            csv.writer(f).writerow([ep, tot / n, vloss, vs["mIoU"], vs["BF1"], vs["Q"], opt.param_groups[0]["lr"], dt])
        if vs["Q"] > best_q:                                   # checkpoint on validation Q only
            best_q, bad = vs["Q"], 0
            torch.save({"width": a.width, "state_dict": model.state_dict(), "epoch": ep, "val": vs, "args": vars(a)}, config.BEST_CKPT)
            save_json({"best_epoch": ep, **vs}, os.path.join(config.RESULTS_DIR, "val_metrics.json"))
        else:
            bad += 1
            if bad >= a.patience: print("early stopping (validation Q stopped improving)"); break
    export.export(config.BEST_CKPT, config.PARAM_FILE)


if __name__ == "__main__":
    main()
