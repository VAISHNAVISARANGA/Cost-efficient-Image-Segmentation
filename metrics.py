"""mIoU (single accumulated 2x2 confusion matrix), Boundary-F1 (5x5 tolerance, vectorised), Quality Q."""
import numpy as np


# ---------------- mIoU ----------------
def confusion_2x2(pred, gt):
    """Return 2x2 matrix C[gt, pred] for binary {0,1} arrays."""
    idx = gt.astype(np.int64).ravel() * 2 + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=4).reshape(2, 2)


def miou_from_cm(cm):
    """IoU_c = TP/(TP+FP+FN); mIoU = mean(IoU_bg, IoU_fg). A class with empty union scores 1.0."""
    ious = []
    for c in (0, 1):
        tp = cm[c, c]
        union = cm[c, :].sum() + cm[:, c].sum() - tp
        ious.append(1.0 if union == 0 else tp / union)
    return float((ious[0] + ious[1]) / 2), ious


# ---------------- Boundary F1 ----------------
def boundary(mask):
    """Object pixels with background directly above/below/left/right (outside image = background)."""
    m = mask.astype(bool)
    p = np.pad(m, 1, constant_values=False)
    inner = m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def _any_within(b, r=2):
    """True where any True pixel lies in the (2r+1)x(2r+1) neighbourhood. Separable shifted-OR
    (equivalent to a 5x5 max filter; written with NumPy so no cv2.dilate is needed)."""
    h, w = b.shape
    p = np.pad(b, ((0, 0), (r, r)), constant_values=False)
    out = np.zeros_like(b)
    for dx in range(2 * r + 1):
        out |= p[:, dx:dx + w]
    p = np.pad(out, ((r, r), (0, 0)), constant_values=False)
    res = np.zeros_like(b)
    for dy in range(2 * r + 1):
        res |= p[dy:dy + h, :]
    return res


def boundary_f1(pred, gt, tol=2):
    pb, gb = boundary(pred), boundary(gt)
    npb, ngb = int(pb.sum()), int(gb.sum())
    if npb == 0 and ngb == 0:
        return 1.0
    if npb == 0 or ngb == 0:
        return 0.0
    precision = (pb & _any_within(gb, tol)).sum() / npb   # predicted boundary px with a true one within 5x5
    recall = (gb & _any_within(pb, tol)).sum() / ngb      # true boundary px with a predicted one within 5x5
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def quality(miou, bf1):
    return 0.70 * miou + 0.30 * bf1


class MetricAccumulator:
    """Accumulates one confusion matrix over the set + per-image BF1 list."""
    def __init__(self):
        self.cm = np.zeros((2, 2), dtype=np.int64)
        self.bf1s = []

    def update(self, pred, gt):
        self.cm += confusion_2x2(pred, gt)
        self.bf1s.append(boundary_f1(pred, gt))

    def summary(self):
        miou, (iou_bg, iou_fg) = miou_from_cm(self.cm)
        bf1 = float(np.mean(self.bf1s)) if self.bf1s else 0.0
        return {"mIoU": miou, "IoU_bg": float(iou_bg), "IoU_fg": float(iou_fg),
                "BF1": bf1, "Q": quality(miou, bf1), "n_images": len(self.bf1s)}
