"""Compact U-Net-style FCN built from depthwise-separable convolutions (trained from scratch)."""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ConvBNAct(nn.Module):
    """Conv -> BN -> ReLU. `fuse()` folds BN into the conv (exact at eval time) for faster CPU inference."""
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.bn, nn.Identity):
            return
        c, bn = self.conv, self.bn
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding,
                          groups=c.groups, bias=True)
        fused.weight.copy_(c.weight * scale.view(-1, 1, 1, 1))
        fused.bias.copy_(bn.bias - bn.running_mean * scale)
        self.conv, self.bn = fused, nn.Identity()


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 (each with BN+ReLU)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, 3, 1, groups=cin)
        self.pw = ConvBNAct(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class SegNet(nn.Module):
    """
    256x256x3 -> stem(s2, 128x128) -> enc(128,64,32,16) -> bottleneck(8, + global ctx)
    -> decoder with skip connections (bilinear up) -> 1x1 head at 128x128 -> bilinear x2 -> 256x256 logits.
    `width` scales all channel counts.
    """
    def __init__(self, width=1.0, mean=config.NORM_MEAN, std=config.NORM_STD):
        super().__init__()
        c = lambda n: max(8, int(round(n * width / 4)) * 4)
        c0, c1, c2, c3, c4, cb = c(16), c(24), c(32), c(64), c(96), c(128)
        self.width = width
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer("inv_std", 1.0 / (torch.tensor(std).view(1, 3, 1, 1) * 255.0), persistent=False)

        self.stem = ConvBNAct(3, c0, 3, 2)              # 128
        self.e1 = DSConv(c0, c1)                        # 128
        self.e2 = DSConv(c1, c2)                        # 64
        self.e3 = DSConv(c2, c3)                        # 32
        self.e4 = nn.Sequential(DSConv(c3, c4), DSConv(c4, c4))  # 16
        self.bott = nn.Sequential(DSConv(c4, cb), DSConv(cb, cb))  # 8
        self.gctx = ConvBNAct(cb, cb, 1)                # global-context branch (1x1 after GAP)

        self.d4 = DSConv(cb + c4, c3)                   # 16
        self.d3 = DSConv(c3 + c3, c(48))                # 32
        self.d2 = DSConv(c(48) + c2, c2)                # 64
        self.d1 = DSConv(c2 + c1, c1)                   # 128
        self.head = nn.Conv2d(c1, 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        """x: float tensor (N,3,256,256), RGB, values 0..255 (normalisation happens inside)."""
        x = (x - self.mean) * self.inv_std
        s = self.e1(self.stem(x))                       # 128
        f2 = self.e2(F.max_pool2d(s, 2))                # 64
        f3 = self.e3(F.max_pool2d(f2, 2))               # 32
        f4 = self.e4(F.max_pool2d(f3, 2))               # 16
        b = self.bott(F.max_pool2d(f4, 2))              # 8
        b = b + self.gctx(F.adaptive_avg_pool2d(b, 1))  # broadcast add of global context
        y = self.d4(torch.cat([self._up(b, f4), f4], 1))
        y = self.d3(torch.cat([self._up(y, f3), f3], 1))
        y = self.d2(torch.cat([self._up(y, f2), f2], 1))
        y = self.d1(torch.cat([self._up(y, s), s], 1))
        y = self.head(y)
        return F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)  # 256x256

    def fuse(self):
        for m in self.modules():
            if isinstance(m, ConvBNAct):
                m.fuse()
        return self


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_inference_model(param_file=config.PARAM_FILE, jit=False):
    """Load the saved parameter file (fp16, BN-fused) -> float32 eval model on CPU."""
    ck = torch.load(param_file, map_location="cpu", weights_only=True)
    model = SegNet(width=ck["width"]).fuse()
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    model.eval()
    if jit:
        ex = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE)
        with torch.inference_mode():
            model = torch.jit.freeze(torch.jit.trace(model, ex))
    return model


class Segmenter:
    """Timed pipeline: 256x256 BGR uint8 ndarray -> 256x256 uint8 {0,1} mask."""
    def __init__(self, model):
        self.model = model

    def __call__(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        with torch.inference_mode():
            logit = self.model(x)
        return (logit[0, 0] > 0).numpy().astype(np.uint8)   # logit>0  <=>  prob>0.5
