"""Fold BN into convs and save the fp16 parameter file that inference loads (this file defines Sdisk)."""
import argparse, os
import torch
import config
from model import SegNet, load_inference_model


def export(ckpt_path=config.BEST_CKPT, out=config.PARAM_FILE):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m = SegNet(width=ck["width"]); m.load_state_dict(ck["state_dict"]); m.eval()
    ref_in = torch.rand(2, 3, 256, 256) * 255
    with torch.no_grad(): ref = m(ref_in)
    m.fuse()
    sd = {k: v.half() for k, v in m.state_dict().items()}       # only learned tensors, float16
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"width": ck["width"], "state_dict": sd}, out)
    chk = load_inference_model(out)
    with torch.no_grad(): got = chk(ref_in)
    print(f"[export] wrote {out}: {os.path.getsize(out)} bytes; max |logit diff| fp32-train vs fp16-fused = "
          f"{(ref - got).abs().max().item():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=config.BEST_CKPT); ap.add_argument("--out", default=config.PARAM_FILE)
    a = ap.parse_args(); export(a.ckpt, a.out)
