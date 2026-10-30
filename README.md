# Cost-Efficient Binary Image Segmentation (Track B: small CNN from scratch, DUTS)

Allowed libs only: stdlib, PyTorch, NumPy, OpenCV, Matplotlib. No pretrained weights, no torchvision, no sklearn/scipy,
no `cv2.dilate/erode/kmeans/grabCut`. (The 5x5 boundary tolerance is done with NumPy shifted-OR, not `cv2.dilate`.)

## 0. Verification status (read this first)
| Item | Status |
|---|---|
| `metrics.py` unit tests (`tests/test_metrics.py`, 9 tests incl. brute-force cross-check) | **Run, passing** (NumPy only) |
| All `.py` files compile; dummy-data generator runs | **Run, OK** |
| Model forward pass, training, export, evaluate, benchmark, predict | **NOT yet run** (the authoring sandbox had no PyTorch / ran out of disk). Run the smoke test in section 2 first. |
| Parameter count, T_CPU, S_disk, mIoU/BF1/Q, CES | **PENDING** - nothing is reported until you measure it |

## 1. Where to put your data  (the only things you must edit)
Edit `config.py` **or** use env vars:
```
export DUTS_ROOT=/path/to/DUTS          # folder containing DUTS-TR and DUTS-TE
export PHOTOS_DIR=/path/to/my_photos    # my_photos/images/*.jpg  +  my_photos/masks/*.png (same stems)
```
Folders are auto-discovered (`DUTS-TR/DUTS-TR-Image`, `DUTS-TR-Mask`, same for TE). If your layout differs, the script
prints the directory tree and stops; then set `DUTS_TR_IMAGE_DIR`, `DUTS_TR_MASK_DIR`, `DUTS_TE_*` in `config.py`.
Pairing is by file stem; unmatched files are printed, and it aborts if >1% are unmatched.

## 2. Smoke test on dummy data (minutes, any machine)
```
pip install torch numpy opencv-python matplotlib
python make_dummy_data.py --root ./data/DUMMY_DUTS --n_train 120 --n_test 40
python -m unittest discover -s tests -v
python train.py --duts_root ./data/DUMMY_DUTS --epochs 3 --batch_size 8 --workers 0 --cache_dir ""
python predict.py --image data/DUMMY_DUTS/DUTS-TE/DUTS-TE-Image/img_00000.jpg --out mask.png
```
`predict.py` prints exactly `T CPU ms: ..` and `S disk bytes: ..` (extra info goes to stderr).

## 3. Real pipeline
```
python train.py --duts_root $DUTS_ROOT --epochs 60 --batch_size 32 --width 1.0          # GPU used if present
python benchmark.py --duts_root $DUTS_ROOT --limit 500      # EARLY speed check (timing only; does not read DUTS-TE masks)
#   if T_CPU is near/over 50 ms: lower --width (e.g. 0.75/0.5) and retrain; tune on VALIDATION only
python evaluate.py --split val
# ---- freeze everything. From here on: DUTS-TE, once ----
python evaluate.py --split test --final           # refuses to re-run unless --force
python benchmark.py --duts_root $DUTS_ROOT        # all 5019 images, one by one, single thread, time.process_time()
python evaluate.py --split photos                 # your captured photos
python report.py                                  # results/report.md + per-photo figures
```
Note: the `--limit 500` benchmark loads the first 500 DUTS-TE *images* only to measure speed; no TE metric is computed
or used for decisions. If you want to be stricter, benchmark on validation images instead.

## 4. Design
- **Model** (`model.py`): U-Net-style FCN; stride-2 stem -> 4 encoder stages of depthwise-separable convs (BN+ReLU) ->
  8x8 bottleneck + global-average-pooled context -> decoder with skip connections (bilinear upsample + concat) ->
  1x1 head -> bilinear x2 -> 256x256 logits. `--width` scales channels. Foreground = logit > 0 (prob > 0.5; threshold not tuned).
- **Preprocessing** (exactly as assignment): `cv2.imread` -> `cv2.resize(256,256, INTER_AREA)`; mask `INTER_NEAREST`, `>=128 -> 1`.
  BGR->RGB inside the timed pipeline. Normalisation (my choice, not mandated): fixed channel mean/std constants applied
  inside the model, identically for train/val/inference.
- **Augmentation** (train only, image+mask geometrically identical): h-flip, rotation +-15 deg / scale 0.8-1.25 / shift 10%
  via `cv2.warpAffine` (LINEAR for image, NEAREST for mask); brightness/contrast jitter on the image only. Validation is deterministic.
- **Loss/optim**: `BCEWithLogitsLoss` (+ optional soft Dice via `--dice_weight`), AdamW, OneCycle LR. Checkpoint and early
  stopping use validation Q only. Split: seeded 90/10 of DUTS-TR, saved to `results/split.json`.
- **CPU speed**: BN folded into convs at export; `torch.inference_mode`; single thread; optional `--jit` (TorchScript trace+freeze)
  in `benchmark.py` - keep only if it measures faster. `--width` is the main speed/quality dial.
- **Sdisk**: `checkpoints/model_fp16.pt` = fp16 BN-fused state_dict + width value; this single file is what `os.path.getsize` measures
  and what inference loads. `best.pt` (fp32 training checkpoint) is NOT counted. `export.py` prints the max logit difference
  caused by fp16/fusion - check it is small.
- **Metrics** (`metrics.py`): one accumulated 2x2 confusion matrix per set -> mean of bg/fg IoU; per-image BF1 (4-neighbour
  boundary, outside = background, 5x5 tolerance, both-empty = 1, one-empty = 0) averaged; Q = 0.7 mIoU + 0.3 BF1.
  (A class with an empty union counts as IoU 1.0 - only relevant for degenerate toy inputs.)
- **Timing**: thread env vars are set before importing numpy/cv2/torch in `benchmark.py` and `predict.py`; then
  `cv2.setNumThreads(1)`, `torch.set_num_threads(1)`. Timed region = BGR->RGB, tensor conversion, forward, threshold -> 0/1 mask.
  Images are pre-loaded/resized before the timer; one warm-up call is made first.

## 5. Related work used as design inspiration (numbers not verified here)
U-Net (Ronneberger et al. 2015) - skip-connected encoder-decoder; MobileNet (Howard et al. 2017) - depthwise-separable convs;
Fast-SCNN (Poudel et al. 2019) - fast low-resolution context path; U^2-Net (Qin et al. 2020) - DUTS-trained salient-object
network (far larger than this budget allows). DUTS is a *salient object* dataset, so the "object" is whatever is most
salient; expect weaker results on cluttered or multi-object scenes, especially for a tiny model.

## 6. Limitations / honesty notes
No result in this repo is pre-filled. Expected tuning loop: if T_CPU > 50 ms the CES is 0 - reduce `--width`, drop a
`DSConv` in `e4`/`bott`, or lower the working resolution, then retrain. Timing is hardware-specific: report your laptop's CPU
(auto-recorded in `results/benchmark.json`). Failure-case explanations in the report must be written by you after
inspecting the figures.
