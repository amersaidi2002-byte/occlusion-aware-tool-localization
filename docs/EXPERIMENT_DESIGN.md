# Experiment design

## Question and approach

Can a small network trained from random initialization localize both ends of a
stylized instrument when parts of it are hidden? The output is a **visible**
segmentation mask plus two **amodal** keypoints: tip and base, in that order.
This is a 2-D perception experiment, not 6-D pose estimation or robot control.

A 66,699-parameter encoder-decoder has two downsampling stages, skip connections,
and a three-channel output. Mask training combines binary cross entropy and soft
Dice loss. Keypoint training minimizes cross entropy against normalized Gaussian
heatmaps. An all-zero heatmap solution is therefore not a cheap way to minimize
a sparse pixel-wise mean-squared error. Training uses Adam, gradient clipping
and deterministic CPU operations; no pretrained parameters are loaded.

## Dataset and evaluation contract

- Train: 1,200 generated images, scene IDs 0-7.
- Validation: 240 generated images, scene IDs 8-9.
- Test: 480 generated images, scene IDs 10-11.
- Geometry-shift test: 480 further images, scene IDs 12-13, with longer/wider tools.
- Renderer seed: 17. Each split has a separate random-number stream.
- Each scene has a fixed generated background style. These are *synthetic scene
  IDs*, not collected camera recording sessions or real instruments.
- Labels come from the renderer. Occlusion bins use measured hidden mask area:
  none <1%, partial 1-30%, heavy >=30%.
- Pixels are RGB uint8. Network input is contiguous float32 NCHW /255 at 64x64.
- Keypoints are decoded by discrete argmax and reported in (x,y) image pixels.
- IoU is calculated per image, then averaged. PCK@5 is the fraction of the two
  labeled keypoints within five pixels, including occluded keypoints. Visible-only
  error is reported separately.

Checkpoints are selected by validation loss, weighted by batch size. Test data
are never used for fitting. The threshold/PCA baseline has no fitted weights:
it thresholds bright, near-neutral pixels, estimates the principal axis and
uses the synthetic brightness gradient to assign the tip/base direction.
This is an intentionally simple, transparent reference, not a state-of-the-art
detector. Its threshold and assumptions are visible in `evaluation.py`.

The confidence gate retains predictions above the 20th percentile of validation
spatial-softmax peak scores. Test coverage and retained quality are reported.
This is an **uncalibrated heuristic**, not a reliability or safety certificate.

## Controlled comparison

The ablation uses the same architecture, seed, image counts, optimizer and
16-epoch budget but removes training occluders, brightness jitter and blur.
Validation/test still contain these perturbations. This jointly tests the
augmentation package; it does not isolate individual augmentation types.
Both trials are reported, even when a baseline or ablation performs better.

## Reproducibility and limits

The published reports contain configuration, validation history, selected epoch,
versions, scene splits and artifact SHA-256 digests. Commands use two CPU threads.
Thread counts/library versions can change numerical trajectories; exact
cross-machine bitwise reproduction is not promised. Single-seed estimates do not
measure variance across training seeds.

The generator is deliberately simple: lighting, occluders and geometry do not
capture tissue, fluids, reflections or surgical optics. The geometry-shift split
can be easier than the standard split because larger shapes cover more pixels.
Results establish executable software and a controlled learning experiment, not
medical performance or deployment readiness. No real robot, patient data,
clinical validation or clinical collaboration was involved.

This personal project was created in November 2026 with AI coding assistance.
It is independent of previous internships. Review and explain the implementation
before presenting it in an interview; do not attribute it to an employer.
