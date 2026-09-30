# MK-UNet screening experiments for Prof. Rahman

This package records AI-assisted screening work for Yuvraj Verma. It contains actual experiments, not invented metrics. The original MK-UNet architecture is preserved, including its BSD-3-Clause license. Candidate review and an analysis rewritten in the candidate's own words are required before submission.

## Recorded results (single seed, native masks)

| Task | Mean Dice | Mean IoU | Selected epoch |
| --- | ---: | ---: | ---: |
| ClinicDB, 62 test images | 89.59% | 83.62% | 58 |
| Massachusetts Buildings, 10 test tiles | 71.95% | 56.25% | 136 |

The source normalized Dice is also 89.59%, below the paper's five-run mean of 93.48%. This is a declared reproduction attempt with protocol differences, not exact equivalence.

The predeclared adaptation pilot accepted 110/310 polyp candidates and retained one update causing >1 Dice-point loss, versus eight ungated harmful candidates. For buildings, it accepted only 1/50 candidates. Noise reduced frozen Dice to 24.11% and 5.34% respectively. These results expose limitations; they do not establish successful adaptation or clinical reliability. See [the independent evidence audit](evidence_audit.json), [per-case records](runs/clinic_seed42/pilot/per_case.json), and [learning/failure figures](figures/).

GPU inference is not bitwise deterministic. Clean pilot versus main per-case Dice differences reach 0.059 percentage points for ClinicDB and 0.006 for buildings; ground-truth counts agree exactly. The paired pilot compares each candidate with its own frozen prediction. Regenerated qualitative scores and differences are recorded in figures/qualitative_verification.json.

## Environment

Executed on Windows with Python 3.10.11, PyTorch 2.6.0+cu124, torchvision 0.21.0, timm 0.9.16 and an RTX 4050 Laptop GPU with 6GB VRAM. CUDA must be available. The scripts cache each dataset partition on the GPU; a CPU-only installation will not work without changes.

Use a separate Python environment. Install PyTorch for your platform first, then install timm==0.9.16, numpy, pillow, gdown. Consult the official PyTorch installation instructions if your GPU cannot use the recorded CUDA build. The recorded environment versions are in each run's config.json and environment_versions.json.

## Reproduce from the extracted project root

```powershell
python download_data.py clinic
python download_data.py mass
python train_screening.py --dataset clinic --epochs 200 --batch 16 --amp
python train_screening.py --dataset mass --epochs 200 --batch 16
python prototype_evaluation.py clinic
python prototype_evaluation.py mass
```

Run the experiments sequentially on a 6GB GPU. Downloads go under data/; runs go under runs/. The code retrieves public datasets from their original sources, and dataset redistribution is intentionally excluded. A failed download should be rerun after checking its error. Verify the expected paired counts before training: ClinicDB 489/61/62 train/val/test; Massachusetts 137/4/10 train/valid/test. Every mask must match its image dimensions.

The archive already contains the completed runs. Before reproducing training, rename the existing runs/ directory to recorded-runs/ to preserve the supplied evidence. Start with a fresh runs/ directory; the epoch logger appends records and is not a resume implementation. Run `python verify_evidence.py` to audit the supplied complete records before moving them. Run `python make_figures.py` only after both training runs finish.

The code is pinned to the unmodified official MK-UNet architecture at commit 2fa8b230a057602539af7655203ad434bf56a5b6. architecture_sha256 in config.json must match the shipped mkunet_network.py. The wrapper follows the published AdamW learning rate and weight decay of 1e-4, 200 epochs, no augmentation, and channels [16,32,64,96,160]. ClinicDB uses the paper's multiscale inputs and mixed precision. Massachusetts uses a separate model trained from scratch and a declared low-resolution, single-scale protocol.

## Important differences from the repository defaults

- The current upstream training defaults include augmentation, lr=5e-4, cosine scheduling and batch size 8; the paper describes no augmentation, lr=1e-4, no stated scheduler and batch size 16.
- The wrapper resizes predictions to the true native mask (height,width), rather than treating PIL (width,height) as (height,width).
- Image resizing uses PIL bilinear instead of the upstream OpenCV/Albumentations implementation. The mask binarization rule is >127 instead of the current loader's >20. The recorded value audit counts the pixels affected by this threshold difference; it is not changed after inspecting performance.
- Original masks are scored at native resolution; ground truth is not resized and resized back for final evaluation.
- Each scale is recomputed from the original batch. The upstream loop reassigns images within its scale loop.
- Validation selects the checkpoint; test metrics are not inspected after every epoch.
- ClinicDB checkpoint selection uses native-resolution min-max-normalized Dice to approach the code's scoring convention. The primary reported scores also include fixed sigmoid >=0.5 results.
- One seed is run, rather than the paper's five-run average. Seeds are fixed, but cuDNN benchmarking is enabled and bitwise determinism is not claimed.
- GPU memory includes cached inputs; it is not comparable to an architecture-only inference footprint.

## Metrics and target preprocessing

Dice and IoU average over original images or complete tiles. Per-image min-max normalization is reported separately from fixed sigmoid thresholding. Massachusetts tiles are resized from 1500x1500 to 512x512, split into four 256x256 patches, predicted separately, stitched as logits, and bilinearly resized to the original mask size before sigmoid and thresholding. This is a low-resolution baseline, not a full-resolution remote-sensing state-of-the-art claim.

An all-empty prediction and all-empty target receive Dice=IoU=1. Each raw image is hashed and exact image overlap across partitions is rejected. This does not establish patient/sequence separation in ClinicDB or geographic independence between aerial tiles.

## Pilot protocol

prototype_evaluation.py declares all shifts and update/gate settings before it runs. The one-step pilot updates only BN affine parameters with frozen running statistics. The loss averages foreground/background entropy strata derived from the frozen prediction. It resets source weights for every image or complete four-patch aerial tile.

The gate accepts only if horizontal-flip Bernoulli Jensen-Shannon disagreement improves by >=1e-6 and predicted foreground area changes by <=0.05 of the input. Labels are used only for offline metrics. The ungated candidate is also recorded. This is not a full TENT, EATA or SAR implementation, and it has no input trigger or boundary-localized gate.

## Evidence files

config.json; history.jsonl; data_manifest.json; best.pt; last.pt; results.json; raw console logs; pilot/protocol.json; pilot/per_case.json; pilot/summary.json. results.json contains the checkpoint hash and every original test image/tile metric. Bootstrap intervals in the pilot are paired image/tile intervals and do not model source video or geographical dependence.

The reports distinguish completed experiments from future proposal components. Actual metrics must never be replaced with a paper's published scores. Before emailing, review the analysis and code, check that the external code link works without private account access, and ensure the prose is your own considered interpretation.

## Research sources

- https://arxiv.org/abs/2509.18493
- https://github.com/SLDGroup/MK-UNet
- https://www.cs.toronto.edu/~vmnih/data/
- https://drive.google.com/drive/folders/1FPJr5f91uUCikxMvkwtZSEnYHemTZq1P
- https://arxiv.org/abs/2006.10726
- https://arxiv.org/abs/2204.02610
- https://arxiv.org/abs/2203.13591
- https://arxiv.org/abs/2302.12400
- https://proceedings.mlr.press/v227/valanarasu24a.html
- https://arxiv.org/abs/2504.02008
- https://arxiv.org/abs/2512.02497
- https://arxiv.org/abs/2607.17693

## Target-mask diagnostic and correction

Massachusetts masks are RGB black/red, not black/white. Grayscale conversion maps red to 76 and would erase it at a >127 threshold. The initial target run was stopped after this error appeared in early validation; it is stored under diagnostics/ and none of its scores enter the final results. The corrected run restarts from seed 42 and reads the red channel, with nonempty/nonfull assertions. Every raw mask was checked to contain only black (0,0,0) and red (255,0,0). The initial foreground audit also averaged channels; the corrected audit counts foreground pixels. The pre-fix wrapper is preserved for provenance of the ClinicDB run, whose decoding branch did not change.
