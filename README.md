# Measuring the Energy Cost of Image Classification Inference

A Green Computing evaluation of model architecture, batch size, and
numerical precision on an NVIDIA T4 GPU.

## Overview

This repository contains the complete measurement code, raw data, and
analysis for a study characterizing how model architecture, batch
size, and numerical precision (FP32/FP16) jointly affect energy,
latency, throughput, and accuracy during image-classification
inference on a single NVIDIA T4 GPU.

- 6 architectures: MobileNetV3-Small, EfficientNet-B0, ResNet-18,
  ResNet-50, ConvNeXt-Tiny, ViT-B/16
- 8 batch sizes: 1, 2, 4, 8, 16, 32, 64, 128
- 2 precisions: FP32, FP16
- 5 independent measurement sessions (480 total configurations)
- Accuracy evaluated on Imagenette v2 and Imagewoof v2 (7,854 images
  per model)

## Repository structure

- `notebooks/` — the full measurement and analysis pipeline, as run
  on Kaggle (NVIDIA T4 ×2 environment)
- `data/raw/` — the primary energy/latency measurement records
  (`all_sessions_results.csv`), one row per (session, model, batch
  size, precision) configuration
- `data/processed/` — all derived analysis outputs (baseline
  comparison, Pareto frontier, McNemar's exact test, factorial ANOVA,
  repeatability tests, CodeCarbon cross-validation)
- `figures/` — final figures used in the paper
- `paper/` — the final paper PDF

## Key findings

- FP16 reduced energy per prediction by 7.6%–77.6% depending on
  architecture (Table 2)
- A three-way factorial ANOVA confirmed a significant Model ×
  Precision interaction (partial η² = 0.95, p < 0.001)
- McNemar's exact test found no statistically significant accuracy
  change from FP16 for any architecture (all p ≥ 0.375, n = 7,854
  images per model)
- Independent cross-validation against CodeCarbon (GPU-only scope)
  confirmed close agreement with our energy measurements (0.36%
  mean difference)

See the paper (`paper/Green_Paper.pdf`) for full methodology, results,
and discussion.

## Reproducing the experiment

1. Open `notebooks/green_computing_experiment.ipynb` in a Kaggle
   environment with GPU access (the pipeline uses NVML via
   `nvidia-smi` and will not run correctly without an NVIDIA GPU).
2. Install dependencies: `pip install -r requirements.txt`
3. Run cells in order. Toggles at the top of the first cell
   (`RUN_ENERGY_SWEEP`, `RUN_ACCURACY_EVAL`, `RUN_EXTRA_SESSION`,
   `RUN_CODECARBON_VALIDATION`) control which stages execute; set
   only the stage(s) you want to (re)run.
4. The full energy sweep (96 configurations × 60 seconds) takes
   approximately 1.5–2 hours per session.

## Reproducing the analysis only

If you only want to reproduce the statistical analysis and figures
from the already-collected data (no GPU required for this part),
load `data/raw/all_sessions_results.csv` and
`data/processed/accuracy_imagenette_imagewoof_combined.csv` and run
the analysis cells in the notebook (set all `RUN_*` toggles to
`False`).

## Data dictionary

See `data/raw/all_sessions_results.csv` columns:

| Column | Description |
|---|---|
| session_id | Identifier for the independent measurement session |
| model | Model architecture |
| batch_size | Batch size (1–128) |
| precision | fp32 or fp16 |
| energy_j | Total energy for the measurement window (Joules) |
| energy_per_sample_j | Energy per prediction (Joules) |
| mean_latency_s | Mean batch latency (seconds) |
| throughput_samples_s | Samples processed per second |
| peak_memory_gb | Peak GPU memory allocated (GB) |
| valid | Whether the configuration passed validity checks |
| warnings | Any flagged warnings for this configuration |

(Additional columns cover temperature, clock frequency, utilization,
throttle-state fractions, and idle-power baselines — see the paper's
Implementation section, §4, for details.)

## Citation

If you use this code or data, please cite the paper (see
`CITATION.cff`).

## License

Code is released under the MIT License (see `LICENSE`). Data files
are released under CC BY 4.0.
