Measuring the Energy Cost of Image Classification Inference
Evaluation of Model Architecture, Batch Size, and Numerical Precision on a Tesla T4 GPU.
About
This repository contains the data, analysis notebooks, scripts, and figures for our study on GPU energy consumption during deep-learning inference. We measure per-sample energy (Joules), latency, and throughput for six ImageNet classifiers (ResNet-18, ResNet-50, MobileNetV3-Small, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16) across eight batch sizes (1–512) in FP32 and FP16 on a Kaggle Tesla T4 (70 W power cap).
Repository Structure
├── data/
│   ├── all_sessions_results.csv      # Aggregated results from all 5 sessions (480 records)
│   ├── session6_results.csv          # Session 6 results (thermal-soak validation)
│   ├── power_traces/                 # Raw NVML power traces per configuration
│   └── predictions/                  # Model prediction outputs for accuracy check
│
├── notebooks/
│   ├── main-session-v1.ipynb         # Main analysis: energy, latency, throughput, statistics
│   ├── session-6-v2.ipynb            # Session 6 analysis (thermal soak, NVML cross-check)
│   └── accuracy-imagenet.ipynb       # ImageNet validation accuracy (FP32 vs FP16)
│
├── scripts/
│   ├── make_figure4d.py              # Figure 4d: ImageNet accuracy bar chart
│   ├── make_figure5a.py              # Figure 5a: FP16 energy reduction heatmap
│   ├── make_figure5def.py            # Figures 5d, 5e, 5f: Pareto frontier plots
│   └── reviewer_reanalysis.py        # Reviewer-requested reanalysis script
│
├── results/
│   ├── anova_with_session_block.csv          # Three-way factorial ANOVA results
│   ├── baseline_comparison.csv               # Idle-power baseline comparison
│   ├── codecarbon_validation.csv             # CodeCarbon cross-check (full system)
│   ├── codecarbon_validation_gpu_only.csv    # CodeCarbon cross-check (GPU only)
│   ├── fp16_reduction_by_model_batch_with_ci.csv  # FP16 energy reduction with CIs
│   ├── friedman_kendall.txt                  # Friedman test and Kendall W results
│   ├── pareto_frontier.csv                   # Pareto-optimal configurations
│   ├── repeatability_friedman_test.csv       # Session repeatability test
│   ├── session_offsets.csv                   # Inter-session offset values
│   ├── Table_A1_compact_regime_summary.csv   # Table A1: regime summary
│   ├── Table_A3_all_configurations_mean_over.csv  # Table A3: all 96 config means
│   ├── Table_imagenet_val_accuracy.csv       # ImageNet validation accuracy table
│   └── wilcoxon_sessions_holm.csv            # Wilcoxon signed-rank with Holm correction
│
└── figures/
    ├── methodology_architecture.png          # Figure 1: measurement architecture
    ├── methodology_six_phase.png             # Figure 2: six-phase protocol
    ├── Figure3_flowchart.png                 # Figure 3: data processing flowchart
    ├── Figure4d_imagenet.png                 # Figure 4d: accuracy bar chart
    ├── Figure5a_fp16_reduction.png           # Figure 5a: FP16 reduction heatmap
    ├── Figure_pareto_*.png                   # Figures 5d–5f: per-model Pareto frontiers
    └── notebook_plots/                       # Additional plots from notebooks

Hardware and Setup
Item
Detail
GPU
NVIDIA Tesla T4 (16 GB, Turing)
Power cap
70 W (enforced via nvidia-smi)
Platform
Kaggle Notebooks (2× Intel Xeon @ 2.00 GHz, 13 GB RAM)
Framework
PyTorch 2.4, torchvision, CUDA 12.1
Power sampling
NVML at ~100 ms polling, trapezoidal integration
Cross-check
CodeCarbon v2.3
Dataset
ImageNet-1K validation set (50,000 images)

How to Reproduce
Clone this repository

 git clone https://github.com/Sumiya-Ahasan/Energy-Cost-of-Image-Classification-Inference.git


Install dependencies

 pip install torch torchvision pandas numpy scipy matplotlib seaborn codecarbon


Run the notebooks in this order:


notebooks/main-session-v1.ipynb — main energy/latency analysis and statistical tests
notebooks/session-6-v2.ipynb — session 6 thermal-soak validation
notebooks/accuracy-imagenet.ipynb — FP32 vs FP16 accuracy comparison
Generate figures

 python scripts/make_figure4d.py
python scripts/make_figure5a.py
python scripts/make_figure5def.py


Key Findings
Energy per sample follows a two-regime model: latency-bound at small batches, throughput-bound at large batches.
FP16 reduces energy by 25.2% on average (net of floor), with the largest savings in the throughput-bound regime.
The three-way interaction (architecture × batch size × precision) is statistically significant (ANOVA, p < 0.001).
Measurement repeatability is strong: median CV = 5.1%, Friedman χ²(4) = 260.64.
NVML cumulative counter and trapezoidal integral agree within 3% (mean difference +0.4%).
FP16 does not degrade ImageNet top-1 accuracy beyond ±0.1 percentage points (TOST equivalence test).




