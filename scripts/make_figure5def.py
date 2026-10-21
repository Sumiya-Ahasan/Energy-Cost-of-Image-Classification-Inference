# Makes the Pareto figures with log axes and batch-size labels (replaces old Figures 5d-5f).
# Input : all_sessions_results.csv  (main sweep: 5 sessions, 480 records)
# Usage : python make_figure5def.py all_sessions_results.csv
# Output: one file per model, Figure_pareto_<model>.pdf/.png (single panel each)
# Each point is one configuration (mean over the 5 sessions). The black line joins the
# Pareto-optimal points (no other configuration of the same model has both lower
# energy per prediction and lower batch latency).
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter

d = pd.read_csv(sys.argv[1]); d = d[d.valid == True]
m = d.groupby(["model", "batch_size", "precision"]).agg(
    E=("energy_per_sample_j", "mean"), L=("mean_latency_s", "mean")).reset_index()
m["E"] *= 1000; m["L"] *= 1000            # mJ per prediction, ms per batch

names = {"mobilenet_v3_small": "MobileNetV3-Small", "efficientnet_b0": "EfficientNet-B0",
         "resnet18": "ResNet-18", "resnet50": "ResNet-50",
         "convnext_tiny": "ConvNeXt-Tiny", "vit_b_16": "ViT-B/16"}

def pareto(g):
    keep = []
    for i, r in g.iterrows():
        dom = ((g.E <= r.E) & (g.L <= r.L) & ((g.E < r.E) | (g.L < r.L))).any()
        keep.append(not dom)
    return g[keep]

tot = 0
for mod in names:
    p = pareto(m[m.model == mod]); tot += len(p)
    print(names[mod], len(p), "Pareto points:", dict(p.precision.value_counts()))
print("total Pareto points:", tot)

FP = {"fp32": ("#d98e04", "o", "FP32"), "fp16": ("#1f4e79", "s", "FP16")}
def draw_one(mod):
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.9))
    g = m[m.model == mod]
    for pr, (c, mk, lab) in FP.items():
        s = g[g.precision == pr]
        ax.scatter(s.L, s.E, c=c, marker=mk, s=34, label=lab, zorder=3, edgecolor="white", linewidth=0.5)
        for _, r in s.iterrows():
            ax.annotate(str(int(r.batch_size)), (r.L, r.E), textcoords="offset points",
                        xytext=(4, 4) if pr == "fp32" else (4, -9), fontsize=6.5, color=c)
    p = pareto(g).sort_values("L")
    ax.step(p.L, p.E, where="post", color="black", lw=0.9, zorder=2, label="Pareto front")
    ax.set_xscale("log"); ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=[1, 2, 5], numticks=20))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(LogLocator(base=10, subs=range(1, 10), numticks=20))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Batch execution latency (ms)"); ax.set_ylabel("Energy per prediction (mJ)")
    ax.set_title(names[mod], fontsize=10)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("Figure_pareto_%s.png" % mod, dpi=250); fig.savefig("Figure_pareto_%s.pdf" % mod)
    plt.close(fig)
for mod in names:
    draw_one(mod)
