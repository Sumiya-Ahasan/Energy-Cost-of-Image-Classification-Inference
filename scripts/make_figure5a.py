# Makes Figure 5(a): mean FP16 energy reduction relative to FP32, by batch regime.
# Input : all_sessions_results.csv  (main sweep: 5 sessions, 480 records)
# Usage : python make_figure5a.py all_sessions_results.csv
# Method: reduction = 100*(1 - E_fp16/E_fp32) per model, batch and session;
#         averaged over the batch sizes of each regime inside every session;
#         mean over the 5 sessions with a paired 95% t-interval (df = 4).
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

d = pd.read_csv(sys.argv[1])
p = d.pivot_table(index=["model", "batch_size", "session_id"], columns="precision",
                  values="energy_per_sample_j").dropna()
p["red"] = 100 * (1 - p["fp16"] / p["fp32"])

order = ["mobilenet_v3_small", "efficientnet_b0", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
names = {"mobilenet_v3_small": "MobileNetV3-Small", "efficientnet_b0": "EfficientNet-B0",
         "resnet18": "ResNet-18", "resnet50": "ResNet-50",
         "convnext_tiny": "ConvNeXt-Tiny", "vit_b_16": "ViT-B/16"}
res = {}
for label, keep in [("small", lambda b: b <= 4), ("large", lambda b: b >= 32)]:
    q = p[keep(p.index.get_level_values(1))].groupby(["model", "session_id"]).red.mean().reset_index()
    o = q.groupby("model").red.agg(["mean", "std", "count"])
    o["ci"] = stats.t.ppf(0.975, o["count"] - 1) * o["std"] / np.sqrt(o["count"])
    res[label] = o.loc[order]
    print(label); print(o.loc[order, ["mean", "ci"]].round(1))

x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(x - w/2, res["small"]["mean"], w, yerr=res["small"]["ci"], capsize=3,
       label=r"Batch $\leq$ 4", color="#e9c46a", edgecolor="#555", linewidth=0.6)
ax.bar(x + w/2, res["large"]["mean"], w, yerr=res["large"]["ci"], capsize=3,
       label=r"Batch $\geq$ 32", color="#264653", edgecolor="#222", linewidth=0.6)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([names[m] for m in order], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("FP16 energy reduction vs FP32 (%)")
ax.set_title("Mean FP16 energy reduction by model and batch regime", fontsize=10)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("Figure5a_fp16_reduction_by_regime.png", dpi=250)
fig.savefig("Figure5a_fp16_reduction_by_regime.pdf")
