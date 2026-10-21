# Makes Figure 4(d): FP32 vs FP16 top-1 accuracy on ImageNet-1k validation.
# Input : accuracy_imagenet_val.csv  (from the Kaggle accuracy notebook "green_accuracy_imagenet")
#         columns: image_id, model, label, pred_fp32, pred_fp16, conf_fp32, conf_fp16
# Usage : python make_figure4d.py accuracy_imagenet_val.csv
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv(sys.argv[1])
acc = d.groupby("model").apply(lambda g: pd.Series({
    "fp32": 100 * (g.pred_fp32 == g.label).mean(),
    "fp16": 100 * (g.pred_fp16 == g.label).mean()}))
acc = acc.sort_index()
print(acc.round(2))

x = np.arange(len(acc)); w = 0.38
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.bar(x - w/2, acc.fp32, w, label="fp32", color="#e76f51")
ax.bar(x + w/2, acc.fp16, w, label="fp16", color="#2a9d8f")
ax.set_xticks(x); ax.set_xticklabels(acc.index, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 100); ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("FP32 vs FP16 top-1 accuracy by model (ImageNet-1k validation, 50,000 images)", fontsize=9)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("Figure4d_imagenet_accuracy.png", dpi=250)
fig.savefig("Figure4d_imagenet_accuracy.pdf")
