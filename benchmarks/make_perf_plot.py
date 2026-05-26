"""Generate the README's headline latency comparison plot.

Pulls numbers from the committed bench CSVs and produces a single chart
showing latency-per-tick across backends and (K, H) configurations.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Numbers from benchmarks/results/ — committed CPU baseline + the v0 GPU run.
# DATA = {
#     "pytorch_cpu":  {(256,  40): 104.68, (1024, 40): 174.45, (4096, 40): 411.41},
#     "pytorch_cuda": {(1024, 40):  87.49, (1024, 80): 174.60,
#                      (4096, 40):  85.83, (4096, 80): 169.40,
#                      (16384, 40): 86.03, (16384, 80): 170.16},
#     "cuda_kernel":  {(1024, 40):   0.30, (1024, 80):   0.31,
#                      (4096, 40):   0.31, (4096, 80):   0.30,
#                      (16384, 40):  0.39, (16384, 80):  0.73},
# }

data = pd.read_csv('./benchmarks/results/latency_20260525_215419.csv')

DATA = {}

for i, row in data.iterrows():
    print(row)
    if row['backend'] not in DATA:
        DATA[row['backend']] = {}
    DATA[row['backend']][(row['K'], row['H'])] = row['mean_ms']


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # ----- Left: grouped bar at H=40 across K -----
    Ks = [1024, 4096, 16384]
    backends = ["pytorch_cpu", "pytorch_cuda", "cuda_kernel"]
    colors = {"pytorch_cpu": "#888888", "pytorch_cuda": "#4477AA", "cuda_kernel": "#CC3322"}
    bar_w = 0.27
    x = np.arange(len(Ks))

    for i, b in enumerate(backends):
        ys = [DATA[b].get((k, 40), np.nan) for k in Ks]
        ax1.bar(x + (i - 1) * bar_w, ys, bar_w, label=b, color=colors[b], log=True)

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}" for k in Ks])
    ax1.set_ylabel("per-tick latency (ms, log)")
    ax1.set_title("MPPI rollout latency, H=40")
    ax1.axhline(5.0, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax1.text(0.05, 5.4, "5 ms (200 Hz real-time)", color="green", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", axis="y", alpha=0.25)

    # Annotate the speedup at K=1024 between cpu and kernel
    # cpu_ms = DATA["pytorch_cpu"][(1024, 40)]
    cpu_ms = DATA["pytorch_cuda"][(1024, 40)]
    ker_ms = DATA["cuda_kernel"][(1024, 40)]
    ax1.annotate(f"{cpu_ms / ker_ms:.0f}× vs torch(GPU)",
                 xy=(0 + bar_w, ker_ms), xytext=(0.45, 1.5),
                 fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.7))

    # ----- Right: kernel scaling with K, H=40 vs H=80 -----
    Ks_kernel = sorted({k for (k, h) in DATA["cuda_kernel"]})
    for h, marker, color in [(40, "o", "#CC3322"), (80, "s", "#AA1100")]:
        ys = [DATA["cuda_kernel"][(k, h)] for k in Ks_kernel]
        ax2.plot(Ks_kernel, ys, marker=marker, color=color,
                 label=f"H={h}", linewidth=1.6, markersize=7)

    ax2.set_xscale("log")
    ax2.set_xticks(Ks_kernel)
    ax2.set_xticklabels([f"{k:,}" for k in Ks_kernel])
    ax2.set_xlabel("K (rollouts per tick)")
    ax2.set_ylabel("per-tick latency (ms)")
    ax2.set_title("Kernel scaling")
    ax2.axhline(5.0, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(1100, 5.2, "5 ms budget", color="green", fontsize=8)
    ax2.set_ylim(0, 6)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.suptitle("Fused MPPI rollout: NVIDIA RTX A5000",
                 fontsize=11, fontweight="bold", y=1.0)
    plt.tight_layout()

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "perf_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
