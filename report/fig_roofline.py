import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6,
                     "xtick.color": "#4a4a46", "ytick.color": "#4a4a46"})
INK, GRID, ROOF = "#2b2b28", "#e4e4df", "#8a8a85"
SWEEP = json.load(open("../results/sweep/sweep.json"))
LAYERS = ["features.3", "features.7", "features.10", "features.14", "features.17", "features.20",
          "features.24", "features.27", "features.30", "features.34", "features.37", "features.40"]
PEAK = {"sparten": 32, "ttfs": 64}          # useful ops per cycle: 32 MAC units, 64 lanes
COLOR = {"sparten": "#eb6834", "ttfs": "#2a78d6"}
NAME = {"sparten": "SparTen W4/A4", "ttfs": "TTFS-64 (ours)"}
BW = 8                                      # weight port: one 64-bit beat per cycle


def point(density, engine, layer):
    s = SWEEP[density][engine]["layers"][layer]
    return s["ops"] / (8 * s["w_beats"]), s["ops"] / s["cycles"]


fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45), sharey=True)
oi = np.logspace(-1, 2.3, 200)
for ax, density, title in zip(axes, ("natural", "0.05"), ("weight density 0.63", "weight density 0.05")):
    for eng in ("sparten", "ttfs"):
        ax.plot(oi, np.minimum(BW * oi, PEAK[eng]), color=COLOR[eng], lw=1.2, zorder=2)
        pts = np.array([point(density, eng, layer) for layer in LAYERS])
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=COLOR[eng], edgecolor="white", linewidth=0.6, zorder=4,
                   label=NAME[eng])
        if eng == "ttfs":
            for i, layer in ((0, "f3"), (11, "f40")):
                ax.annotate(layer, pts[i], xytext=(4, -9), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(0.12, 160)
    ax.set_ylim(0.7, 100)
    ax.set_xticks([0.25, 1, 4, 16, 64], ["1/4", "1", "4", "16", "64"])
    ax.set_yticks([1, 4, 16, 32, 64], ["1", "4", "16", "32", "64"])
    ax.minorticks_off()
    ax.set_title(title, fontsize=9, color=INK, pad=3)
    ax.set_xlabel("useful ops per weight-port byte", fontsize=8, color="#4a4a46", labelpad=2)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.text(0.14, 14, "weight port\n8 B/cycle", fontsize=7, color="#4a4a46")
    ax.text(150, 70, "64 lanes", fontsize=7, color="#4a4a46", ha="right")
    ax.text(150, 35, "32 units", fontsize=7, color="#4a4a46", ha="right")
axes[0].set_ylabel("useful ops per cycle", fontsize=8)
axes[0].legend(frameon=False, fontsize=7.2, loc="lower right", handletextpad=0.3)
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig("fig_roofline.pdf")
for density in ("natural", "0.05"):
    for eng in ("sparten", "ttfs"):
        print(density, eng, [tuple(round(v, 1) for v in point(density, eng, l)) for l in LAYERS])
