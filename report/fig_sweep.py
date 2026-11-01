import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6,
                     "xtick.color": "#4a4a46", "ytick.color": "#4a4a46"})
INK, GRID = "#2b2b28", "#e4e4df"
d = json.load(open("../results/sweep/sweep.json"))
order = ["natural", "0.6", "0.5", "0.3", "0.1", "0.05", "loas"]
labels = [f"{d[k]['weight_density']:.2f}" if k != "loas" else "LoAS" for k in order]
cats = [("arithmetic", "arithmetic cells", "#2a78d6"), ("other_logic", "clock, control, leakage, indexing", "#eb6834"),
        ("local_sram", "local SRAM", "#1baf7a"), ("global_buffer", "global buffer", "#eda100")]
measured = {"sparten": {"0.6": 396.4, "0.3": 270.0, "loas": 148.6}, "ttfs": {"0.6": 226.2, "0.3": 163.7, "loas": 85.5}}
total = {e: [sum(d[k][e][c] for c, _, _ in cats) for k in order] for e in ("sparten", "ttfs")}
x = np.arange(len(order))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), sharey=True)
for ax, eng, title in zip(axes, ("sparten", "ttfs"), ("SparTen W4/A4", "TTFS-64 S/U, gated (ours)")):
    bottom = np.zeros(len(order))
    for key, name, color in cats:
        v = np.array([d[k][eng][key] for k in order])
        ax.bar(x, v, 0.62, bottom=bottom, color=color, edgecolor="white", linewidth=0.8, label=name, zorder=3)
        bottom += v
    mx = [i for i, k in enumerate(order) if k in measured[eng]]
    ax.scatter(mx, [measured[eng][order[i]] for i in mx], marker="D", s=14, color=INK, zorder=4, label="measured total")
    if eng == "ttfs":
        for i in range(len(order)):
            ax.text(x[i], total["ttfs"][i] + 8, f"{total['sparten'][i] / total['ttfs'][i]:.2f}$\\times$",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, labels)
    ax.set_title(title, fontsize=9, color=INK, pad=3)
    ax.set_xlabel("weight density", fontsize=8, color="#4a4a46", labelpad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("energy per image ($\\mu$J)", fontsize=8)
axes[1].legend(frameon=False, fontsize=7.2, loc="upper right", handlelength=1.2)
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig("fig_sweep.pdf")
