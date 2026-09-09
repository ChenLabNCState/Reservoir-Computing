"""
IPC sweep over injection scaling parameter i (ι) for the driven logistic map.
Reproduces the stacked bar chart from Appeltant et al., Sci. Rep. 2, 514 (2012).
 
Run from the project root (same directory that contains your `classes/` folder):
    python ipc_sweep.py
"""
 
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
 
# ── Make sure your classes/ folder is importable ──────────────────────────────
sys.path.insert(0, ".")
from classes.RC import RC_TimeSeries
from classes.logistic_map import LogisticMapRC
 
 
# ── Sweep settings ────────────────────────────────────────────────────────────
I_VALUES  = np.round(np.arange(0.02, 0.21, 0.02), 4)   # ι = 0.02 … 0.20
RHO       = 2.5          # paper's bifurcation parameter
D_MAX     = 4            # degrees 1–4
TAU_MAX   = 10           # max delay for IPC targets
DATA_SIZE = 5000         # IPC time series length
WASHOUT   = 200          # must be ≥ TAU_MAX
 
# ── Colour map: dark blue → cyan → yellow/orange (matching paper) ─────────────
DEGREE_COLORS = ["#1a3a8f", "#2176c8", "#5cb8e8", "#c8e83c"]
 
 
# ── Run sweep ─────────────────────────────────────────────────────────────────
print(f"Sweeping ι over {I_VALUES}")
print(f"rho={RHO}, d_max={D_MAX}, tau_max={TAU_MAX}, data_size={DATA_SIZE}\n")
 
results = np.zeros((len(I_VALUES), D_MAX))
dummy_train = np.random.default_rng(0).uniform(-1, 1, WASHOUT + 10)
 
for idx, i_inj in enumerate(I_VALUES):
    print(f"[{idx+1}/{len(I_VALUES)}]  ι = {i_inj:.2f}", flush=True)
 
    rc = LogisticMapRC(
        training_data = dummy_train,
        washout       = WASHOUT,
        rho           = RHO,
        i_inj         = i_inj,
        initial_x     = 0.5,
    )
 
    total, cap_array = rc.evaluate_IPC_joint(
        data_size = DATA_SIZE,
        d_max     = D_MAX,
        tau_max   = TAU_MAX,
        threshold = 0
    )
 
    results[idx] = cap_array
    print(f"           total IPC = {total:.4f}  |  by degree: {cap_array}")
 
print("\nSweep complete.")
 
 
# ── Normalise to relative capacity ────────────────────────────────────────────
row_totals = results.sum(axis=1, keepdims=True)
row_totals_safe = np.where(row_totals == 0, 1, row_totals)
results_rel = results / row_totals_safe
 
 
# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
 
x = np.arange(len(I_VALUES))
bar_width = 0.7
bottoms = np.zeros(len(I_VALUES))
 
for d in range(D_MAX):
    ax.bar(
        x,
        results_rel[:, d],
        width     = bar_width,
        bottom    = bottoms,
        color     = DEGREE_COLORS[d],
        edgecolor = "black",
        linewidth = 0.6,
    )
    bottoms += results_rel[:, d]
 
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.2f}" for v in I_VALUES], fontsize=8)
ax.set_xlabel("Scaling parameter ι", fontsize=11)
ax.set_ylabel("Capacity (relative)", fontsize=11)
ax.set_title("Logistic map", fontsize=12)
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.grid(axis="y", linestyle="--", alpha=0.3)
 
# ── Colorbar legend (matching paper's right-side gradient) ───────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "degree_cmap", DEGREE_COLORS, N=256
)
norm = mcolors.Normalize(vmin=1, vmax=D_MAX)
 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cax = inset_axes(ax, width="5%", height="70%", loc="center right",
                 bbox_to_anchor=(0.12, 0, 1, 1), bbox_transform=ax.transAxes,
                 borderpad=0)
cb  = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical")
cb  = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
cb.set_label("degree", fontsize=10)
cb.set_ticks(range(1, D_MAX + 1))
cb.set_ticklabels([str(d) for d in range(1, D_MAX + 1)])
 
plt.subplots_adjust(right=0.88)   # make room for colorbar
 
out_path = "ipc_logistic_sweep.png"
plt.savefig(out_path, bbox_inches="tight")
print(f"Plot saved → {out_path}")
plt.show()
 
