import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.JJ import JJ
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def plot_ipc_by_degree(capacities_list, total_capacity:float, figsize=(10, 4.5), save_path=None):
    """
    Plots IPC breakdown grouped by degree D = sum(degrees), showing absolute capacity
    and fractional contribution per degree similar to Dambre et al. (2012).
    
    Parameters:
        capacities_dict (dict): Dictionary mapping ((deg1, deg2, ...), (tau1, tau2, ...)) -> C_i
        total_capacity (float, optional): Sum of all capacities. Calculated if None.
        figsize (tuple): Dimensions of the figure.
        save_path (str, optional): File path to save the plot.
    """


    # 2. Calculate fractional capacities per degree
    fractions = capacities_list / total_capacity if total_capacity > 0 else np.zeros_like(capacities_list)

    # 3. Create two-panel plot (Absolute & Fractional)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(capacities_list)))

    # Panel 1: Absolute Capacity per Degree (C_D)
    bars1 = ax1.bar([f"Deg {d}" for d in range(1,len(capacities_list)+1)], capacities_list, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Absolute Capacity $C_D$", fontsize=11)
    ax1.set_title("Capacity per Degree $D$", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Add numeric labels on top of absolute capacity bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(capacities_list), 
                 f"{yval:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Fraction of Total Capacity (C_D / C_total)
    bars2 = ax2.bar([f"Deg {d}" for d in range(1,len(capacities_list)+1)], fractions * 100, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Fraction of Total Capacity (%)", fontsize=11)
    ax2.set_title(f"Degree Contribution ($C_{{total}} = {total_capacity:.2f}$)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Add percentage labels on top of fractional bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, 
                 f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



training_data_size = 1000
testing_data_size =250
plot_cycles = 200
washout = 50
delay = 1
start_idx = 100



JJ_RC_OP1 = JJ(washout=washout,
           virtual_nodes= 20,
           k_inj=.25,
           I_dc=0.5,
           delay=delay,
           theta = 1,
           )

JJ_RC_OP2 = JJ(washout=washout,
           virtual_nodes=70,
           k_inj=.15,
           I_dc=1.5,
           delay=delay,
           alpha = 1,
           theta = 1,
           )

JJ_RC_OP3 =JJ(washout=washout,
           virtual_nodes=20,
           k_inj=.1,
           I_dc=.95,
           delay=delay,
           alpha = 1,
           theta = 1,
           )

reservoir_list = [JJ_RC_OP1,JJ_RC_OP2,JJ_RC_OP3]
# reservoir_list=[JJ_RC_OP2]

for (i,reservoir) in enumerate(reservoir_list):
    is_OP2 = False
    if i == 1:
        is_OP2 = True


    # Run IPC evaluation
    total_C, cap_list = reservoir.evaluate_IPC(
        data_size=5000,
        d_max=4,
        tau_max=10,
        threshold=0,
    )

    # Render the plots
    plot_ipc_by_degree(cap_list, total_capacity=total_C, save_path=f"OP{i+1}_capacities.png")

    print(f"IPC for reservoir in OP{i+1} is {total_C}" )
    





