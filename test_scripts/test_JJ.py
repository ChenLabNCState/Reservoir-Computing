import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.JJ import JJ
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from classes.RC import generate_mackey_glass
from pathlib import Path
from collections import defaultdict

def plot_ipc_by_degree(capacities_list, total_capacity=None, figsize=(10, 4.5), save_path=None):
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
training_data = generate_mackey_glass(training_data_size,dt=3,tau=50)
testing_data = generate_mackey_glass(testing_data_size,dt=3,tau=50)
plot_cycles = 200
washout = 50
delay = 1
start_idx = 100



JJ_RC_OP1 = JJ(washout=washout,
           virtual_nodes=20,
           k_inj=.25,
           I_dc=0.5,
           window_size=0,
           delay=delay,
           training_data=training_data[:-delay],
           training_targets=training_data[delay:],
           theta = 1,
           )

JJ_RC_OP2 = JJ(washout=washout,
           virtual_nodes=20,
           k_inj=.15,
           I_dc=1.5,
           window_size=0,
           delay=delay,
           training_data=training_data[:-delay],
           training_targets=training_data[delay:],
           alpha = 1,
           theta = 1,
           )

JJ_RC_OP3 =JJ(washout=washout,
           virtual_nodes=20,
           k_inj=.1,
           I_dc=.95,
           window_size=0,
           delay=delay,
           training_data=training_data[:-delay],
           training_targets=training_data[delay:],
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
    total_C, cap_list = reservoir.evaluate_IPC_me(
    window_max=10,
    time_steps=1000,
    d_max = 4,
    )

    # Render the plots
    plot_ipc_by_degree(cap_list, total_capacity=total_C, save_path=f"OP{i+1}_capacities.png")

    print(f"IPC for reservoir in OP{i+1} is {total_C}" )
    
    # reservoir.train(save_dynamics = True)
    
    # print(f"Weight matrix for OP{i+1} is: \n{reservoir.W}")
    
    # result = reservoir.test(test_data=testing_data[:-delay],test_targets=testing_data[delay:])
    
    # print(f" OP{i+1} has Error of :{result[1]}")
    
    # reservoir.plot(
    # save_dir=os.path.join(Path.cwd().parent,
    #     f"test_plots\\JJ_OP{i+1}"
    # ),
    # save_fig=True
    # )
    # reservoir.plot_dynamics(start_step=start_idx,is_OP2=is_OP2)
    # reservoir.plot_dynamics_2(start_step= start_idx,cycles=200,is_OP2=is_OP2)
    # reservoir.plot_dynamics_3(start_step= start_idx, cycles=5,is_OP2=is_OP2)
    # # reservoir.plot_combined_dynamics2(start_step=100,phase_cycles=200,phase_cycles2=5)


# def run_parameter_sweep(base_params, sweeps: List[SweepSpec], base_dir):

#     mg = generate_mackey_glass(total_length, dt, base_params.tau)

#     if len(sweeps) < 2:
#         raise ValueError("Need at least two sweeps (curve + x-axis).")

#     x_sweep = sweeps[-1]
#     curve_sweep = sweeps[-2]
#     folder_sweeps = sweeps[:-2]

#     # iterate through every combination of outer parameters
#     outer_value_lists = [s.values for s in folder_sweeps]

#     sweep_num = 0
#     curve_num = 0
#     sim_num = 0

#     for outer_vals in product(*outer_value_lists) if outer_value_lists else [()]:

#         params = replace(base_params)

#         save_dir = base_dir

#         # apply outer sweep parameters
#         for sweep_spec, val in zip(folder_sweeps, outer_vals):

#             sweep_num +=1
#             print(f"\n\n ---------- Starting sweep {sweep_num} for {sweep_spec.name}:{val} ----------\n\n")

#             setattr(params, sweep_spec.name, val)

#             save_dir = os.path.join(
#                 save_dir,
#                 f"{sweep_spec.name}_{val}"
#             )

#         os.makedirs(save_dir, exist_ok=True)

#         # save parameters JSON
#         with open(os.path.join(save_dir, "parameters.json"), "w") as f:
#             json.dump(asdict(params), f, indent=4, default=json_converter)

#         error_matrix = []
#         best_predictions = []
#         best_shifts = []

#         for curve_val in curve_sweep.values:

#             curve_num +=1
#             print(f"\n--- Starting curve {curve_num} for {curve_sweep.name}:{curve_val} ---")

#             errors = []
#             predictions = []
#             shift = []


#             for sweep_val in x_sweep.values:

#                 sim_num +=1
#                 print(f"Running sim {sim_num} for {x_sweep.name}:{sweep_val}")

#                 current_params = replace(params)

#                 setattr(current_params, curve_sweep.name, curve_val)
#                 setattr(current_params, x_sweep.name, sweep_val)

#                 y_t, y_p, err = run_simulation(current_params,mg)

#                 errors.append(err)
#                 predictions.append((y_t, y_p))

#             errors = np.array(errors)

#             best_idx = np.argmin(errors)

#             error_matrix.append(errors)
#             best_predictions.append(predictions[best_idx][1])


#         error_matrix = np.array(error_matrix)

#         plot_and_save(
#             mg,
#             x_sweep,
#             curve_sweep,
#             x_sweep.values,
#             curve_sweep.values,
#             error_matrix,
#             best_predictions,
#             save_dir
#         )









