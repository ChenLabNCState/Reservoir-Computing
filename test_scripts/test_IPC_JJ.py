import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.JJ import JJ
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict



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
    





