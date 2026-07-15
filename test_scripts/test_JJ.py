import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.JJ import JJ
import qutip as qt
import numpy
import matplotlib.pyplot as plt
from classes.RC import generate_mackey_glass
from pathlib import Path



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
           alpha = 1,
           theta = 1,
           )

JJ_RC_OP2 = JJ(washout=washout,
           virtual_nodes=70,
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
    
    reservoir.train(save_dynamics = True)
    
    print(f"Weight matrix for OP{i+1} is: \n{reservoir.W}")
    
    result = reservoir.test(test_data=testing_data[:-delay],test_targets=testing_data[delay:])
    
    print(f" OP{i+1} has Error of :{result[1]}")
    
    reservoir.plot(
    save_dir=os.path.join(Path.cwd().parent,
        f"test_plots\\JJ_OP{i+1}"
    ),
    save_fig=True
    )
    reservoir.plot_dynamics(start=start_idx,is_OP2=is_OP2)
    reservoir.plot_dynamics_2(start = start_idx,cycles=200,is_OP2=is_OP2)
    reservoir.plot_dynamics_3(start = start_idx, cycles=5,is_OP2=is_OP2)
    # reservoir.plot_combined_dynamics2(start_step=100,phase_cycles=200,phase_cycles2=5)










