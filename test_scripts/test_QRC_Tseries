import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import qutip as qt
from classes.QRC import QRC
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from classes.RC import generate_mackey_glass

def custom_pauli(N, state_a_index, state_b_index):
    sx_mat = np.zeros((N, N), dtype=complex)
    sy_mat = np.zeros((N, N), dtype=complex)
    sz_mat = np.zeros((N, N), dtype=complex)

    sx_mat[state_a_index, state_b_index] = 1.0
    sx_mat[state_b_index, state_a_index] = 1.0

    sy_mat[state_a_index, state_b_index] = -1j
    sy_mat[state_b_index, state_a_index] = 1j

    sz_mat[state_a_index, state_a_index] = 1.0
    sz_mat[state_b_index, state_b_index] = -1.0

    return [qt.Qobj(sx_mat), qt.Qobj(sy_mat), qt.Qobj(sz_mat)]

def custom_dissipator(N, destroy_index):
    destroy = np.zeros((N, N), dtype=complex)
    if destroy_index >= 1:
        destroy[destroy_index-1, destroy_index] = 1
        destroy = qt.Qobj(destroy)
    return destroy

#Run a test with 3 level system and compare to Fock state
N_dim = 3
pulse_duration = 5
pulse_time_steps = 50
window_size = 5
#3level system

plot_cycles = 200
washout = 0
delay = 1
start_idx = 100
training_data_size = 1000
testing_data_size = 500

training_data = generate_mackey_glass(training_data_size,dt=3,tau=50)
testing_data = generate_mackey_glass(testing_data_size,dt=3,tau=50)


training_targets = training_data[window_size+delay:]
testing_targets = testing_data[window_size+delay:]

QRC_test = QRC_time()