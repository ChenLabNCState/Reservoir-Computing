import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import qutip as qt
from classes.QRC import QRC
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from classes.RC import generate_mixed_amplitude_sequence

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

training_data,training_targets = QRC.generate_mixed_amplitude_sequence(noise_level=0.0)

testing_data,testing_targets = QRC.generate_mixed_amplitude_sequence(noise_level=0)

training_targets = training_targets[window_size-1:]
testing_targets = testing_targets[window_size-1:]



start_pulse_duration = 1
end_pulse_duration= 11
pulse_features = 5
base_pulse_duration = 5

# 1. Enclose the loop in brackets [] to make it a valid list comprehension first
pulse_durations_list = [
    np.linspace(base_pulse_duration, max_val, window_size) 
    for max_val in np.linspace(start_pulse_duration, end_pulse_duration, pulse_features)
]

# 2. Convert the clean list of lists into a 2D NumPy array
pulse_durations = np.array(pulse_durations_list)

def test_3level_base():
    subspace_index_offset = 1
    kappa_low = 0
    kappa_high = .5
    measurement_ops = custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)[1:]
    sigma_x_subspace, sigma_y_subpace, sigma_z_subpace= custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)

    destroy_ground = custom_dissipator(N_dim, subspace_index_offset)
    destroy_excited = custom_dissipator(N_dim, subspace_index_offset + 1)

    c_ops = [
        np.sqrt(kappa_low) * destroy_ground,
        np.sqrt(kappa_high) * destroy_excited
    ]

    initial_state = qt.fock(N_dim, 1)

    H_int = sigma_x_subspace

    QRC_3level= QRC.QRC_Classification(N=N_dim,
                                                collapse_ops=c_ops,
                                                H_interaction=H_int,
                                                measurement_ops=measurement_ops,
                                                initial_state=initial_state,
                                                training_data=training_data,
                                                training_targets=training_targets,
                                                classification_dim=2,
                                                subspace_dim=2,
                                                subspace_start_index=subspace_index_offset,
                                                window_size=window_size
                                                )

    QRC_3level.train()

    _, error = QRC_3level.test(test_data=testing_data,test_targets=testing_targets)

    print(f"Trial run for 3_level with error of {error}")

    QRC_3level.plot(save_dir=os.path.join(os.getcwd(),"test_plots\\3level_base"),
                    save_fig=True)
    return


def test_3level_upgraded(local_dir, kappa_low = 4,kappa_high = .1):


    subspace_index_offset = 1

    measurement_ops = custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)[1:]
    sigma_x_subspace, sigma_y_subpace, sigma_z_subpace= custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)

    destroy_ground = custom_dissipator(N_dim, subspace_index_offset)
    destroy_excited = custom_dissipator(N_dim, subspace_index_offset + 1)

    c_ops = [
        np.sqrt(kappa_low) * destroy_ground,
        np.sqrt(kappa_high) * destroy_excited
    ]

    initial_state = qt.fock(N_dim, 1)

    H_int = sigma_x_subspace

        # 1. Enclose the loop in brackets [] to make it a valid list comprehension first
    pulse_durations_list = [
        np.linspace(base_pulse_duration, max_val, window_size) 
        for max_val in np.linspace(start_pulse_duration, end_pulse_duration, pulse_features)
    ]

    # 2. Convert the clean list of lists into a 2D NumPy array
    pulse_durations = np.array(pulse_durations_list)
    QRC_3level= QRC.QRC_Classification_upgraded(N=N_dim,
                                                collapse_ops=c_ops,
                                                H_interaction=H_int,
                                                measurement_ops=measurement_ops,
                                                initial_state=initial_state,
                                                training_data=training_data,
                                                training_targets=training_targets,
                                                classification_dim=2,
                                                subspace_dim=2,
                                                subspace_start_index=subspace_index_offset,
                                                window_size=window_size,
                                                pulse_durations=pulse_durations
                                                )

    QRC_3level.train()

    _, error = QRC_3level.test(test_data=testing_data,test_targets=testing_targets)

    print(f"Trial run for 3_level with error of {error}")

    QRC_3level.plot(save_dir=os.path.join(os.getcwd(),local_dir),
                    save_fig=True)
    return


def test_fock_upgraded():
    N = 2
    subspace_dim = 2
    initial_state = qt.fock(N)
    measurement_ops = []
    kappa = 1
    for i in range(subspace_dim):
        measurement_ops.append(qt.fock_dm(N,i))


    H_int = qt.create(N) + qt.destroy(N)

    c_ops = [kappa*qt.destroy(N)]

    QRC_fock= QRC.QRC_Classification_upgraded(N=N,
                                                collapse_ops=c_ops,
                                                H_interaction=H_int,
                                                measurement_ops=measurement_ops,
                                                initial_state=initial_state,
                                                training_data=training_data,
                                                training_targets=training_targets,
                                                classification_dim=2,
                                                window_size=window_size,
                                                pulse_durations=pulse_durations
                                                )
    
    QRC_fock.train()

    _, error = QRC_fock.test(test_data=testing_data,test_targets=testing_targets)

    print(f"Trial run for 3_level with error of {error}")

    QRC_fock.plot(save_dir=os.path.join(os.getcwd(),"test_plots\\fock_upgraded"),
                    save_fig=True)
    
    return

def test_fock():
    N = 10
    subspace_dim = 8
    initial_state = qt.fock(N)
    measurement_ops = []
    for i in range(subspace_dim):
        measurement_ops.append(qt.fock_dm(N,i))


    H_int = qt.create(N) + qt.destroy(N)

    c_ops = [qt.destroy(N)]

    QRC_fock= QRC.QRC_Classification(N=N,
                                                collapse_ops=c_ops,
                                                H_interaction=H_int,
                                                measurement_ops=measurement_ops,
                                                initial_state=initial_state,
                                                training_data=training_data,
                                                training_targets=training_targets,
                                                classification_dim=2,
                                                window_size=window_size,
                                                )
    
    QRC_fock.train()

    _, error = QRC_fock.test(test_data=testing_data,test_targets=testing_targets)

    print(f"Trial run for 3_level with error of {error}")

    QRC_fock.plot(save_dir=os.path.join(os.getcwd(),"test_plots\\fock"),
                    save_fig=True)


test_3level_upgraded("test_plots\\3level_upgraded_test_fock",kappa_low=0,kappa_high=3.9)
test_3level_upgraded("test_plots\\3level_upgraded_test_fock2",kappa_low=0)
test_3level_upgraded("test_plots\\3level_upgraded_test_non_hermitian")



