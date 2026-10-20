import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import qutip as qt
from classes.QRC import QRC
from classes.RC import plot_ipc_by_degree
from classes.QRC import QRCTimeSeries
from utilities.generate import generate_mackey_glass
import numpy as np
import matplotlib.pyplot as plt

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

# #Run a test with 3 level system and compare to Fock state
# N_dim = 3
# pulse_duration = 5
# pulse_time_steps = 50
# window_size = 20
# d = 500
#3level system

# training_data,training_targets = QRC.generate_mixed_amplitude_sequence(noise_level=0.0)

# testing_data,testing_targets = QRC.generate_mixed_amplitude_sequence(noise_level=0)



# training_targets = training_targets[window_size-1:]
# testing_targets = testing_targets[window_size-1:]

# start_pulse_duration = 1
# end_pulse_duration= 11
# pulse_features = 5
# base_pulse_duration = 5

# # 1. Enclose the loop in brackets [] to make it a valid list comprehension first
# pulse_durations_list = [
#     np.linspace(base_pulse_duration, max_val, window_size) 
#     for max_val in np.linspace(start_pulse_duration, end_pulse_duration, pulse_features)
# ]

# # 2. Convert the clean list of lists into a 2D NumPy array
# pulse_durations = np.array(pulse_durations_list)

# def test_3level_base():
#     subspace_index_offset = 1
#     kappa_low = 0
#     kappa_high = .5
#     measurement_ops = custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)[1:]
#     sigma_x_subspace, sigma_y_subpace, sigma_z_subpace= custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)

#     destroy_ground = custom_dissipator(N_dim, subspace_index_offset)
#     destroy_excited = custom_dissipator(N_dim, subspace_index_offset + 1)

#     c_ops = [
#         np.sqrt(kappa_low) * destroy_ground,
#         np.sqrt(kappa_high) * destroy_excited
#     ]

#     initial_state = qt.fock(N_dim, 1)

#     H_int = sigma_x_subspace

#     QRC_3level= QRC.QRCClassification(N=N_dim,
#                                                 collapse_ops=c_ops,
#                                                 H_interaction=H_int,
#                                                 measurement_ops=measurement_ops,
#                                                 initial_state=initial_state,
#                                                 training_data=training_data,
#                                                 training_targets=training_targets,
#                                                 classification_dim=2,
#                                                 subspace_dim=2,
#                                                 subspace_start_index=subspace_index_offset,
#                                                 window_size=window_size
#                                                 )

#     QRC_3level.train()

#     _, error = QRC_3level.test(test_data=testing_data,test_targets=testing_targets)

#     print(f"Trial run for 3_level with error of {error}")

#     QRC_3level.plot(save_dir=os.path.join(os.getcwd(),"test_plots\\3level_base"),
#                     save_fig=True)
#     return


# def test_3level_upgraded(local_dir, kappa_low = 4,kappa_high = .1):


#     subspace_index_offset = 1

#     measurement_ops = custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)[1:]
#     sigma_x_subspace, sigma_y_subpace, sigma_z_subpace= custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)

#     destroy_ground = custom_dissipator(N_dim, subspace_index_offset)
#     destroy_excited = custom_dissipator(N_dim, subspace_index_offset + 1)

#     c_ops = [
#         np.sqrt(kappa_low) * destroy_ground,
#         np.sqrt(kappa_high) * destroy_excited
#     ]

#     initial_state = qt.fock(N_dim, 1)

#     H_int = sigma_x_subspace

#         # 1. Enclose the loop in brackets [] to make it a valid list comprehension first
#     pulse_durations_list = [
#         np.linspace(base_pulse_duration, max_val, window_size) 
#         for max_val in np.linspace(start_pulse_duration, end_pulse_duration, pulse_features)
#     ]

#     # 2. Convert the clean list of lists into a 2D NumPy array
#     pulse_durations = np.array(pulse_durations_list)
#     QRC_3level= QRC.QRCClassificationUpgraded(N=N_dim,
#                                                 collapse_ops=c_ops,
#                                                 H_interaction=H_int,
#                                                 measurement_ops=measurement_ops,
#                                                 initial_state=initial_state,
#                                                 training_data=training_data,
#                                                 training_targets=training_targets,
#                                                 classification_dim=2,
#                                                 subspace_dim=2,
#                                                 subspace_start_index=subspace_index_offset,
#                                                 window_size=window_size,
#                                                 pulse_durations=pulse_durations
#                                                 )

#     QRC_3level.train()

#     _, error = QRC_3level.test(test_data=testing_data,test_targets=testing_targets)

#     print(f"Trial run for 3_level with error of {error}")

#     QRC_3level.plot(save_dir=os.path.join(os.getcwd(),local_dir),
#                     save_fig=True)
#     return



def create_qubit(delay=1,washout=0,window_size = 20,kappa=2.7,pulse_duration =.1, pulse_time_steps = 1):
    N = 2
    initial_state = qt.basis(2,0)
    measurement_ops = [qt.sigmaz(),qt.sigmay()]
    kappa = kappa
    


    H_int = qt.create(N) + qt.destroy(N)

    c_ops = [np.sqrt(kappa)*qt.destroy(N)]


    QRC_qubit = QRCTimeSeries(
                            H_base= 0*qt.qeye(N),
                            N = N,    
                            collapse_ops=c_ops,
                            H_interaction=H_int,
                            measurement_ops=measurement_ops,
                            initial_state=initial_state,
                            washout= washout,
                            window_size = window_size,
                            delay = delay,
                            pulse_duration= pulse_duration,
                            pulse_time_steps=pulse_time_steps
    )

    return QRC_qubit

def create_fock(N=10,Subspace_dim = 8,delay=1,washout=0,window_size = 20,kappa=2.7,pulse_duration =.1, pulse_time_steps = 100):
    N = 10
    subspace_dim = 8
    initial_state = qt.fock(N)
    measurement_ops = []
    kappa = kappa
    for i in range(subspace_dim):
        measurement_ops.append(qt.fock_dm(N,i))


    H_int = qt.create(N) + qt.destroy(N)

    c_ops = [np.sqrt(kappa)*qt.destroy(N)]


    QRC_fock = QRCTimeSeries(
                                H_base= 0*qt.qeye(N),
                                N = N,    
                                collapse_ops=c_ops,
                                H_interaction=H_int,
                                measurement_ops=measurement_ops,
                                initial_state=initial_state,
                                washout= washout,
                                window_size = window_size,
                                delay = delay,
                                pulse_duration= pulse_duration,
                                pulse_time_steps=pulse_time_steps
        )

    return QRC_fock


def create_multiqubit(coupling_matrix, 
                      decay_list,
                      delay=1, 
                      washout=0, 
                      window_size=20, 
                      h=0.5, 
                      pulse_duration=.1, 
                      pulse_time_steps=1,
                      correlators=True, 
                      input_scaling=1):


    def op_at(op, i, n):
        """Embed a single-qubit op at site i of an n-qubit register."""
        ops = [qt.qeye(2)] * n
        ops[i] = op
        return qt.tensor(ops)

    
    J = np.asarray(coupling_matrix, dtype=float)
    kappa_list = np.asarray(decay_list)

    if len(kappa_list)!= J.shape[0]:
        raise ValueError(f"decay_list needs elements equal to number of qubits present")

    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"coupling_matrix must be square, got shape {J.shape}")
    if not np.allclose(J, J.T):
        raise ValueError("coupling_matrix must be symmetric (J[i][j] == J[j][i])")
    n = J.shape[0]

    X = [op_at(qt.sigmax(), i, n) for i in range(n)]
    Y = [op_at(qt.sigmay(), i, n) for i in range(n)]
    Z = [op_at(qt.sigmaz(), i, n) for i in range(n)]
    a = [op_at(qt.destroy(2), i, n) for i in range(n)]   # same lowering convention as create_qubit

    # Exchange coupling on the upper triangle: no self-coupling, each edge counted once
    edges = list(zip(*np.nonzero(np.triu(J, 1))))
    H_base = qt.qzero([2] * n)          # qutip 4: 0 * qt.tensor([qt.qeye(2)] * n)
    for i, j in edges:
        H_base += J[i, j] * (X[i] * X[j])
    H_base += h * sum(Z)

    # Input drive assumed to only drive the first qubit
    H_int = input_scaling * X[0]

    c_ops = [np.sqrt(kappa_list[i]) * a[i] for i in range(n)]

    # Features: local <X_i>,<Y_i>,<Z_i> plus optional <Z_i Z_j> on coupled pairs
    measurement_ops = X + Y + Z
    if correlators:
        measurement_ops += [Z[i] * Z[j] for i, j in edges]

    initial_state = qt.tensor([qt.basis(2, 0)] * n)

    return QRCTimeSeries(
        H_base=H_base,
        N=2**n,
        collapse_ops=c_ops,
        H_interaction=H_int,
        measurement_ops=measurement_ops,
        initial_state=initial_state,
        washout=washout,
        window_size=window_size,
        delay=delay,
        pulse_duration=pulse_duration,
        pulse_time_steps=pulse_time_steps,
    )



def test_qubit(data,delay=1,kappa:float = 2.7):
    
    QRC_qubit = create_qubit(delay=delay,kappa=kappa)

    f = (3*len(data))//4
    training_data = data[:f]
    testing_data = data[f:]

    QRC_qubit.train(data= training_data)

    _, error = QRC_qubit.test(test_data = testing_data)

    print(f"Trial run for qubit with error of {error}")

    QRC_qubit.plot(save_dir=os.path.join(os.path.abspath("/home/grant/Documents/Development/Reservoir-Computing/test_plots"),f"qubit_test_delay{delay}"),
                   error = error,
                    save_fig=True)

    return error
        
def test_fock(data,delay=1,kappa:float = 2.7):
    
    QRC_fock = create_fock(delay=delay,kappa=kappa)

    f = (3*len(data))//4
    training_data = data[:f]
    testing_data = data[f:]
    
    QRC_fock.train(data= training_data)

    _, error = QRC_fock.test(test_data = testing_data)

    print(f"Trial run for cavity with error of {error}")

    QRC_fock.plot(save_dir=os.path.join(os.path.abspath("/home/grant/Documents/Development/Reservoir-Computing/test_plots"),f"fock_test_delay{delay}"),
                   error = error,
                    save_fig=True)

    return error
        

array_size = 1000
data = generate_mackey_glass(array_size,tau=17,dt=1)


def test_dynamics(delay = 1,kappa = 2.7):

    f = (3*len(data))//4
    training_data = data[:f]


    qubit = create_qubit(delay=delay,kappa=kappa)
    fock = create_fock(delay=delay,kappa=kappa)

    qubit.plot_window_dynamics(window= training_data[0:qubit.window_size],file_name="qubit_dynamics.png")
    fock.plot_window_dynamics(window= training_data[0:fock.window_size],file_name="fock_dynamics.png")


def run_delay_sweep(delay_min,delay_max,step=5,kappa = 2.7):
    delays = []
    qubit_delay_error = []
    fock_delay_error = []
    for delay in np.arange(delay_min,delay_max+1,step=step):
        delays.append(delay)
        print(f"Calculating Predictions for delay: {delay}... ")
        qubit_delay_error.append(test_qubit(data=data,delay=delay,kappa=kappa))
        fock_delay_error.append(test_fock(data = data,delay = delay,kappa=kappa))


    plt.plot(delays,qubit_delay_error,label = 'qubit')
    plt.plot(delays,fock_delay_error,label = 'cavity')
    plt.title("Qubit Delay vs Error")
    plt.legend()
    plt.ylabel("NRMSE")
    plt.xlabel("Delay")
    plt.savefig(os.path.join(os.path.curdir,"delay_error.png"))



# run_delay_sweep(
#     delay_min=1,
#     delay_max=21,
#     step = 1,
# )

# test_dynamics(delay=1,kappa=.5)

qubit = create_qubit(washout=100)
fock = create_fock(washout = 100)
chain = create_multiqubit(
    coupling_matrix=[[0, 1, 0,0,0], [1, 0, 1,0,0], [0, 1, 0,1,0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]],
    decay_list=[2.7, 2.7, 2.7,2.7,2.7],
    washout=100,
)
star = create_multiqubit(
    coupling_matrix=[[0, 1, 1,1,1], [1, 0, 1,1,1], [1, 1, 0,1,1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]],
    decay_list=[2.7, 2.7, 2.7,2.7,2.7],
    washout=100,
)


total_C, cap_dict = qubit.evaluate_IPC(
        data_size=20000,
        d_max=5,
        tau_max=40,
        threshold=0,
)


chain_C, cap_dict_chain = chain.evaluate_IPC(
        data_size=20000,
        d_max=5,
        tau_max=40,
        threshold=0,
)



star_C, cap_dict_star = star.evaluate_IPC(
        data_size=20000,
        d_max=5,
        tau_max=40,
        threshold=0,
)
plot_ipc_by_degree(cap_dict, total_capacity=total_C, save_path=f"qubit_capacities.png")

plot_ipc_by_degree(cap_dict_star, total_capacity=star_C, save_path=f"star_capacities.png")

plot_ipc_by_degree(cap_dict_chain, total_capacity=chain_C, save_path=f"chain_capacities.png")


# fock_total_C, fock_cap_dict = fock.evaluate_IPC(
#         data_size=5000,
#         d_max=6,
#         tau_max=20,
#         threshold=0,
#     )

#plot_ipc_by_degree(fock_cap_dict, total_capacity=fock_total_C, save_path=f"fock_capacities.png")
# fock.evaluate_IPC()


