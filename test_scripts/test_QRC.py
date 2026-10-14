import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import json
from dataclasses import asdict

import qutip as qt
from classes.QRC import QRC
from classes.RC import RC, RCTimeSeries, plot_ipc_by_degree
from classes.QRC import QRCTimeSeries
from classes.RC import LinearRegTimeSeries
from utilities.generate import generate_mackey_glass
from utilities.save import json_converter
import numpy as np
import matplotlib.pyplot as plt

#Repo-level test_plots folder, independent of where the script is run from
PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_plots"))


def create_qubit(delay=1,
                 washout=0,
                 window_size = 20,
                 kappa=2.7,
                 pulse_duration =.1, 
                 pulse_time_steps = 100):
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

def create_fock(N=10,Subspace_dim = 8,
                delay=1,
                washout=0,
                window_size = 20,
                kappa=2.7,
                pulse_duration =.1, 
                pulse_time_steps = 100):
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

def create_linear(delay=1,
                  washout=0,
                  window_size = 20):
    return LinearRegTimeSeries(washout=washout, window_size=window_size, delay=delay)





def create_multiqubit(coupling_matrix, 
                      decay_list,
                      delay=1, 
                      washout=0, 
                      window_size=20, 
                      h=0.5, 
                      pulse_duration=.1, 
                      pulse_time_steps=100,
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

def train_and_test(reservoir:RCTimeSeries, data, name:str, save_plot:bool = False):
    """
    Train on the first 3/4 of data and test on the rest.
    Targets are left as None, so each model predicts data[t + delay] (time-aligned convention).
    Returns the trained reservoir (holding predictions and test_targets) and its NRMSE.
    """
    f = (3*len(data))//4
    training_data = data[:f]
    testing_data = data[f:]

    reservoir.train(data=training_data)

    _, error = reservoir.test(test_data=testing_data)

    print(f"Trial run for {name} with error of {error}")

    if save_plot:
        reservoir.plot(save_dir=os.path.join(PLOT_DIR, f"{name}_test_delay{reservoir.delay}"),
                       error=error,
                       save_fig=True)

    return reservoir, error

def test_qubit(data,delay=1,kappa:float = 2.7):
    return train_and_test(create_qubit(delay=delay,kappa=kappa), data, "qubit")

def test_fock(data,delay=1,kappa:float = 2.7):
    return train_and_test(create_fock(delay=delay,kappa=kappa), data, "fock")

def test_linear(data,delay=1):
    return train_and_test(create_linear(delay=delay), data, "linear")

def test_3qubit_chain(data,delay=1,kappa:float=2.7):
    adj_mat= np.asarray([[0,1,0],
                         [1,0,1],
                         [0,1,0]])
    decay_list = [kappa,kappa,kappa]
    return train_and_test(create_multiqubit(adj_mat,decay_list,delay),data,"3qubit_chain")

def test_3qubit_triangle(data,delay=1,kappa:float=2.7):
    adj_mat= np.asarray([[0,1,1],
                         [1,0,1],
                         [1,1,0]])
    decay_list = [kappa,kappa,kappa]
    return train_and_test(create_multiqubit(adj_mat,decay_list,delay),data,"3qubit_triangle")



def multi_reservoir_plot(reservoirs:list[RCTimeSeries], 
                         labels:list[str], 
                         save_dir=None, 
                         save_fig=False, 
                         filename="comparison_plot.png"):
        """
        Plotting multiple reservoirs' predictions against the same target data.

        Each reservoir must already have been tested on the same test data, and their aligned
        test_targets must match (same window_size, washout and delay), so they share one x-axis.
        """
        if len(labels) != len(reservoirs):
            raise ValueError(f"Got {len(labels)} labels for {len(reservoirs)} reservoirs")

        for reservoir, label in zip(reservoirs, labels):
            # Guard clause in case predictions haven't been generated yet
            if reservoir.predictions is None:
                raise ValueError(f"No predictions found for {label}. Run test() first.")

        targets = np.asarray(reservoirs[0].test_targets)
        for reservoir, label in zip(reservoirs[1:], labels[1:]):
            if not np.array_equal(np.asarray(reservoir.test_targets), targets):
                raise ValueError(f"{label} was tested on different targets than {labels[0]} "
                                 "(check test data, window_size, washout and delay)")

        x_values = np.arange(0, len(targets))

        plt.figure(figsize=(10, 4))
        plt.plot(x_values, targets, "-", color="black", lw=2, label="Targets")

        for reservoir, label in zip(reservoirs, labels):
            error = reservoir._calc_nrmse(targets)
            plt.plot(x_values, np.asarray(reservoir.predictions).ravel(), "-o", markersize=2, label=f"{label} (NRMSE:{error:.4f})")

        plt.legend()
        plt.title(f"Predictions vs Targets (delay = {reservoirs[0].delay})")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_fig:
            # If no directory is given, use the current working directory
            if save_dir is None:
                save_dir = os.getcwd()

            # Ensure the target directory actually exists to prevent OsErrors
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, "parameters.json"), "w") as f:
                json.dump({label: asdict(reservoir) for reservoir, label in zip(reservoirs, labels)},
                          f, indent=4, default=json_converter)

            full_path = os.path.join(save_dir, filename)
            plt.savefig(full_path)

        plt.close()


if __name__ == "__main__":
    data = generate_mackey_glass(1000, tau=17, dt=1)

    for delay in [1]:
        # qubit, _ = test_qubit(data, delay=delay)
        # fock, _ = test_fock(data, delay=delay)
        linear, _ = test_linear(data, delay=delay)
        chain,_ = test_3qubit_chain(data,delay=delay)
        triangle,_ = test_3qubit_triangle(data,delay=delay)

        multi_reservoir_plot([chain, triangle, linear],
                             labels=["Chain", "Triangle", "Linear regression"],
                             save_dir=os.path.join(PLOT_DIR, f"comparison_delay{delay}"),
                             save_fig=True)

