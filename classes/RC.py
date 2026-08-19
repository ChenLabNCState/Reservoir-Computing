import numpy as np
import qutip as qt
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
import json
import random
from enum import Enum
from scipy.special import eval_legendre
import itertools
class IPC_type(Enum):
    UNIFORM = 1
    NORMAL = 2

def _normalized_legendre(n):
    """
    Factory function returning a normalized Legendre polynomial of degree n.
    tilde_P_n(x) = sqrt(2n + 1) * P_n(x)
    """
    scale = np.sqrt(2 * n + 1)
    # The default argument 'n=n' binds the loop value immediately
    return lambda x, n=n, s=scale: s * eval_legendre(n, x)

@dataclass(kw_only=True)
class RC(ABC):
    #Mandatory Parameters
    training_data: np.ndarray
    washout: int 


    #Optional parameters with defaults
    window_size: int = field(default=10,init=True)
   

    # Data to be assigned later or given in subclasses
    training_targets: np.ndarray | None = field(default=None)
    dynamics_data: list = field(default_factory=list, init=False)
    is_trained: bool = field(default=False, init=False)
    rest_time_steps: int | None = field(default=None, init=False)
    predictions: np.ndarray | None = field(default=None, init=False)
    test_targets: np.ndarray | None = field(default=None, init=False)
    W: np.ndarray | None = field(default=None, init=False)
    
    def __post_init__(self):

        self.training_targets = self.training_data[self.washout:]


    def train(self,save_dynamics:bool = False) -> np.ndarray:
        """
        Overarching method that will train the model with the data given and update the weight matrix
        
        """

        if self.training_targets is None:
            raise ValueError(
                f"Training targets must be assigned and are currently {None}"
            )

        #Exclude washout period in results
        training_results = self.simulate_data(self.training_data,save_dynamics=save_dynamics,is_train=True)[:,self.washout:]

        inverse_train = np.linalg.pinv(training_results)

        self.W = self.training_targets @ inverse_train
        
        self.is_trained = True

        return self.W

    def test(self,test_data,test_targets) -> tuple[np.ndarray,float]:
        """
        Method to test the reservoir with the weight matrix calculated from self.train()
        method on given testing_data.

        This function does not handle plotting.
          
        """

        if self.W is None or self.is_trained is False:
            raise ValueError(
                f"Weight matrix has yet to be calculated, First run train method to find weight matrix"
            )
        
        self.test_targets = test_targets[self.washout:]
        
        testing_results = self.simulate_data(test_data,is_train=False,save_dynamics=False)[:,self.washout:]

        self.predictions = self.W @ testing_results

        error = self._calc_nrmse(self.test_targets)

        return (self.predictions,error)
    
    @abstractmethod
    def simulate_data(self,data,is_train:bool,save_dynamics:bool=False) -> np.ndarray:
        """
        Abstract method that will contain the core logic for simulating the data into the reservoir.
        
        This is to be implimented in subclasses of the RC class.

        """
        pass

    def plot(self, save_dir=None, save_fig=False, filename="prediction_plot.png"):
        """
        Plotting method that will plot and save self.testing_targets and self.predictions.
        By default these values are set by the self.train() method and will be changed if self.train()
        is called inbetween calls of this function.

        This method also handles saving of plot and of parameters into JSON file.
        """
        # Guard clause in case predictions haven't been generated yet
        if self.predictions is None:
            raise ValueError("No predictions found to plot. Run test() first.")

        # FIX 1: Changed np.range to np.arange
        x_values = np.arange(0, len(self.predictions))
        
        plt.figure(figsize=(10, 4))
        
        plt.plot(x_values, self.predictions, "-o",markersize=2, color="orange", label="Predictions")
        # FIX 3: Updated self.test_targets to self.testing_targets to match your dataclass
        
        if self.test_targets is not None:
            plt.plot(x_values, self.test_targets, "-o",color="black",markersize =2, lw=2, label="Targets")
        else:
            raise ValueError("test_targets is None and cannot be plotted")
        
        plt.legend()
        plt.title("Predictions vs Targets")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # FIX 2: Check 'save_fig' instead of the function 'plt.savefig'
        if save_fig:
            # THE CORE FIX: If no directory is given, use the current working directory
            if save_dir is None:
                save_dir = os.getcwd()
                
            # Ensure the target directory actually exists to prevent OsErrors
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, "parameters.json"), "w") as f:
                json.dump(asdict(self), f, indent=4, default=json_converter)
            
            full_path = os.path.join(save_dir, filename)
            plt.savefig(full_path)

        plt.close()


    def _calc_nrmse(self,targets):
        """
        Calculates the Normalized Root Mean Squared Error.
        Forces inputs to 1D to prevent broadcasting errors.
        """
        # 1. Force to 1D arrays and ensure float type
        preds = np.asarray(self.predictions).ravel()
        targs = np.asarray(targets).ravel()
        
        # 2. Handle NaNs if they exist
        mask = ~np.isnan(preds) & ~np.isnan(targs)
        preds = preds[mask]
        targs = targs[mask]

        if targs.size == 0:
            return np.nan 
        
        # 3. Calculate RMSE
        # Formula: sqrt(mean((y_hat - y)^2))
        rmse = np.sqrt(np.mean((preds - targs)**2))
        
        # 4. Normalize by the standard deviation of the targets
        target_std = np.std(targs)

        # Prevent division by zero if targets are constant
        if target_std == 0 or np.isnan(target_std):
            return np.nan 
    
        return rmse / target_std

    def evaluate_IPC_me(self,tests_funcs,data_size,type = IPC_type.UNIFORM,d_max:int = 5,max_delay:int = 20):
        """
        This method computes the information proccessing capacity of the reservoir. This number can slightly over different runs
        as it is based on the reservoirs response to inputs sampled from various distributions (default to UNIFORM).
                
        
             
                
        """


        #Create uniform input sequence of length T claled u(t)

        
        rng = np.random.default_rng()

        samples = rng.uniform(low=-1, high=1, size=data_size)

        #Helper functions to generate all possible legendre polynomial and delay
        #combinations as lists of tuples for a given order.

        def integer_compositions(n, k):
            """Generates all compositions of n into k positive integers (i >= 1)."""
            if k == 1:
                yield (n,)
                return
            for cuts in itertools.combinations(range(1, n), k - 1):
                yield tuple(b - a for a, b in zip((0,) + cuts, cuts + (n,)))

        def get_tuple_combinations(degree, maximum_delay, min_j=1):
            results = []
            available_j = range(min_j, maximum_delay)  # j < delay_max
            
            # Combination length k can range from 1 up to min(degree, len(available_j))
            for k in range(1, min(degree, len(available_j)) + 1):
                for j_combo in itertools.combinations(available_j, k):
                    for i_comp in integer_compositions(degree, k):
                        results.append(list(zip(i_comp, j_combo)))
                        
            return results

        tuple_dict = {}
        for d in range(d_max):

            tuple_dict[d] = get_tuple_combinations(d,maximum_delay=max_delay)
            





        

        #Construct all possible permutations of products of legendre/hermite polynomials up to degree of d_max
        #   - the polynomials are functions of the uniform input u(t-tau)
        #   - EX: P1(u(t-tau1))P2(u(t-tau3)) is one of many degree 3 polynomials.
        #   - We can represent these functions by a tuple representing the legendre polymials and delays ie L:(1,2), tau:(1,3)
        #   - Example of unallowed state: P1(u(t-tau1))P1(u(t-tau1)) as this is not orthogonal to P2(u(t-tau1)) 


        #For all these possible polynomials find all possible allowed combinations of delays. 

        #Evaluate all these functions 


        if type == IPC_type.NORMAL:
            pass
        if type == IPC_type.UNIFORM:


            return
    def evaluate_IPC(self, data_size: int = 1000, d_max: int = 3, tau_max: int = 20, threshold: float = 1e-3, type: IPC_type = IPC_type.UNIFORM) -> tuple[float, dict]:
        """
        Calculates the Information Processing Capacity (IPC) of the reservoir.
        Assumes simulate_data() already strips the washout period internally.
        """
        if type != IPC_type.UNIFORM:
            raise NotImplementedError("Currently, IPC is only implemented for UNIFORM distribution.")

        # 1. Generate uniform random input sequence u(t) ~ U[-1, 1] of full length data_size
        rng = np.random.default_rng()
        u = rng.uniform(low=-1.0, high=1.0, size=data_size)

        # 2. Drive reservoir -> states ALREADY have washout stripped by simulate_data
        raw_states = self.simulate_data(u, is_train=False, save_dynamics=False)
        states = np.asarray(raw_states)

        # Ensure states is strictly 2D with shape (T_eff, K)
        if states.ndim == 1:
            states = states[:, np.newaxis]  # Convert (T_eff,) -> (T_eff, 1)
        elif states.ndim == 2:
            # Handle (K, T_eff) vs (T_eff, K) layout
            if states.shape[0] < states.shape[1] and states.shape[1] == (data_size - self.washout):
                states = states.T
            elif states.shape[0] != (data_size - self.washout) and states.shape[1] == (data_size - self.washout):
                states = states.T

        # X is ready directly — do NOT slice self.washout again!
        X = states  
        T_eff = X.shape[0]

        if T_eff <= 0:
            raise ValueError(f"T_eff is 0. Verify data_size ({data_size}) is greater than washout ({self.washout}).")

        # 3. Generate target specs
        target_specs = self._generate_ipc_specs(d_max=d_max, tau_max=tau_max)

        # 4. Construct Target Matrix Z (shape: T_eff x N)
        Z_list = []
        valid_specs = []

        for degrees, delays in target_specs:
            if max(delays) > self.washout:
                continue

            z_i = np.ones(T_eff, dtype=float)
            for deg, tau in zip(degrees, delays):
                # u has length data_size. 
                # Post-washout output spans u[washout : data_size].
                # Delayed input u(t - tau) spans u[washout - tau : data_size - tau].
                # Both produce exact length: (data_size - tau) - (washout - tau) = T_eff
                u_delayed = u[self.washout - tau : data_size - tau]
                z_i = z_i * _normalized_legendre(deg)(u_delayed)

            Z_list.append(z_i)
            valid_specs.append((degrees, delays))

        if not Z_list:
            raise ValueError(f"No valid targets generated. Ensure washout ({self.washout}) >= tau_max ({tau_max}).")

        Z = np.column_stack(Z_list)  # Shape: (T_eff, N)

        # 5. Linear Regression W* = argmin ||X W - Z||^2 (Shape of W_opt: K x N)
        # Instead of np.linalg.lstsq(X, Z):
        K = X.shape[1]
        gamma = 1e-5  # Ridge parameter
        W_opt = np.linalg.inv(X.T @ X + gamma * np.eye(K)) @ (X.T @ Z)

        # 6. Reconstruct targets Z_hat = X @ W_opt (Shape: T_eff x N)
        Z_hat = X @ W_opt

        # 7. Compute individual target capacities C_i
        capacities_dict = {}
        total_capacity = 0.0

        for i, spec in enumerate(valid_specs):
            z_true = Z[:, i]
            z_pred = Z_hat[:, i]

            var_z = np.var(z_true)
            if var_z == 0:
                continue

            mse = np.mean((z_pred - z_true) ** 2)
            c_i = float(1.0 - (mse / var_z))

            c_i = max(0.0, c_i)
            if c_i >= threshold:
                capacities_dict[spec] = c_i
                total_capacity += c_i

        return total_capacity, capacities_dict

    def _generate_ipc_specs(self, d_max: int, tau_max: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """
        Helper method to generate all valid orthogonal degree and delay tuple pairs.
        Prevents duplicate terms according to IPC mathematical rules.
        """
        from itertools import combinations_with_replacement, product

        specs = []
        
        # Helper to partition degree D into degree tuples
        def get_degree_tuples(total_degree):
            if total_degree == 1:
                yield (1,)
                return
            for k in range(1, total_degree + 1):
                for comp in combinations_with_replacement(range(1, total_degree + 1), k):
                    if sum(comp) == total_degree:
                        yield comp

        for D in range(1, d_max + 1):
            for deg_tuple in set(get_degree_tuples(D)):
                # Generate valid delay tuples for this degree tuple
                L = len(deg_tuple)
                for delay_tuple in product(range(tau_max + 1), repeat=L):
                    # Rule: If degrees are identical, delays MUST be strictly increasing (tau_1 < tau_2 < ...)
                    # to prevent duplicate basis targets.
                    is_valid = True
                    for i in range(L - 1):
                        if deg_tuple[i] == deg_tuple[i+1] and delay_tuple[i] >= delay_tuple[i+1]:
                            is_valid = False
                            break
                    if is_valid:
                        specs.append((deg_tuple, delay_tuple))

        return specs
    
def json_converter(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return f"Big Array with shape = {o.shape}"
    if isinstance(o, qt.Qobj):
        # Safely bypass the object by saving its string description metadata
        return f"<QuTiP Qobj: dims={o.dims}, type={o.type}>"
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")



@dataclass(kw_only=True)
class RC_Classification(RC):

    classification_dim:int

    def __post_init__(self):
        # 1. Enforce that it is required for THIS subclass
        if self.training_targets is None:
            raise ValueError(
                "For QRC_TimeSeries, 'training_targets' is a required parameter "
                "and cannot be left as None."
                
            )
        
        # if len(self.training_data) - len(self.training_targets) == self.window_size-1:
        #     self.training_targets = self.clean_target(self.training_targets)

        # 2. Pass control up to the rest of your MRO chain
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    def clean_target(self,targets):
        return targets[self.window_size:]
    
    pass



@dataclass(kw_only=True)
class RC_TimeSeries(RC):
    
    delay:int = 1

    training_target_length: int | None = field(default=None,init=False)

    def __post_init__(self):

        
        # self.training_targets = self.training_data[self.wash_out:]
        # self.training_target_length = len(self.training_targets)
        # self.training_data = self.training_data[:len(self.training_data)-self.window_size-self.delay]
        # CRITICAL: Pass the execution to the next class in the MRO chain!
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
    

def generate_mixed_amplitude_sequence(
    total_points=150,
    segment_length_min=10,
    segment_length_max=10,
    sine_amplitude=1.0,
    square_amplitude=1.0,
    frequency=0.2,
    noise_level=0.0
):
    sequence = []
    labels = []
    current_points = 0

    while current_points < total_points:
        length = random.randint(segment_length_min, segment_length_max)
        if current_points + length > total_points:
            length = total_points - current_points

        pulse_type = random.choice(['sine', 'square'])
        t = np.arange(length)

        if pulse_type == 'sine':
            pulse = sine_amplitude * np.sin(2 * np.pi * frequency * t)
            label = 0
        else:
            pulse = square_amplitude * np.ones(length)
            label = 1

        if noise_level > 0:
            pulse += np.random.normal(0, noise_level, length)

        sequence.extend(pulse)
        labels.extend([label] * length)
        current_points += length

    return np.array(sequence), np.array(labels)


def generate_mackey_glass(length, dt=0.1, tau=17, beta=0.2, gamma=0.1, n=10):
    delay_steps = int(tau / dt)
    history = np.ones(delay_steps) * 0.9
    x = history[-1]
    series = []
    for _ in range(length):
        x_tau = history[0]
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x
        x += dx * dt
        series.append(x)
        history = np.roll(history, -1)
        history[-1] = x
    series = np.array(series)
    return series / (series.max())

