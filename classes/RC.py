import os
import sys


from matplotlib.pylab import permutation
import numpy as np
import qutip as qt
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import json

from enum import Enum
from scipy.special import eval_legendre
from itertools import combinations

from utilities.save import json_converter

class IPC_type(Enum):
    UNIFORM = 1
    NORMAL = 2

class reg_type(Enum):
    PINV = 1
    RIDGE = 2

def _regression(data,targets,inv_type:reg_type = reg_type.PINV,ridge:float = 1e-3):
    match inv_type:
        case reg_type.PINV:
            inverse_train = np.linalg.pinv(data)

        case reg_type.RIDGE:
            inverse_train = np.linalg.inv(data.T @ data 
                                        + ridge * np.eye(data.shape[1]) ) @ data.T @ targets

    return inverse_train

@dataclass(kw_only=True)
class RC(ABC):
    #Number of elements to exclude when proceessing data. 
    #Ie simulate data returns (data_size) elements and then conducts training/testing on (data_size-washout)    
    washout: int 
    window_size: int = field(default=20, init=True)
   
    #State variables
    is_trained: bool = field(default=False, init=False)

    # Data to be assigned later or given in subclasses
    dynamics_data: list = field(default_factory=list, init=False)
    predictions: np.ndarray | None = field(default=None, init=False)
    test_targets: np.ndarray | None = field(default=None, init=False)
    W: np.ndarray | None = field(default=None, init=False)
    
    def __post_init__(self):

        return

    def train(self,data:np.ndarray,
              targets: np.ndarray,
              reg_type:reg_type = reg_type.PINV,
              save_dynamics:bool = False) -> np.ndarray:
        """
        Overarching method that will train the model with the data given and update the weight matrix
        
        """


        #Exclude washout period in results
        training_results = self.simulate_data(
            data,
            is_train=True,
        )[:,self.washout:]

        #Different inverse calculations
        
        inverse_train = _regression(
            data=training_results,
            targets=targets,
            inv_type=reg_type
        )

        self.W = targets @ inverse_train
        
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

    #Method to plot predictions vs test targets and save to a directtory
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

    #Method to calculate nrmse between targets and self.predictions
    def _calc_nrmse(self,targets) -> float: 
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


    def evaluate_IPC(self, 
                    data_size: int = 5000, 
                    d_max: int = 4,
                    tau_max: int = 10,
                    threshold: float = 1e-3) -> tuple[float, np.ndarray]:

        def find_permutations(n, target, current_path=None):
            """Ordered compositions of target into n positive integers."""
            if current_path is None:
                current_path = []
            if len(current_path) == n - 1:
                if target >= 1:
                    yield current_path + [target]
                return
            for i in range(1, target - (n - 1 - len(current_path)) + 1):
                yield from find_permutations(n, target - i, current_path + [i])

        def eval_legendre_norm(degree, x):
            """Evaluates the orthonormal Legendre polynomial P_n(x) with unit variance."""
            norm_factor = np.sqrt(2 * degree + 1)
            return norm_factor * eval_legendre(degree, x)

        rng = np.random.default_rng()
        u = rng.uniform(-1.0, 1.0, size=data_size)

        #Simulate raw data into reservoir
        raw = self.simulate_data(u, is_train=False, save_dynamics=False)


        clean_data = np.asarray(raw)[:, self.washout:]  # (K, T_eff)
        X = clean_data.T                                # (T_eff, K)
        T_eff, K = X.shape

        # Adaptive noise floor threshold — scales with K/T_eff
        noise_floor = K / T_eff
        effective_threshold = max(threshold, 3.0 * noise_floor)

        max_delay = min(tau_max, self.washout)
        targets, degree_groups = [], []

        for D in range(1, d_max + 1):
            print(f"Calculating Capacity for degree {D}")
            for n_poly in range(1, D + 1):
                # Ordered degree assignments summing to D
                deg_permutations = list(find_permutations(n_poly, D))
                # DISTINCT time lags only — no same-lag products
                delays_list = list(combinations(range(1,max_delay + 1), n_poly))

                for perm in deg_permutations:
                    for delay_tuple in delays_list:
                        z = np.ones(T_eff)
                        for d, tau in zip(perm, delay_tuple):
                            u_lag = u[self.washout - tau : data_size - tau]
                            z *= eval_legendre_norm(d, u_lag)
                        targets.append(z)
                        degree_groups.append(D)

        Z = np.column_stack(targets)  # (T_eff, N)

        # One joint regression — correctly bounds total IPC ≤ K

        W_opt = _regression(X, Z)
        Z_hat = X @ W_opt  # (T_eff, N)

        capacity_array = np.zeros(d_max)
        for i, D in enumerate(degree_groups):
            var_z = np.var(targets[i])
            if var_z == 0:
                continue
            mse = np.mean((Z_hat[:, i] - targets[i]) ** 2)
            c_i = max(0.0, 1.0 - mse / var_z)
            if c_i >= effective_threshold:
                capacity_array[D - 1] += c_i

        return float(np.sum(capacity_array)), capacity_array
    


@dataclass(kw_only=True)
class RC_Classification(RC):

    classification_dim:int

    def __post_init__(self):
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
    
    pass


@dataclass(kw_only=True)
class RC_TimeSeries(RC):
    
    delay:int = 1



    def __post_init__(self):

        
        # self.training_targets = self.training_data[self.wash_out:]
        # self.training_target_length = len(self.training_targets)
        # self.training_data = self.training_data[:len(self.training_data)-self.window_size-self.delay]
        # CRITICAL: Pass the execution to the next class in the MRO chain!
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
    
