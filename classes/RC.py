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
from utilities.inversion import invert,RegType

class IPCType(Enum):
    UNIFORM = 1
    NORMAL = 2




@dataclass(kw_only=True)
class RC(ABC):
    #Number of elements to exclude when proceessing data. 
    #Ie simulate data returns (data_size) elements and then conducts training/testing on (data_size-washout)    
    washout: int
    window_size: int = field(default=20, init=True)
    #Steps between the newest input a feature column has seen and the target it is trained on
    delay: int = 0

    #State variables
    is_trained: bool = field(default=False, init=False)

    # Data to be assigned later or given in subclasses
    dynamics_data: list = field(default_factory=list, init=False)
    predictions: np.ndarray | None = field(default=None, init=False)
    test_targets: np.ndarray | None = field(default=None, init=False)
    W: np.ndarray | None = field(default=None, init=False)

    
    def __post_init__(self):

        return

    def _align(self, features: np.ndarray, targets: np.ndarray, data_length: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Pair feature column i (newest input data[i + output_offset]) with target targets[i + output_offset + delay],
        dropping the washout columns and the final columns whose target runs past the end of the data.

        targets must be time-aligned with the input data: same length (last axis), targets[t] is the desired
        output at time t. Do not pre-trim targets for the window, washout or delay; that is done here.
        """
        targets = np.asarray(targets)
        if targets.shape[-1] != data_length:
            raise ValueError(
                f"targets must be time-aligned with data (same length): got {targets.shape[-1]} targets "
                f"for {data_length} data points. Do not pre-trim targets for window_size, washout or delay."
            )
        if features.shape[1] != data_length - self.output_offset:
            raise RuntimeError(
                f"simulate_data returned {features.shape[1]} columns for {data_length} data points, "
                f"expected {data_length - self.output_offset} (output_offset = {self.output_offset})"
            )

        features = features[:, self.washout:features.shape[1] - self.delay]
        targets = targets[..., self.output_offset + self.delay + self.washout:]
        return features, targets

    def train(self,data:np.ndarray,
              targets: np.ndarray | None = None,
              reg_type:RegType = RegType.RIDGE,
              save_dynamics:bool = False) -> np.ndarray:
        """
        Overarching method that will train the model with the data given and update the weight matrix

        targets must be time-aligned with data (same length, targets[t] is the desired output at time t).
        If no targets are given, the data itself is used, so the model learns to predict data[t + delay].
        delay is applied to explicit targets as well.
        """

        #If no explicit targets are given, assume they are the input data shifted by the delay given
        if targets is None:
            targets = data

        training_results, targets = self._align(
            self.simulate_data(data, is_train=True), targets, len(data)
        )

        #Different inverse calculations
        inverse_train = invert(
            data=training_results,
            inv_type=reg_type
        )

        self.W = targets @ inverse_train

        self.is_trained = True

        return self.W

    def test(self, test_data, test_targets=None) -> tuple[np.ndarray, float]:
        """
        Method to test the reservoir with the weight matrix calculated from self.train()
        method on given testing_data.

        test_targets follow the same time-aligned convention as train(); if omitted, test_data is used.
        This function does not handle plotting.
        """

        if self.W is None or not self.is_trained:
            raise ValueError("Weight matrix has yet to be calculated please run self.train() first")

        if test_targets is None:
            test_targets = test_data

        testing_results, self.test_targets = self._align(
            self.simulate_data(test_data, is_train=False), test_targets, len(test_data)
        )

        self.predictions = np.asarray(self.W @ testing_results)

        error = self._calc_nrmse(self.test_targets)

        return self.predictions, error


    
    @property
    def output_offset(self) -> int:
        """Index of the last input seen by the first column returned from simulate_data."""
        return 0

    @abstractmethod
    def simulate_data(self,data,is_train:bool,save_dynamics:bool=False) -> np.ndarray:
        """
        Abstract method that will contain the core logic for simulating the data into the reservoir.
        
        This is to be implimented in subclasses of the RC class.

        """
        pass

    #Method to plot predictions vs test targets and save to a directtory
    def plot(self, error,save_dir=None, save_fig=False, filename="prediction_plot.png"):
        """
        Plotting method that will plot and save self.testing_targets and self.predictions.
        By default these values are set by the self.train() method and will be changed if self.train()
        is called inbetween calls of this function.

        This method also handles saving of plot and of parameters into JSON file.
        """
        # Guard clause in case predictions haven't been generated yet
        if self.predictions is None:
            raise ValueError("No predictions found to plot. Run test() first.")

      
        x_values = np.arange(0, len(self.predictions))
        
        plt.figure(figsize=(10, 4))
        
        plt.plot(x_values, self.predictions, "-o",markersize=2, color="orange", label="Predictions")

        
        if self.test_targets is not None:
            plt.plot(x_values, self.test_targets, "-o",color="black",markersize =2, lw=2, label="Targets")
        else:
            raise ValueError("test_targets is None and cannot be plotted")
        
        plt.legend()
        plt.title(f"Predictions vs Targets (NRMSE:{error:.4f})")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
       
        if save_fig:
            
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
        print(f"Simulating {data_size} data points into the reservoir...")
        raw = self.simulate_data(u, is_train=False)


        clean_data = np.asarray(raw)[:, self.washout:]  # (K, T_eff)
        X = clean_data.T                                # (T_eff, K)
        T_eff, K = X.shape

        # Adaptive noise floor threshold — scales with K/T_eff
        noise_floor = K / T_eff
        effective_threshold = max(threshold, 3 * noise_floor)

        # Input index of the last sample seen by the first post-washout output
        start = self.output_offset + self.washout
        if start < tau_max:
            raise ValueError(
                f"washout + output_offset ({start}) must be >= tau_max ({tau_max}) to build lagged targets"
            )
        max_delay = tau_max
        targets, degree_groups = [], []

        #Structure is [capacities], max_capacity,  total_num_capacities, num_successful_capacities
        capacity_dict = {D: [[], 0.0,0,0] for D in range(1, d_max + 1)}
        for D in range(1, d_max + 1):
            num_capacities = 0
            print(f"Calculating target functions for degree {D}")
            for n_poly in range(1, D + 1):
                # Ordered degree assignments summing to D
                deg_permutations = list(find_permutations(n_poly, D))
                # DISTINCT time lags only — no same-lag products
                delays_list = list(combinations(range(1,max_delay + 1), n_poly))

                for perm in deg_permutations:
                    for delay_tuple in delays_list:
                        z = np.ones(T_eff)
                        for d, tau in zip(perm, delay_tuple):
                            u_lag = u[start - tau : start - tau + T_eff]
                            z *= eval_legendre_norm(d, u_lag)
                        targets.append(z)
                        degree_groups.append(D)
                        num_capacities += 1
            capacity_dict[D][2] = num_capacities

        Z = np.column_stack(targets)  # (T_eff, N)

        # One joint regression — correctly bounds total IPC ≤ K

        W_opt = invert(X) @ Z  # (K, N)
        Z_hat = X @ W_opt      # (T_eff, N)

        
        for i, D in enumerate(degree_groups):
            var_z = np.var(targets[i])
            if var_z == 0:
                continue
            mse = np.mean((Z_hat[:, i] - targets[i]) ** 2)
            c_i = max(0.0, 1.0 - mse / var_z)
            if c_i >= effective_threshold:
                #Count the number of successful capacites for each degree
                capacity_dict[D][3] += 1
                capacity_dict[D][0].append(c_i)

        total_cap = 0
        for (key,value) in capacity_dict.items():
            print(f"Calculating capacities for degree {key}")
            value[1] = max(value[0],default=0.0)
            total_cap += np.sum(value[0])
        

        return float(total_cap), capacity_dict
    

#Helper functions
def plot_ipc_by_degree(capacity_dict, 
                       total_capacity:float=None, 
                       figsize=(10, 4.5), 
                       save_path=None):
    """
    Plots IPC breakdown grouped by degree D = sum(degrees), showing absolute capacity
    and fractional contribution per degree similar to Dambre et al. (2012).
    A second figure shows the average capacity per successful target function and the max
    capacity of any single target function for each degree, along with the fraction of
    target functions that passed the threshold.

    Parameters:
        capacity_dict (dict): Dictionary from RC.ipc mapping
                              D -> ([C_i, ...], max C_i, total_num_capacities, num_successful_capacities).
        total_capacity (float, optional): Sum of all capacities. Calculated if None.
        figsize (tuple): Dimensions of the figure.
        save_path (str, optional): File path to save the plot. The per-target figure
                                   is saved with a "_per_target" suffix.
    """

    # 1. Sum, average, and max of individual capacities within each degree D
    degrees = sorted(capacity_dict.keys())
    capacities_list, avg_list, max_list, num_total_list, num_success_list = [], [], [], [], []
    for D in degrees:
        caps, max_cap, num_total, num_success = capacity_dict[D]
        capacities_list.append(float(np.sum(caps)))
        # Average over the successful capacities (those above threshold)
        avg_list.append(float(np.sum(caps)) / num_success if num_success > 0 else 0.0)
        max_list.append(float(max_cap))
        num_total_list.append(num_total)
        num_success_list.append(num_success)
    capacities_list = np.array(capacities_list)
    avg_list = np.array(avg_list)
    max_list = np.array(max_list)
    num_total_list = np.array(num_total_list)
    num_success_list = np.array(num_success_list)
    kept_pct = np.divide(100.0 * num_success_list, num_total_list,
                         out=np.zeros(len(degrees)), where=num_total_list > 0)

    # 2. Fraction of total capacity per degree
    if total_capacity is None:
        total_capacity = float(np.sum(capacities_list))
    fractions = capacities_list / total_capacity if total_capacity > 0 else np.zeros_like(capacities_list)
    labels = [f"Deg {D}" for D in degrees]

    # 3. Create two-panel plot (Absolute & Fractional)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(degrees)))

    # Panel 1: Absolute Capacity per Degree (C_D)
    bars1 = ax1.bar(labels, capacities_list, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Absolute Capacity $C_D$", fontsize=11)
    ax1.set_title("Capacity per Degree $D$", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Add numeric labels on top of absolute capacity bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(capacities_list), 
                 f"{yval:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Fraction of Total Capacity (C_D / C_total)
    bars2 = ax2.bar(labels, fractions * 100, color=colors, edgecolor="black", alpha=0.85)
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

    # 4. Separate plot: average capacity per successful target vs. max single-target capacity
    def compact(n):
        """Shorten large counts, e.g. 177100 -> 177.1k."""
        return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.1f}k" if n >= 1e4 else f"{n}"

    fig2, ax3 = plt.subplots(figsize=(max(figsize[0], 1.6 * len(degrees)), figsize[1] + 0.5))
    x = np.arange(len(degrees))
    width = 0.38

    bars_avg = ax3.bar(x - width/2, avg_list, width, label="Average per successful target", color=colors, edgecolor="black", alpha=0.85)
    bars_max = ax3.bar(x + width/2, max_list, width, label="Max single target", color=colors, edgecolor="black", alpha=0.85, hatch="//")
    ax3.set_xticks(x)
    # Tick labels show successful / total target functions and percentage kept per degree
    ax3.set_xticklabels([f"Deg {D}\n{compact(s)} / {compact(n)}\n{p:.1f}% kept"
                         for D, s, n, p in zip(degrees, num_success_list, num_total_list, kept_pct)],
                        fontsize=9, linespacing=1.4)
    ax3.set_xlabel("Successful / total target functions", fontsize=10, labelpad=8)
    ax3.set_ylabel("Capacity $C_i$", fontsize=11)
    ax3.set_title("Per-Target Capacity per Degree $D$", fontsize=12, fontweight="bold", pad=30)
    # Scale y-axis to the data (capped at 1) so small capacities stay readable
    y_top = max(float(np.max(max_list, initial=0.0)), 1e-3)
    ax3.set_ylim(0, min(1.1, 1.3 * y_top))
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    # Add numeric labels on top of per-target bars
    for bar in list(bars_avg) + list(bars_max):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * y_top,
                 f"{yval:.3f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()

    if save_path:
        root, ext = os.path.splitext(save_path)
        fig2.savefig(f"{root}_per_target{ext}", dpi=300, bbox_inches="tight")




@dataclass(kw_only=True)
class RCClassification(RC):

    """
    This Class needs some work! 
    """

    classification_dim:int

    def __post_init__(self):
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
    
    pass


@dataclass(kw_only=True)
class RCTimeSeries(RC):

    """
    This class for all intensive purposes is just RC but with a modified option of closed loop testing.
    """
    
    delay:int = 1



    def __post_init__(self):


        # self.training_targets = self.training_data[self.wash_out:]
        # self.training_target_length = len(self.training_targets)
        # self.training_data = self.training_data[:len(self.training_data)-self.window_size-self.delay]
        # CRITICAL: Pass the execution to the next class in the MRO chain!
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    def test(self, test_data, test_targets=None, open_loop: bool = True) -> tuple[np.ndarray, float]:
        """
        open_loop=True: every prediction is made from true test inputs (see RC.test).
        open_loop=False: autonomous one-step prediction, each prediction is fed back in as the next input.
        Closed loop predicts the input series itself, so it requires delay == 1 and no explicit test_targets.
        """
        if open_loop:
            return super().test(test_data, test_targets)

        if self.W is None or not self.is_trained:
            raise ValueError("Weight matrix has yet to be calculated please run self.train() first")
        if test_targets is not None:
            raise ValueError("Closed-loop prediction feeds predictions back as inputs, so test_targets must be None")
        if self.delay != 1:
            raise ValueError(f"Closed-loop prediction requires delay == 1, got delay = {self.delay}")

        test_data = np.asarray(test_data)
        #Index of the newest input seen by the first post-washout feature column
        start = self.output_offset + self.washout
        self.test_targets = test_data[start + 1:]

        history = test_data[:start + 1]
        preds = []
        for _ in range(len(self.test_targets)):
            #Newest column's features predict the next value
            prediction = self.W @ self.simulate_data(history, is_train=False)[:, -1]
            preds.append(prediction)
            history = np.append(history, prediction)[1:]   # np.append doesn't mutate in place

        self.predictions = np.array(preds)
        error = self._calc_nrmse(self.test_targets)
        return self.predictions, error



@dataclass(kw_only=True)
class LinearRegTimeSeries(RCTimeSeries):
    """
    Baseline with no reservoir: the features are the raw input window itself,
    so train() fits ordinary linear regression on the last window_size inputs.

    Uses the same windowing as QRC, so the two can be compared on identical data.
    """

    #Append a constant row so the fit has an intercept
    use_bias: bool = True

    @property
    def output_offset(self) -> int:
        #Column i comes from window u[i : i+window_size], so its last input is u[i+window_size-1]
        return self.window_size - 1

    
    def simulate_data(self, data, is_train: bool, save_dynamics: bool = False) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        n_windows = len(data) - self.window_size + 1
        features = np.zeros((self.window_size, n_windows))

        #Fill each column with the raw window data to be used for linear regresion
        for i in range(n_windows):
            features[:, i] = data[i:i + self.window_size]

        #Add an optional bias to the features 
        if self.use_bias:
            features = np.vstack([features, np.ones(n_windows)])

        return features