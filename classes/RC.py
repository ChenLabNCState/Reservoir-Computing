import numpy as np
import qutip as qt
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
import json
import random


@dataclass(kw_only=True)
class RC(ABC):
    #Mandatory Parameters
    training_data: np.ndarray
    washout: int 


    #Optional parameters with defaults
    window_size: int = field(default=10,init=True)
   

    # Data to be assigned later or given in subclasses
    training_targets: np.ndarray = field(default=None)
    dynamics_data:list = field(default_factory=list,init=False)
    is_trained:bool = field(default=False, init=False)
    rest_time_steps: int = field(default=None, init=False)
    predictions: np.ndarray = field(default=None, init=False)
    test_targets: np.ndarray = field(default=None, init=False)
    W: np.ndarray = field(default=None, init=False)
    
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


        training_results = self.simulate_data(self.training_data,save_dynamics=save_dynamics,is_train=True)

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
        
        testing_results = self.simulate_data(test_data,is_train=False,save_dynamics=False)

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
        plt.plot(x_values, self.test_targets, "-o",color="black",markersize =2, lw=2, label="Targets")
        
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

    training_target_length: int = field(default=None,init=False)

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

