import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from classes.RC import RC_TimeSeries,RC_Classification,RC

from utilities.inversion import invert,reg_type

@dataclass(kw_only=True)
class QRC(RC):

    #Quantum specific things
    N: int   
    H_interaction: qt.Qobj
    collapse_ops: list[qt.Qobj]
    measurement_ops: list[qt.Qobj]
    initial_state: qt.Qobj
    

    #Optional parameters if a subspace is used
    H_base: qt.Qobj
    subspace_dim: int = field(default = 0,init=False)
    # subspace_indices:list[int] = field(default = [],init=False)  # Default to None, set in __post_init__
    pulse_duration:float  = field(default=.1,init=True)
    pulse_time_steps: int = field(default=50,init=True)
    subspace_norm_threshold: float = 1e-9



    # def __post_init__(self):

    
    #     if len(self.subspace_indices) == 0:
    #         self.subspace_indices = np.arange(0,self.N).tolist()

    #     self.subspace_dim = len(self.subspace_indices)

    #     # # CRITICAL: Pass the execution to the next class in the chain!
    #     # if hasattr(super(), '__post_init__'):
    #     #     super().__post_init__()
    
    
    def _simulate_window(self,window_data):
        state = self.initial_state
        dynamics = []
        for pulse_amp in window_data:

            tlist = np.linspace(0, self.pulse_duration, self.pulse_time_steps) 
            """
            This might need to be changed later to H_int + H_base when we 
            have more than 1 qubit because the rotating frame for multiple qubits
            will still have precession of other qubits
            
            """
            result = qt.mesolve(self.H_interaction*pulse_amp, state, tlist, self.collapse_ops)
            state = result.states[-1]
            dynamics.append(result)
        

        # Process state through the normalization function. Is valid will be False if the norm is smaller than the threshold
        #******CRITICAL****:  Currently this only works with reservoirs that are not tensor products of multiple qubits
        # if self.subspace_dim < self.N:
        #     state, is_valid = self._normalize_subspace(state)

        #     #Assuming norm is large enough calculate expectation values, otherwise just 
        #     if is_valid:
        #         features = np.array([qt.expect(op, state=state) for op in self.measurement_ops])
        #     else:
        #         features = np.zeros(len(self.measurement_ops))
        # else:

        #Calculate features (expected values of operators)
        features= np.array([qt.expect(op, state=state) for op in self.measurement_ops])
        
        return features

    def simulate_window_dynamics(self, window_data):
        """Run one window and return the full time trace of each measurement operator.

        Returns:
            times: (n_pulses * pulse_time_steps,) times, continuous across pulses
            expect: (n_ops, n_times) expectation values
            pulse_edges: start time of each pulse (for plotting boundaries)
        """
        state = self.initial_state
        tlist = np.linspace(0, self.pulse_duration, self.pulse_time_steps)
        times, traces, pulse_edges = [], [], []

        for k, pulse_amp in enumerate(window_data):
            result = qt.mesolve(self.H_interaction * pulse_amp, state, tlist, self.collapse_ops,
                                e_ops=self.measurement_ops,
                                options={"store_final_state": True})
            state = result.final_state

            # drop the first point after pulse 0 so the boundary points aren't duplicated
            s = 0 if k == 0 else 1
            t0 = k * self.pulse_duration
            times.append(tlist[s:] + t0)
            traces.append(np.array(result.expect)[:, s:])
            pulse_edges.append(t0)

        return np.concatenate(times), np.concatenate(traces, axis=1), np.array(pulse_edges)

    def plot_window_dynamics(self, window, labels=None,file_name:str = "test_dynamics.png"):
        times, expect, edges = self.simulate_window_dynamics(window)
        fig, (ax_in, ax_ex) = plt.subplots(2, 1, sharex=True, figsize=(10, 5),
                                        gridspec_kw={"height_ratios": [1, 3]})

        ax_in.step(np.append(edges, edges[-1] + self.pulse_duration),
                np.append(window, window[-1]), where="post", color="black")
        ax_in.set_ylabel("pulse amp")

        for i, trace in enumerate(expect):
            ax_ex.plot(times, trace, label=labels[i] if labels else f"op {i}")
        for e in edges:
            ax_ex.axvline(e, color="gray", lw=0.5, alpha=0.4)

        ax_ex.set_xlabel("time")
        ax_ex.set_ylabel("⟨O⟩")
        ax_ex.legend()
        plt.tight_layout()
        folder = os.path.join(os.path.curdir,"test_dynamics")
        os.makedirs(folder,exist_ok = True)
        plt.savefig(os.path.join(folder,file_name))

    # def _normalize_subspace(self, state: qt.Qobj) -> tuple[qt.Qobj, bool]:
        
    #     """
    #     Extracts, checks, and renormalizes a state. Returns the new normalized state

    #     This method is only meant to be called internally
        
    #     """
    #     if self.subspace_dim == self.N:
    #         return state, True
            
    #     if state.type == 'ket':
    #         state = qt.ket2dm(state)
            
    #     state_matrix = np.array(state.full(), dtype=complex)
        
    #     # Slice & calculate trace
    #     start = self.subspace_start_index
    #     end = start + self.subspace_dim
    #     subspace_trace = np.trace(state_matrix[start:end, start:end])
        
    #     if np.abs(subspace_trace) <= self.subspace_norm_threshold:
    #         return state, False  # Mark as invalid to set features to zero
            
    #     return qt.Qobj(state_matrix / subspace_trace, dims=state.dims), True

        


@dataclass(kw_only=True)
class QRC_TimeSeries(RC_TimeSeries,QRC):

    def __post_init__(self):
  
        # CRITICAL: Pass the execution to the next class in the chain!
        if hasattr(super(), '__post_init__'):
            super().__post_init__()


    #Returns results vector of dimension len(time_series) - window_size
    def simulate_data(self,time_series,is_train:bool,plot_dynamics:bool = True) -> np.ndarray:
        features = []

        for window_index in range(0,len(time_series)-(self.window_size)):
            window = time_series[window_index:window_index + self.window_size]
            window_features = self._simulate_window(window)
            features.append(window_features)

        results = np.array(features).T

        return  results

    def train(self,data:np.ndarray,
                  targets = None,
                  reg_type:reg_type = reg_type.PINV,
                  save_dynamics:bool = False) -> np.ndarray:
            """
            Overarching method that will train the model with the data given and update the weight matrix
            
            """

            #If no explicit targets are given, assume they are the shifted input data by the delay given 
            if targets is None:
                targets = data[self.delay + self.window_size:]
            else:
                targets = data
    
            #Exclude washout period in results
            training_results = self.simulate_data(
                data,
                is_train=True,
            )[:,self.washout:-self.delay]
    
            #Different inverse calculations
            
            inverse_train = invert(
                data=training_results,
                inv_type=reg_type
            )
    
            self.W = targets @ inverse_train
            
            self.is_trained = True
    
            return self.W

    def test(self, test_data, test_targets = None, open_loop: bool = True) -> tuple[np.ndarray, float]:
        
        if self.W is None or not self.is_trained:
            raise ValueError("Weight matrix has yet to be calculated please run self.train() first")

        #If no explicit test targets given just use the test data which we will 
        if test_targets is None:
            test_targets = test_data

            self.test_targets = test_targets[self.delay + self.window_size:]
        else:
            self.test_targets = test_targets


        if open_loop:
            testing_results = self.simulate_data(test_data, is_train=False)[:, self.washout:-self.delay]
            predictions = np.asarray(self.W @ testing_results)
        else:
            window = test_data[0:self.window_size + self.washout]
            preds = []
            for _ in range(len(test_data) - self.window_size):
                testing_result = self.simulate_data(window, is_train=False)[:, self.washout:-self.delay]
                prediction = self.W @ testing_result
                preds.append(prediction)
                window = np.append(window, prediction)[1:]   # np.append doesn't mutate in place
            predictions = np.array(preds)

        self.predictions = predictions
        error = self._calc_nrmse(self.test_targets)
        return predictions, error
    

@dataclass(kw_only=True)
class QRC_Classification(RC_Classification,QRC):
    def __post_init__(self):
  
        # CRITICAL: Pass the execution to the next class in the chain!
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    
    def simulate_data(self,time_series,is_train:bool,save_dynamics:bool):
        all_probabilities = []

        for window_index in range(len(time_series) - self.window_size + 1):
            window = time_series[window_index:window_index + self.window_size]
            window_probabilites = self._simulate_window(window)
            all_probabilities.append(window_probabilites)

        results = np.array(all_probabilities).T

        return  results


@dataclass(kw_only=True)
class QRC_Classification_upgraded(QRC_Classification):

    pulse_durations: np.ndarray

    def __post_init__(self):
  
        # CRITICAL: Pass the execution to the next class in the chain!
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        
        # if self.pulse_durations.shape[1] != self.window_size or len(self.pulse_durations.shape) !=2:
        #     raise ValueError(
        #         f"pulse_durations must by a 2D numpy array with second index having dimension of window_size"
        #         f"Window size is {self.window_size} and your dimension is {self.pulse_duration.shape[1]}"
        #     )

    def _simulate_window(self, window_data,save_dynamics:bool):
        features = []
        state = self.initial_state
        for j,pulse_duration_list in enumerate(self.pulse_durations):
            for i,pulse_amp in enumerate(window_data):

                tlist = np.linspace(0, pulse_duration_list[i], self.pulse_time_steps) 

                """
                This might need to be changed later to H_int + H_base when we 
                have more than 1 qubit because the rotating frame for multiple qubits
                will still have precession of other qubits
                
                """

                result = qt.mesolve(self.H_interaction*pulse_amp, state, tlist, self.collapse_ops)
                if save_dynamics:
                    self.dynamics_data.append([[qt.expect(op, state=state) for op in self.measurement_ops] for state in result.states])
                state = result.states[-1]
        

            # Process state through the normalization function. Is valid will be False if the norm is smaller than the threshold
            #******CRITICAL****:  Currently this only works with reservoirs that are not tensor products of multiple qubits
            # if self.subspace_dim < self.N:
            #     state, is_valid = self._normalize_subspace(state)

            #     #Assuming norm is large enough calculate expectation values, otherwise just 
            #     if is_valid:
            #         features.append(np.array([qt.expect(op, state=state) for op in self.measurement_ops]))
            #     else:
            #         features.append(np.zeros(len(self.measurement_ops)))
            # else:
            features.append(np.array([qt.expect(op, state=state) for op in self.measurement_ops]))
        
        features = np.asarray(features).ravel()

        return features
            
def normalize_subspace(state: qt.Qobj,subspace_dim:int,subspace_start_index:int,threshold = 1e-12) -> tuple[qt.Qobj, bool]:
        
        """
        Extracts, checks, and renormalizes a state. Returns the new normalized state

        This method is only meant to be called internally
        
        """
        # if subspace_dim == state.dims:
        #     return state, True
            
        if state.type == 'ket':
            state = qt.ket2dm(state)
            
        state_matrix = np.array(state.full(), dtype=complex)
        
        # Slice & calculate trace
        start = subspace_start_index
        end = start + subspace_dim
        subspace_trace = np.trace(state_matrix[start:end, start:end])
        
        if np.abs(subspace_trace) <= threshold:
            return state, False  # Mark as invalid to set features to zero
            
        return qt.Qobj(state_matrix / subspace_trace, dims=state.dims), True
