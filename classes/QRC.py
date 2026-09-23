import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
import qutip as qt
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from classes.RC import RC_TimeSeries,RC_Classification

@dataclass(kw_only=True)
class QRC:

    #Quantum specific things
    N: int   
    H_interaction: qt.Qobj
    collapse_ops: list[qt.Qobj]
    measurement_ops: list[qt.Qobj]
    initial_state: qt.Qobj

    #Optional parameters if a subspace is used
    H_base: qt.Qobj = field(default = None)
    subspace_dim: int = field(default=None)  # Default to None, set in __post_init__
    subspace_start_index:int = field(default=None)
    pulse_duration:float  = field(default=.1,init=True)
    pulse_time_steps: int = field(default=100,init=True)
    subspace_norm_threshold: float = 1e-9

    def __post_init__(self):

    
        # Handle the dynamic default for subspace_dim
        if self.subspace_dim is None:
            self.subspace_dim = self.N
        
        if self.subspace_start_index is None:
            self.subspace_start_index = 0

        if self.H_base is None:
            self.H_base = qt.qeye(self.N)

        # CRITICAL: Pass the execution to the next class in the chain!
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
    
    
    def _simulate_window(self,window_data):
        state = self.initial_state
        for pulse_amp in window_data:

            tlist = np.linspace(0, self.pulse_duration, self.pulse_time_steps) 
            """
            This might need to be changed later to H_int + H_base when we 
            have more than 1 qubit because the rotating frame for multiple qubits
            will still have precession of other qubits
            
            """
            result = qt.mesolve(self.H_interaction*pulse_amp, state, tlist, self.collapse_ops)
            state = result.states[-1]
        

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
        features= np.array([qt.expect(op, state=state) for op in self.measurement_ops])
            
        return features

    def _normalize_subspace(self, state: qt.Qobj) -> tuple[qt.Qobj, bool]:
        
        """
        Extracts, checks, and renormalizes a state. Returns the new normalized state

        This method is only meant to be called internally
        
        """
        if self.subspace_dim == self.N:
            return state, True
            
        if state.type == 'ket':
            state = qt.ket2dm(state)
            
        state_matrix = np.array(state.full(), dtype=complex)
        
        # Slice & calculate trace
        start = self.subspace_start_index
        end = start + self.subspace_dim
        subspace_trace = np.trace(state_matrix[start:end, start:end])
        
        if np.abs(subspace_trace) <= self.subspace_norm_threshold:
            return state, False  # Mark as invalid to set features to zero
            
        return qt.Qobj(state_matrix / subspace_trace, dims=state.dims), True

    def _normalize_subspace(self, state: qt.Qobj) -> tuple[qt.Qobj, bool]:
        
        """
        Extracts, checks, and renormalizes a state. Returns the new normalized state

        This method is only meant to be called internally
        
        """
        if self.subspace_dim == self.N:
            return state, True
            
        if state.type == 'ket':
            state = qt.ket2dm(state)
            
        state_matrix = np.array(state.full(), dtype=complex)
        
        # Slice & calculate trace
        start = self.subspace_start_index
        end = start + self.subspace_dim
        subspace_trace = np.trace(state_matrix[start:end, start:end])
        
        if np.abs(subspace_trace) <= self.subspace_norm_threshold:
            return state, False  # Mark as invalid to set features to zero
            
        return qt.Qobj(state_matrix / subspace_trace, dims=state.dims), True
        


@dataclass(kw_only=True)
class QRC_TimeSeries(RC_TimeSeries,QRC):

    def __post_init__(self):
  
        # CRITICAL: Pass the execution to the next class in the chain!
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    
    def simulate_data(self,time_series,is_train:bool):
        features = []

        for window_index in range(0,len(time_series)-(self.window_size)):
            window = time_series[window_index:window_index + self.window_size]
            window_features = self._simulate_window(window)
            features.append(window_features)

        results = np.array(features).T

        return  results

    def test(self,test_data,test_targets,open_loop:bool = True) -> tuple[np.ndarray,float]:
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

        if open_loop:
        
            testing_results = self.simulate_data(test_data,is_train=False,save_dynamics=False)[:,self.washout:]

            self.predictions = self.W @ testing_results

        if not open_loop:

            #Slice initial window
            window = test_data[0:self.window_size+self.washout]

            predictions = []

            for i in range(0,len(test_data)-self.window_size):

                #Simulate data for one window
                testing_result = self.simulate_data(window,is_train=False,save_dynamics=False)[:,self.washout:]

                #Add predicted target for one particular window to predictions
                prediction = self.W @ testing_result 
                predictions.append[prediction]

                #Append new results to the window and slice off the first index,
                np.append(window,prediction)
                window = window[1:]

            self.predictions = np.array(predictions)

        error = self._calc_nrmse(self.test_targets)

        return (self.predictions,error)
    

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
            window_probabilites = self._simulate_window(window,save_dynamics=save_dynamics)
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
        
        if self.pulse_durations.shape[1] != self.window_size or len(self.pulse_durations.shape) !=2:
            raise ValueError(
                f"pulse_durations must by a 2D numpy array with second index having dimension of window_size"
                f"Window size is {self.window_size} and your dimension is {self.pulse_duration.shape[1]}"
            )

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
