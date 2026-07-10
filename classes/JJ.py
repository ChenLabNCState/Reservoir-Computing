import qutip as qt
import numpy as np
from RC import RC_TimeSeries
import scipy
from dataclasses import dataclass
import os
from dataclasses import dataclass, field, asdict

@dataclass(kw_only=True)
class JJ(RC_TimeSeries):
    """
    Class to implement the Josephson Junction Reservoir Computing model.
    """

    virtual_nodes:int = 20
    dt: float 
    time_steps: int = 50
    alpha:float 
    initial_phi: float = 5
    initial_v: float = 5
    k_inj:float = 0.25
    I_dc:float = 1.5

    random_seed:int = field(default=21,init=True)


    def simulate_data(self, data):

        
        if self.training_target_length is None:
            raise ValueError(
                f"Please make sure to input training data"
            )

        node_results = np.zeros((self.virtual_nodes,len(data)))
        phi = self.initial_phi
        V = self.initial_v

        rng = np.random.default_rng(seed=self.random_seed)
        mask = rng.uniform(-1,1,self.virtual_nodes)
        for i,value in enumerate(data):
            # mask = rng.uniform(-1,1,self.virtual_nodes)
            masked_data = self.I_dc+value*mask*self.k_inj

            sub_dt = self.dt/self.virtual_nodes
            for j,input_current in enumerate(masked_data):
                start = i*self.dt + j* sub_dt
                end = i*self.dt + (j+1)*sub_dt

                y0 = [phi,V]

                solution = scipy.integrate.solve_ivp(
                    JJ_diff_eq, 
                    (start,end), 
                    y0,
                    args=(input_current,self.alpha)
                    )
                phi = solution.y[0][-1]
                V = solution.y[1][-1]
                
                node_results[j,i] = V
        return node_results



def JJ_diff_eq(phi, V, I, alpha):
    phi, V = V  # Unpack the state variables
    dphi_dt = V
    dv_dt = I - np.sin(phi) - alpha * V
    return [dphi_dt, dv_dt]

def random_mask(size,seed):
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(-1,1,size)
