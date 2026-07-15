import qutip as qt
import numpy as np
from classes.RC import RC_TimeSeries
import scipy
from dataclasses import dataclass
import os
from dataclasses import dataclass, field, asdict
import  matplotlib.pyplot as plt
import math

@dataclass(kw_only=True)
class JJ(RC_TimeSeries):
    """
    Class to implement the Josephson Junction Reservoir Computing model.
    """

    virtual_nodes:int = 20
    theta: float 
    dt:float = field(init=False)
    time_steps: int = 50
    alpha:float 
    initial_phi: float = 0.1
    initial_v: float = 1
    k_inj:float = 0.25
    I_dc:float = 1.5
    plotting_substeps = 10

    random_seed:int = field(default=21,init=True)

    def __post_init__(self):
        self.dt = self.virtual_nodes*self.theta
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        
    def simulate_data(self, data,is_train:bool,save_dynamics:bool):


        node_results = np.zeros((self.virtual_nodes,len(data)-self.washout))
        phi = self.initial_phi
        V = self.initial_v

        rng = np.random.default_rng(seed=self.random_seed)
        mask = rng.uniform(-1,1,self.virtual_nodes)
        for i,value in enumerate(data):


            # mask = rng.unifrm(-1,1,self.virtual_nodes)
            masked_data = self.I_dc+value*mask*self.k_inj

            sub_dt = self.dt/self.virtual_nodes
            for j, input_current in enumerate(masked_data):
                start = i * self.dt + j * sub_dt
                end = i * self.dt + (j + 1) * sub_dt

                y0 = [phi, V]

                # 1. Enable dense_output to capture the path between start and end
                solution = scipy.integrate.solve_ivp(
                    JJ_diff_eq, 
                    (start, end), 
                    y0,
                    args=(input_current, self.alpha),
                    dense_output=True
                )
                
               
                # Update the boundary values for the next node step
                phi = solution.y[0][-1]
                V = solution.y[1][-1]

                if i>= self.washout:
                    if is_train and save_dynamics:
                        # 2. Sample 5-10 intermediate points along this sub-step
                        t_steps = np.linspace(start, end, self.plotting_substeps)
                        y_steps = solution.sol(t_steps)  # Extracts the smooth trajectory
                    
                        # 3. Append all intermediate points to your dynamics data
                        for k in range(len(t_steps)):
                            self.dynamics_data.append([y_steps[0, k], y_steps[1, k],input_current])

                    node_results[j, i-self.washout] = V
        return node_results

    #Plot entire dynamics for the testing
    def plot_dynamics(self,start,is_OP2:bool = False):

        start_idx = start*self.virtual_nodes

        dynamics_array = np.array(self.dynamics_data)

        input_offset = self.I_dc

        V = dynamics_array[start_idx:(start+self.plotting_substeps)*self.virtual_nodes,1]
        current = dynamics_array[start_idx:(start+self.plotting_substeps)*self.virtual_nodes,2]
        t_values = np.linspace(start_idx,(start+self.plotting_substeps)*self.virtual_nodes,self.virtual_nodes*self.plotting_substeps)

        if is_OP2 == False:
           current = current-input_offset
           

        plt.plot(t_values,V,'b-',linewidth = 1.5, label = "V")
        plt.plot(t_values,current,'g-',linewidth = 1.5,label = "I_dc mask")
        plt.xlabel("Time Step")
        plt.title("Current Vs Voltage Dynamics")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

    def plot_dynamics_2(self,start,cycles,is_OP2:bool = False):
        if not self.dynamics_data:
            print("No dynamics data to plot.")
            return
            
        dynamics_array = np.array(self.dynamics_data)

        phi_vals = dynamics_array[start:start+cycles*self.virtual_nodes, 0]
        V = dynamics_array[start:start+cycles*self.virtual_nodes, 1]

        y_label = "Phase ($\phi$)"
        if is_OP2:
            phi_vals = np.sin(phi_vals)
            y_label = r"$\sin\phi_{\mathrm{output}}$"


        
        # dynamics_array[:, 1] is the entire Voltage column (X-axis)
        # dynamics_array[:, 0] is the entire Phase column (Y-axis)
        plt.figure(figsize=(8, 6))
        plt.plot(V, phi_vals, 'b-', linewidth=1.5, label="Trajectory")
        
        plt.xlabel("Voltage (V)")
        plt.ylabel(y_label)
        plt.title("Josephson Junction Phase Space Dynamics")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

    #Plot one cycle phase dynamics
    def plot_dynamics_3(self,start,cycles,is_OP2:bool = False):
        if not self.dynamics_data:
            print("No dynamics data to plot.")
            return
            
        dynamics_array = np.array(self.dynamics_data)
        phi_vals = dynamics_array[start:start+cycles*self.virtual_nodes, 0]
        v_vals = dynamics_array[start:start+cycles*self.virtual_nodes, 1]
        
        y_label = "Phase ($\phi$)"
        if is_OP2:
            phi_vals = np.sin(phi_vals)
            y_label = r"$\sin\phi_{\mathrm{output}}$"

        
        # Set up a clean, professional publication-style figure
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
        
        # 2. Plot the main solid curve (blue/magenta profile)
        ax.plot(v_vals, phi_vals, color='blue', linewidth=2, label="Trajectory")
        
        # 3. Emulate the markers along the limit cycle
        # We sample a few evenly-spaced indices to scatter markers
        sample_indices = np.linspace(0, len(v_vals) - 1, 12, dtype=int)
        
        # Alternating green triangles and red circles like the image
        for idx, i in enumerate(sample_indices):
            if idx % 2 == 0:
                ax.plot(v_vals[i], phi_vals[i], 'g^', markersize=9)  # Green triangle
            else:
                ax.plot(v_vals[i], phi_vals[i], 'ro', markersize=8)  # Red circle
                
        # 4. Clean up styling to match the paper's minimalist look
        ax.set_xlabel(r"$V_{\mathrm{output}}$", fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
        ax.set_title(r"$V_{\mathrm{output}}$", fontsize=12, pad=10)
        
        # # Axis Limits and Custom Ticks
        # ax.set_yticks([-1, 0, 1])
        
        # # Enable minor ticks for that precise instrumented look
        # ax.minorticks_on()
        # ax.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6)
        # ax.tick_params(axis='both', which='minor', direction='out', length=3)
        
        # Add labels inside the box corners
        
        plt.tight_layout()
        plt.show()

        
    def plot_combined_dynamics(self, start_step, phase_cycles,phase_cycles2):
        if not self.dynamics_data:
            print("No dynamics data to plot.")
            return

        dynamics_array = np.array(self.dynamics_data)
        
        # Unpack columns: 0=phi, 1=V, 2=input_current
        phi_all = dynamics_array[:, 0]
        v_all = dynamics_array[:, 1]
        current_all = dynamics_array[:, 2]

        # 1. Adjusted figsize: Made it slightly wider (20) and shorter (4.5) 
        # to give the 3 subplots more room to breathe side-by-side.
        fig, axs = plt.subplots(1, 3, figsize=(20, 4.5), dpi=150)

        # ==========================================
        # SUBPLOT 1: Current vs Voltage Dynamics
        # ==========================================
        start_idx = start_step * self.virtual_nodes
        end_idx = (start_step + 10) * self.virtual_nodes
        
        v_slice = v_all[start_idx:end_idx]
        current_slice = current_all[start_idx:end_idx]
        t_values = np.linspace(start_idx, end_idx, len(v_slice))

        axs[0].plot(t_values, v_slice, 'b-', linewidth=1.5, label="V")
        axs[0].plot(t_values, current_slice, 'g-', linewidth=1.5, label=r"$I_{\mathrm{dc}}$ mask")
        axs[0].set_xlabel("Step Index", fontsize=11)
        axs[0].set_ylabel("Magnitude", fontsize=11)
        
        # Reduced fontsize slightly to 11 to prevent wide titles from overlapping
        axs[0].set_title("Current vs Voltage Dynamics", fontsize=11, fontweight='bold', pad=10)
        axs[0].grid(True, linestyle='--', alpha=0.5)
        axs[0].legend(loc='upper right', fontsize=9)
        axs[0].text(0.03, 0.92, "(a)", transform=axs[0].transAxes, fontsize=13, fontweight='bold')

        # ==========================================
        # SUBPLOT 2: Phase Space (Multi-cycle Wrapped Line)
        # ==========================================
        p2_start = start
        p2_end = start + (phase_cycles * self.virtual_nodes)

        phi_p2 = phi_all[p2_start:p2_end]
        v_p2 = v_all[p2_start:p2_end]
        
        # Wrapped phase
        phi_wrapped = phi_p2 % (2 * np.pi)
        
        # Prevent horizontal line jump artifacts using NaN masking
        diffs = np.abs(np.diff(phi_wrapped))
        jump_indices = np.where(diffs > np.pi)[0]
        v_plot2 = np.insert(v_p2, jump_indices + 1, np.nan)
        phi_plot2 = np.insert(phi_wrapped, jump_indices + 1, np.nan)

        axs[1].plot(v_plot2, phi_plot2, 'b-', linewidth=1.5, label="Trajectory")
        axs[1].set_xlabel("Voltage (V)", fontsize=11)
        axs[1].set_ylabel(r"Phase ($\phi$ mod $2\pi$)", fontsize=11)
        axs[1].set_title("Josephson Phase Space Dynamics", fontsize=11, fontweight='bold', pad=10)
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[1].legend(loc='upper right', fontsize=9)
        axs[1].text(0.03, 0.92, "(b)", transform=axs[1].transAxes, fontsize=13, fontweight='bold')

        # ==========================================
        # SUBPLOT 3: Single Cycle Orbit Style
        # ==========================================
        p3_start = start
        p3_end = start + self.virtual_nodes*phase_cycles2
        
        phi_p3 = phi_all[p3_start:p3_end]
        v_p3 = v_all[p3_start:p3_end]
        sin_phi = np.sin(phi_p3)

        axs[2].plot(v_p3, sin_phi, color='blue', linewidth=2, label="Trajectory")
        
        # Alternating green triangles and red circles along the track
        sample_indices = np.linspace(0, len(v_p3) - 1, 12, dtype=int)
        for idx, i in enumerate(sample_indices):
            if idx % 2 == 0:
                axs[2].plot(v_p3[i], sin_phi[i], 'g^', markersize=8)
            else:
                axs[2].plot(v_p3[i], sin_phi[i], 'ro', markersize=7)

        axs[2].set_xlabel(r"$V_{\mathrm{output}}$", fontsize=11)
        axs[2].set_ylabel(r"$\sin\phi_{\mathrm{output}}$", fontsize=11)
        axs[2].set_title("Josephson Junction Orbit", fontsize=11, fontweight='bold', pad=10)
        axs[2].set_ylim(-1.5, 1.5)
        axs[2].set_yticks([-1, 0, 1])
        
        axs[2].minorticks_on()
        axs[2].tick_params(axis='both', which='major', direction='out', length=5)
        axs[2].text(0.03, 0.92, "(h)", transform=axs[2].transAxes, fontsize=13, fontweight='bold')
        axs[2].text(0.82, 0.06, "OP2", transform=axs[2].transAxes, fontsize=11, fontweight='bold')

        # 2. Tight layout computes general box limits
        plt.tight_layout()
        
        # 3. Explicitly force a wider width space (wspace) between the columns.
        # 0.3 to 0.4 adds a very comfortable structural gap.
        plt.subplots_adjust(wspace=0.35)
        
        plt.show()


def JJ_diff_eq(t, y, I, alpha):
    phi, V = y  
    dphi_dt = V
    dv_dt = I - np.sin(phi) - alpha * V
    
    return [dphi_dt, dv_dt]

def random_mask(size,seed):
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(-1,1,size)
