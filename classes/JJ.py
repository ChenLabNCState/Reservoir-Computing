import math
import os
from dataclasses import asdict, dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy
from classes.RC import RC_TimeSeries


@dataclass(kw_only=True)
class JJ(RC_TimeSeries):
    """Class to implement the Josephson Junction Reservoir Computing model."""

    virtual_nodes: int = 20
    theta: float
    dt: float = field(init=False)
    time_steps: int = 50
    alpha: float = 1.5
    initial_phi: float = 0.1
    initial_v: float = 1
    k_inj: float = 0.25
    I_dc: float = 1.5
    plotting_substeps: int = 10

    random_seed: int = field(default=21, init=True)

    def __post_init__(self):
        self.dt = self.virtual_nodes * self.theta
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def simulate_data(self, data, is_train: bool, save_dynamics: bool):
        node_results = np.zeros((self.virtual_nodes, len(data)))
        phi = self.initial_phi
        V = self.initial_v

        rng = np.random.default_rng(seed=self.random_seed)
        mask = rng.uniform(-1, 1, self.virtual_nodes)

        for i, value in enumerate(data):
            masked_data = self.I_dc + value * mask * self.k_inj
            sub_dt = self.dt / self.virtual_nodes

            for j, input_current in enumerate(masked_data):
                start = i * self.dt + j * sub_dt
                end = i * self.dt + (j + 1) * sub_dt
                y0 = [phi, V]

                solution = scipy.integrate.solve_ivp(
                    JJ_diff_eq,
                    (start, end),
                    y0,
                    args=(input_current, self.alpha),
                    dense_output=True,
                )

                phi = solution.y[0][-1]
                V = solution.y[1][-1]

                if is_train and save_dynamics:
                    t_steps = np.linspace(
                        start, end, self.plotting_substeps
                    )
                    y_steps = solution.sol(t_steps)

                    for k in range(len(t_steps)):
                        self.dynamics_data.append(
                            [y_steps[0, k], y_steps[1, k], input_current]
                        )

                node_results[j, i] = V

        return node_results

    # Helper method to calculate points per time step
    @property
    def points_per_step(self) -> int:
        return self.virtual_nodes * self.plotting_substeps

    def plot_dynamics(self, start_step: int, is_OP2: bool = False):
        if not self.dynamics_data:
            print("No dynamics data to plot.")
            return

        start_idx = start_step * self.points_per_step
        end_idx = start_idx + (10 * self.points_per_step)

        dynamics_array = np.array(self.dynamics_data)
        input_offset = self.I_dc

        V = dynamics_array[start_idx:end_idx, 1]
        current = dynamics_array[start_idx:end_idx, 2]
        t_values = np.arange(len(V))

        if not is_OP2:
            current = current - input_offset

        plt.plot(t_values, V, "b-", linewidth=1.5, label="V")
        plt.plot(t_values, current, "g-", linewidth=1.5, label="I_dc mask")
        plt.xlabel("Substep Index")
        plt.ylabel("Magnitude")
        plt.title("Current vs Voltage Dynamics")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    def plot_dynamics_2(self, start_step: int, cycles: int, is_OP2: bool = False):
        if not self.dynamics_data:
            print("No dynamics data to plot.")
            return

        dynamics_array = np.array(self.dynamics_data)
        start_idx = start_step * self.points_per_step
        end_idx = start_idx + (cycles * self.points_per_step)

        phi_vals = dynamics_array[start_idx:end_idx, 0]
        V = dynamics_array[start_idx:end_idx, 1]

        y_label = r"Phase ($\phi$)"
        if is_OP2:
            phi_vals = np.sin(phi_vals)
            y_label = r"$\sin\phi_{\mathrm{output}}$"

        plt.figure(figsize=(8, 6))
        plt.plot(V, phi_vals, "b-", linewidth=1.5, label="Trajectory")
        plt.xlabel("Voltage (V)")
        plt.ylabel(y_label)
        plt.title("Josephson Junction Phase Space Dynamics")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    def plot_dynamics_3(self, start_step: int, cycles: int, is_OP2: bool = False):
        if not self.dynamics_data:
            print("No dynamics data to plot.")
            return

        dynamics_array = np.array(self.dynamics_data)
        start_idx = start_step * self.points_per_step
        end_idx = start_idx + (cycles * self.points_per_step)

        phi_vals = dynamics_array[start_idx:end_idx, 0]
        v_vals = dynamics_array[start_idx:end_idx, 1]

        y_label = r"Phase ($\phi$)"
        if is_OP2:
            phi_vals = np.sin(phi_vals)
            y_label = r"$\sin\phi_{\mathrm{output}}$"

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
        ax.plot(v_vals, phi_vals, color="blue", linewidth=2, label="Trajectory")

        sample_indices = np.linspace(0, len(v_vals) - 1, 12, dtype=int)
        for idx, i in enumerate(sample_indices):
            if idx % 2 == 0:
                ax.plot(v_vals[i], phi_vals[i], "g^", markersize=9)
            else:
                ax.plot(v_vals[i], phi_vals[i], "ro", markersize=8)

        ax.set_xlabel(r"$V_{\mathrm{output}}$", fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        ax.set_title(r"$V_{\mathrm{output}}$", fontsize=12, pad=10)

        plt.tight_layout()
        plt.show()

    def plot_combined_dynamics(self, start_step: int, phase_cycles: int, phase_cycles2: int):
        if not self.dynamics_data:
            print("No dynamics data to plot.")
            return

        dynamics_array = np.array(self.dynamics_data)

        phi_all = dynamics_array[:, 0]
        v_all = dynamics_array[:, 1]
        current_all = dynamics_array[:, 2]

        fig, axs = plt.subplots(1, 3, figsize=(20, 4.5), dpi=150)

        # ==========================================
        # SUBPLOT 1: Current vs Voltage Dynamics
        # ==========================================
        start_idx = start_step * self.points_per_step
        end_idx = start_idx + (10 * self.points_per_step)

        v_slice = v_all[start_idx:end_idx]
        current_slice = current_all[start_idx:end_idx]
        t_values = np.arange(len(v_slice))

        axs[0].plot(t_values, v_slice, "b-", linewidth=1.5, label="V")
        axs[0].plot(
            t_values, current_slice, "g-", linewidth=1.5, label=r"$I_{\mathrm{dc}}$ mask"
        )
        axs[0].set_xlabel("Substep Index", fontsize=11)
        axs[0].set_ylabel("Magnitude", fontsize=11)
        axs[0].set_title(
            "Current vs Voltage Dynamics", fontsize=11, fontweight="bold", pad=10
        )
        axs[0].grid(True, linestyle="--", alpha=0.5)
        axs[0].legend(loc="upper right", fontsize=9)
        axs[0].text(
            0.03, 0.92, "(a)", transform=axs[0].transAxes, fontsize=13, fontweight="bold"
        )

        # ==========================================
        # SUBPLOT 2: Phase Space (Multi-cycle Wrapped Line)
        # ==========================================
        p2_start = start_step * self.points_per_step
        p2_end = p2_start + (phase_cycles * self.points_per_step)

        phi_p2 = phi_all[p2_start:p2_end]
        v_p2 = v_all[p2_start:p2_end]

        phi_wrapped = phi_p2 % (2 * np.pi)

        diffs = np.abs(np.diff(phi_wrapped))
        jump_indices = np.where(diffs > np.pi)[0]
        v_plot2 = np.insert(v_p2, jump_indices + 1, np.nan)
        phi_plot2 = np.insert(phi_wrapped, jump_indices + 1, np.nan)

        axs[1].plot(v_plot2, phi_plot2, "b-", linewidth=1.5, label="Trajectory")
        axs[1].set_xlabel("Voltage (V)", fontsize=11)
        axs[1].set_ylabel(r"Phase ($\phi$ mod $2\pi$)", fontsize=11)
        axs[1].set_title(
            "Josephson Phase Space Dynamics", fontsize=11, fontweight="bold", pad=10
        )
        axs[1].grid(True, linestyle="--", alpha=0.5)
        axs[1].legend(loc="upper right", fontsize=9)
        axs[1].text(
            0.03, 0.92, "(b)", transform=axs[1].transAxes, fontsize=13, fontweight="bold"
        )

        # ==========================================
        # SUBPLOT 3: Single Cycle Orbit Style
        # ==========================================
        p3_start = start_step * self.points_per_step
        p3_end = p3_start + (phase_cycles2 * self.points_per_step)

        phi_p3 = phi_all[p3_start:p3_end]
        v_p3 = v_all[p3_start:p3_end]
        sin_phi = np.sin(phi_p3)

        axs[2].plot(v_p3, sin_phi, color="blue", linewidth=2, label="Trajectory")

        sample_indices = np.linspace(0, len(v_p3) - 1, 12, dtype=int)
        for idx, i in enumerate(sample_indices):
            if idx % 2 == 0:
                axs[2].plot(v_p3[i], sin_phi[i], "g^", markersize=8)
            else:
                axs[2].plot(v_p3[i], sin_phi[i], "ro", markersize=7)

        axs[2].set_xlabel(r"$V_{\mathrm{output}}$", fontsize=11)
        axs[2].set_ylabel(r"$\sin\phi_{\mathrm{output}}$", fontsize=11)
        axs[2].set_title(
            "Josephson Junction Orbit", fontsize=11, fontweight="bold", pad=10
        )
        axs[2].set_ylim(-1.5, 1.5)
        axs[2].set_yticks([-1, 0, 1])

        axs[2].minorticks_on()
        axs[2].tick_params(axis="both", which="major", direction="out", length=5)
        axs[2].text(
            0.03, 0.92, "(h)", transform=axs[2].transAxes, fontsize=13, fontweight="bold"
        )
        axs[2].text(
            0.82, 0.06, "OP2", transform=axs[2].transAxes, fontsize=11, fontweight="bold"
        )

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        plt.show()


def JJ_diff_eq(t, y, I, alpha):
    phi, V = y
    dphi_dt = V
    dv_dt = I - np.sin(phi) - alpha * V
    return [dphi_dt, dv_dt]


def random_mask(size, seed):
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(-1, 1, size)