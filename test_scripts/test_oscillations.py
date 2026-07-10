import qutip as qt
import QRC
import numpy as np
import matplotlib.pyplot as plt
from QRC import normalize_subspace


def custom_pauli(N, state_a_index, state_b_index):
    sx_mat = np.zeros((N, N), dtype=complex)
    sy_mat = np.zeros((N, N), dtype=complex)
    sz_mat = np.zeros((N, N), dtype=complex)

    sx_mat[state_a_index, state_b_index] = 1.0
    sx_mat[state_b_index, state_a_index] = 1.0

    sy_mat[state_a_index, state_b_index] = -1j
    sy_mat[state_b_index, state_a_index] = 1j

    sz_mat[state_a_index, state_a_index] = 1.0
    sz_mat[state_b_index, state_b_index] = -1.0

    return [qt.Qobj(sx_mat), qt.Qobj(sy_mat), qt.Qobj(sz_mat)]

def custom_dissipator(N, destroy_index):
    destroy = np.zeros((N, N), dtype=complex)
    if destroy_index >= 1:
        destroy[destroy_index-1, destroy_index] = 1
        destroy = qt.Qobj(destroy)
    return destroy

#Run a test with 3 level system
N_dim = 3
subspace_index_offset = 1
kappa_low = 5
kappa_high = 0
pulse_duration = 5
pulse_time_steps = 100
g = 2.6

measurement_ops = custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)
sigma_x_subspace, sigma_y_subpace, sigma_z_subpace= custom_pauli(N_dim, subspace_index_offset, subspace_index_offset + 1)

destroy_ground = custom_dissipator(N_dim, subspace_index_offset)
destroy_excited = custom_dissipator(N_dim, subspace_index_offset + 1)

c_ops = [
    np.sqrt(kappa_low) * destroy_ground,
    np.sqrt(kappa_high) * destroy_excited
]

initial_state = qt.fock(N_dim, 1)

H_int = g*sigma_x_subspace

times = np.linspace(0,pulse_duration,pulse_time_steps)

result = qt.mesolve(H_int,initial_state,tlist=times,c_ops = c_ops)

spin_values = np.zeros((3,pulse_time_steps))

for t,state in enumerate(result.states):
    for i,op in enumerate(measurement_ops):
        state = normalize_subspace(state=state,subspace_dim=2,subspace_start_index=subspace_index_offset)[0]
        spin_values[i,t] = qt.expect(op,state)

# =====================================================================
# 3. MATPLOTLIB MULTI-PLOT WINDOW BLOCK
# =====================================================================
# Create a 3-row, 1-column canvas sharing the same X-axis (Time)
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

labels = [r"$\langle S_x \rangle$", r"$\langle S_y \rangle$", r"$\langle S_z \rangle$"]
colors = ["crimson", "teal", "royalblue"]

# Loop through each row axis and plot its corresponding spin calculation array
for i in range(3):
    axs[i].plot(times, spin_values[i], lw=2, color=colors[i], label=labels[i])
    axs[i].set_ylabel("Expectation Value", fontsize=11)
    axs[i].grid(alpha=0.3)
    axs[i].legend(loc="upper right")
    
# Clean up global decorations on specific panels
axs[0].set_title("Subspace Bloch Vector Components", fontsize=14, fontweight="bold")
axs[2].set_xlabel("Time", fontsize=12)

# Prevent title/label overlaps automatically
plt.tight_layout()
plt.show()

# test_QRC = QRC.QRC_TimeSeries(N=N_dim,
#                               dissipations=c_ops,
#                               H_base=qt.qeye(3),
#                               H_interaction=H_int,
#                               measurement_ops=measurement_ops,
#                               initial_state=initial_state,
#                               subspace_dim=2,
#                               subspace_offset=1)