import matplotlib.pyplot as plt
import numpy as np
import scqubits as scq

# -----------------------------------------------------------------------------
# 1. Transmon Qubit Setup & Plotting
# -----------------------------------------------------------------------------
# Parameters in GHz
transmon = scq.Transmon(
    EJ=30.0,
    EC=1.0,
    ng=0.0,
    ncut=31
)

# Plot Transmon spectrum vs. offset charge (ng)
fig, ax = transmon.plot_evals_vs_paramvals(
    param_name='ng',
    param_vals=np.linspace(-1.0, 1.0, 100),
    evals_count=4,
    color='blue',
    show=False
)
ax.set_title("Transmon: Energy Levels vs. Offset Charge ($n_g$)")
plt.tight_layout()
plt.show()

# Plot Transmon wavefunctions in phase basis (phi)
fig, ax = transmon.plot_wavefunction(
    which=[0, 1, 2],
    mode='real',
    show=False
)
ax.set_title("Transmon: Wavefunctions in $\phi$-basis")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 2. Fluxonium Qubit Setup & Plotting
# -----------------------------------------------------------------------------
# Parameters in GHz
fluxonium = scq.Fluxonium(
    EJ=8.9,
    EC=2.5,
    EL=0.5,
    flux=0.5,  # Half-flux quantum (frustration point)
    cutoff=110
)

# Plot Fluxonium spectrum vs. external flux
fig, ax = fluxonium.plot_evals_vs_paramvals(
    param_name='flux',
    param_vals=np.linspace(0.0, 1.0, 100),
    evals_count=5,
    color='red',
    show=False
)
ax.set_title("Fluxonium: Energy Levels vs. External Flux ($\Phi/\Phi_0$)")
plt.tight_layout()
plt.show()

# Plot Fluxonium wavefunctions potential overlay at flux = 0.5
fig, ax = fluxonium.plot_wavefunction(
    which=[0, 1, 2],
    mode='real',
    show=False
)
ax.set_title("Fluxonium: Wavefunctions & Potential Well at $\Phi = 0.5 \Phi_0$")
plt.tight_layout()
plt.show()