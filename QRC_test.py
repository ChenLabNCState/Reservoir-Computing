"""
QRC with the input encoded as the DRIVE AMPLITUDE of a transverse-field Ising model.
 
Difference from Fujii-Nakajima:
    they encode s_k by RESETTING a qubit (a non-unital channel -> fading memory for free).
    Here s_k multiplies the drive term in H, and the evolution is unitary, so there is
    NO forgetting. We therefore add Lindblad dissipation; gamma is now the knob that
    sets the memory timescale, playing the role the reset channel used to play.
 
    H(s_k) = sum_{i<j} J_ij X_i X_j  +  h sum_i Z_i  +  A * s_k * sum_i X_i
    drho/dt = -i[H(s_k), rho] + gamma * sum_i D[sigma_i^-](rho)      for time tau
 
Because H is piecewise-constant on each interval, we build the Liouvillian once as
L(s) = L0 + s * Ldrive and exponentiate it per step -- no ODE solver needed.
"""
 
import numpy as np
from scipy.linalg import expm
from qutip import qeye, sigmax, sigmaz, sigmam, tensor, liouvillian
 
rng = np.random.default_rng(0)
 
 
def mackey_glass(n_out, tau_mg=17, sigma=0.1, subsample=10, y0=1.2, burn=1000):
    delay = int(round(tau_mg / sigma))
    n_raw = (n_out + burn) * subsample + delay + 10
    y = np.empty(n_raw)
    y[:delay + 1] = y0 + 0.01 * rng.standard_normal(delay + 1)
    for k in range(delay, n_raw - 1):
        yd = y[k - delay]
        y[k + 1] = y[k] + sigma * (0.2 * yd / (1.0 + yd**10) - 0.1 * y[k])
    s = y[delay::subsample][burn:burn + n_out]
    return (s - s.min()) / (s.max() - s.min())
 
 
class DrivenIsingReservoir:
    def __init__(self, n_qubits=4, V=5, tau=2.0, J=1.0, h=1.0,
                 drive=2.0, gamma=0.15, seed=1):
        self.N, self.V, self.tau, self.A = n_qubits, V, tau, drive
        rg = np.random.default_rng(seed)
        d = 2 ** n_qubits
 
        def op(o, i):
            return tensor([o if j == i else qeye(2) for j in range(n_qubits)])
 
        # --- static part of the Hamiltonian ---
        H0 = 0 * op(qeye(2), 0)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                H0 += rg.uniform(-J / 2, J / 2) * op(sigmax(), i) * op(sigmax(), j)
            H0 += h * op(sigmaz(), i)
 
        # --- the term the INPUT multiplies ---
        Hd = sum(op(sigmax(), i) for i in range(n_qubits))
 
        # --- dissipation: this is what gives the reservoir fading memory ---
        c_ops = [np.sqrt(gamma) * op(sigmam(), i) for i in range(n_qubits)]
 
        # L(s) = L0 + s * Ldrive  (superoperators, d^2 x d^2)
        self.L0 = liouvillian(H0, c_ops).full()
        self.Ld = liouvillian(self.A * Hd).full()
 
        self.Z = [op(sigmaz(), i).full() for i in range(n_qubits)]
        self.X = [op(sigmax(), i).full() for i in range(n_qubits)]
        self.d = d
        self.rho = np.zeros((d, d), complex); self.rho[0, 0] = 1.0
        self._cache = {}
 
    def _prop(self, s):
        """exp(L(s) * tau/V), cached on a rounded s so repeated values are free."""
        key = round(float(s), 4)
        if key not in self._cache:
            if len(self._cache) > 40000:
                self._cache.clear()
            self._cache[key] = expm((self.L0 + key * self.Ld) * (self.tau / self.V))
        return self._cache[key]
 
    def step(self, s):
        """Drive the reservoir with amplitude s for time tau, reading out at V
        subdivided times -> N*2*V virtual nodes (<Z_i> and <X_i>)."""
        P = self._prop(np.clip(s, 0.0, 1.0))
        feats = []
        v = self.rho.reshape(-1)                       # column-stacking convention
        for _ in range(self.V):
            v = P @ v
            rho = v.reshape(self.d, self.d)
            feats += [np.real(np.trace(O @ rho)) for O in self.Z]
            feats += [np.real(np.trace(O @ rho)) for O in self.X]
        self.rho = v.reshape(self.d, self.d)
        return np.asarray(feats)
 
 
def run(n_qubits=4, V=5, tau=2.0, gamma=0.15, drive=2.0,
        L=1500, washout=150, n_auto=300, noise=1e-4, ridge=1e-7, seed=1, verbose=True):
 
    res = DrivenIsingReservoir(n_qubits, V, tau, drive=drive, gamma=gamma, seed=seed)
    mg = mackey_glass(washout + L + n_auto + 2)
 
    # ---------- phase 1: teacher forcing (single forward sweep) ----------
    X, Y = [], []
    for k in range(washout + L):
        f = res.step(mg[k] + rng.uniform(-noise, noise))
        if k >= washout:
            X.append(np.concatenate(([1.0], f)))
            Y.append(mg[k + 1])
    X, Y = np.asarray(X), np.asarray(Y)
 
    # ---------- phase 2: one linear solve ----------
    w = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)
    tf_nmse = np.mean((X @ w - Y) ** 2) / np.var(Y)
 
    # ---------- phase 3: close the loop (rho is NOT reset) ----------
    f = res.step(mg[washout + L])
    gen = []
    for _ in range(n_auto):
        y = float(np.dot(w, np.concatenate(([1.0], f))))
        gen.append(y)
        f = res.step(y)
    gen = np.asarray(gen)
    tgt = mg[washout + L + 1: washout + L + 1 + n_auto]
    auto_nmse = np.mean((gen - tgt) ** 2) / np.var(tgt)
 
    if verbose:
        print(f"N={n_qubits} V={V} tau={tau} gamma={gamma} A={drive}  "
              f"nodes={len(f)}  teacher-forced NMSE={tf_nmse:.2e}  "
              f"autonomous NMSE={auto_nmse:.2e}  std(gen)={gen.std():.3f}")
    return dict(w=w, gen=gen, tgt=tgt, tf_nmse=tf_nmse, auto_nmse=auto_nmse)
 
 
if __name__ == "__main__":
    print("--- gamma sets the memory: too large forgets, too small never forgets ---")
    for g in (0.5, 0.15, 0.05, 0.01):
        run(gamma=g)
