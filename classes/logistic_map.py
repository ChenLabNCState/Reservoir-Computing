import numpy as np
from dataclasses import dataclass, field
from classes.RC import RC_TimeSeries
 
 
# ── Piecewise linear saturation  S(y) ────────────────────────────────────────
def S(y: float) -> float:
    """Piecewise linear saturation function from the paper."""
    if y < 0:
        return 0.0
    elif y > 1:
        return 1.0
    else:
        return y
 
 
# ── Driven logistic map  x(t+1) = ρ · v(t) · (1 − v(t)) ─────────────────────
# v(t) = S(x(t) + i · u(t))
def driven_logistic_step(x: float, u: float, rho: float, i_inj: float) -> float:
    v = S(x + i_inj * u)
    return rho * v * (1.0 - v)
 
 
# ── RC subclass ───────────────────────────────────────────────────────────────
@dataclass(kw_only=True)
class LogisticMapRC(RC_TimeSeries):
    """
    Reservoir computer using the driven logistic map from:
      Appeltant et al., Scientific Reports 2, 514 (2012)
 
      x(t+1) = rho * v(t) * (1 - v(t))
      v(t)   = S(x(t) + i_inj * u(t))
 
    One state x per input step — no time-multiplexing.
    Output shape: (1, len(data))
    """
 
    rho: float = 3.7       # bifurcation parameter  (ρ in the paper)
    i_inj: float = 0.2     # injection scaling       (i in the paper)
    initial_x: float = 0.5
 
    def simulate_data(
        self,
        data: np.ndarray,
        is_train: bool,
        save_dynamics: bool = False,
    ) -> np.ndarray:
        """
        Drive the logistic map with each input sample and collect x(t).
 
        Returns
        -------
        node_results : ndarray, shape (1, len(data))
        """
        n_samples = len(data)
        node_results = np.zeros((1, n_samples))
 
        x = self.initial_x
 
        for t, u_t in enumerate(data):
            x = driven_logistic_step(x, u_t, self.rho, self.i_inj)
 
            if save_dynamics and is_train:
                self.dynamics_data.append([t, x])
 
            node_results[0, t] = x
 
        return node_results
 
 
# ── Data generation helpers ───────────────────────────────────────────────────
 
def generate_logistic_sequence(
    n: int,
    rho: float = 3.9,
    x0: float = 0.5,
    n_transient: int = 100,
) -> np.ndarray:
    """Free-running logistic map x(t+1) = rho*x*(1-x), transients discarded."""
    x = x0
    for _ in range(n_transient):
        x = rho * x * (1.0 - x)
    out = np.empty(n)
    for t in range(n):
        x = rho * x * (1.0 - x)
        out[t] = x
    return out
 
