import numpy as np
import qutip as qt

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
