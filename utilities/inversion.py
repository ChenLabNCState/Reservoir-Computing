import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
from enum import Enum


class reg_type(Enum):
    PINV = 1
    RIDGE = 2

def invert(data,inv_type:reg_type = reg_type.PINV,ridge:float = 1e-3):
    match inv_type:
        case reg_type.PINV:
            inverse_train = np.linalg.pinv(data)

        case reg_type.RIDGE:
            inverse_train = np.linalg.inv(data.T @ data 
                                        + ridge * np.eye(data.shape[1]) ) @ data.T

    return inverse_train