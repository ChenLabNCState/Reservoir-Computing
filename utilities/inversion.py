import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
from enum import Enum


class RegType(Enum):
    PINV = 1
    RIDGE = 2

def invert(data,inv_type:RegType = RegType.PINV,ridge:float = 1e-3):
    match inv_type:
        case RegType.PINV:
            inverse_train = np.linalg.pinv(data)

        case RegType.RIDGE:
            inverse_train = np.linalg.inv(data.T @ data 
                                        + ridge * np.eye(data.shape[1]) ) @ data.T

    return inverse_train