import numpy as np
import scipy.integrate as integrate
from scipy.special import eval_legendre
import matplotlib.pyplot as plt

# Define the degrees of the two Legendre polynomials
m = 15
n = 15


# # Integrate the product from -1 to 1
# result, error = integrate.quad(lambda x: P_m(x) * P_n(x), -1, 1)

def ortho_legendre_11(n, x):
    """
    Evaluates the n-th degree orthonormal Legendre polynomial on [-1, 1].
    
    Parameters:
        n (int): Degree of the polynomial
        x (float or np.ndarray): Input values in the range [-1, 1]
    """
    # Standard Legendre on [-1, 1] multiplied by sqrt(2n + 1)
    return np.sqrt(2 * n + 1) * eval_legendre(n, x)
print(ortho_legendre_11(2,0.5))
