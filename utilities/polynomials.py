from scipy.special import eval_legendre,legendre
import numpy as np






def normalized_legendre(n, x):
            # Standard Legendre polynomial of degree n
            pn = legendre(n)
            # L2 normalization factor for interval [-1, 1]
            norm_factor = np.sqrt((2 * n + 1) / 2)
            return norm_factor * pn(x)
