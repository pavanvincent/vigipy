import numpy as np
from scipy.special import gdtr
from scipy.optimize import brentq
from joblib import Parallel, delayed
import multiprocessing

def _solve_single_quantile(threshold, Q, a1, b1, a2, b2):
    """
    Calcule le quantile pour une seule ligne de données en utilisant la méthode de Brent.
    """
    # Définition de la fonction objectif pour une seule observation
    def f_cost(p):
        val1 = Q * gdtr(p, a1, b1)
        val2 = (1 - Q) * gdtr(p, a2, b2)
        # Gestion propre des NaNs uniquement sur l'élément courant
        val1 = 0.0 if np.isnan(val1) else val1
        val2 = 0.0 if np.isnan(val2) else val2
        return val1 + val2 - threshold

    try:
        # Finding the root between 0 and 100,000 
        # brentq is implemented in C and converges quadratically (extremely fast)
        return brentq(f_cost, 0, 100000, xtol=1e-5)
    except ValueError:
        # In case of convergence errors or irregular bounds, a safe default value is returned.
        return 1.0

def quantiles(threshold, Q, a1, b1, a2, b2, n_jobs=-1):
    """
    Calculates the confidence interval bounds (DuMouchel 1999)
    parallelized on the CPU cores 
    """
    # If the inputs are scalars (single float), we execute directly
    if isinstance(Q, (float, np.float64, int)):
        return _solve_single_quantile(threshold, Q, a1, b1, a2, b2)
    
    # Automatic determination of the number of available cores 
    num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    
    # Transforming the inputs into arrays of identical size if they are unique scalars
    length = len(Q)
    a1_arr = np.repeat(a1, length) if isinstance(a1, (float, int, np.float64)) else a1
    b1_arr = np.repeat(b1, length) if isinstance(b1, (float, int, np.float64)) else b1
    a2_arr = np.repeat(a2, length) if isinstance(a2, (float, int, np.float64)) else a2
    b2_arr = np.repeat(b2, length) if isinstance(b2, (float, int, np.float64)) else b2

    # Parallelizing line-by-line resolution with joblib
    results = Parallel(n_jobs=num_cores, backend="threading")(
        delayed(_solve_single_quantile)(threshold, Q[i], a1_arr[i], b1_arr[i], a2_arr[i], b2_arr[i])
        for i in range(length)
    )
    
    return np.asarray(results, dtype=np.float64)
