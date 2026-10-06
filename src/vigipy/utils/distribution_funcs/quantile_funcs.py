# import numpy as np
# from scipy.special import gdtr
# from scipy.optimize import brentq
# from joblib import Parallel, delayed
# import multiprocessing
#
# def _solve_single_quantile(threshold, Q, a1, b1, a2, b2):
#    """
#    Calcule le quantile pour une seule ligne de données en utilisant la méthode de Brent.
#    """
#    # Définition de la fonction objectif pour une seule observation
#    def f_cost(p):
#        val1 = Q * gdtr(p, a1, b1)
#        val2 = (1 - Q) * gdtr(p, a2, b2)
#        # Gestion propre des NaNs uniquement sur l'élément courant
#        val1 = 0.0 if np.isnan(val1) else val1
#        val2 = 0.0 if np.isnan(val2) else val2
#        return val1 + val2 - threshold
#
#    try:
#        # Finding the root between 0 and 100,000 
#        # brentq is implemented in C and converges quadratically (extremely fast)
#        return brentq(f_cost, 0, 100000, xtol=1e-5)
#    except ValueError:
#        # In case of convergence errors or irregular bounds, a safe default value is returned.
#        return 1.0
#
# def quantiles(threshold, Q, a1, b1, a2, b2, n_jobs=-1):
#   """
#    Calculates the confidence interval bounds (DuMouchel 1999)
#    parallelized on the CPU cores 
#    """
#    # If the inputs are scalars (single float), we execute directly
#    if isinstance(Q, (float, np.float64, int)):
#        return _solve_single_quantile(threshold, Q, a1, b1, a2, b2)
#    
#    # Automatic determination of the number of available cores 
#    num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
#    
#    # Transforming the inputs into arrays of identical size if they are unique scalars
#    length = len(Q)
#    a1_arr = np.repeat(a1, length) if isinstance(a1, (float, int, np.float64)) else a1
#    b1_arr = np.repeat(b1, length) if isinstance(b1, (float, int, np.float64)) else b1
#    a2_arr = np.repeat(a2, length) if isinstance(a2, (float, int, np.float64)) else a2
#    b2_arr = np.repeat(b2, length) if isinstance(b2, (float, int, np.float64)) else b2
#
#    # Parallelizing line-by-line resolution with joblib
#    results = Parallel(n_jobs=num_cores, backend="threading")(
#        delayed(_solve_single_quantile)(threshold, Q[i], a1_arr[i], b1_arr[i], a2_arr[i], b2_arr[i])
#        for i in range(length)
#    )
#    
#    return np.asarray(results, dtype=np.float64)


import numpy as np
from scipy.special import gdtr


# Calculation of CI lower bound
def quantiles(threshold, Q, a1, b1, a2, b2, max_iter: int = 35, tol: float = 1e-5):
    """Calculate empirical Bayes quantiles for a two-component Gamma-Poisson mixture.

    Implements a vectorized bisection search algorithm based on DuMouchel (1999)
    to invert the mixture cumulative distribution function:
        F(x) = Q * gdtr(x, a1, b1) + (1 - Q) * gdtr(x, a2, b2) = threshold

    Parameters:
        threshold: Cumulative probability target (e.g. 0.05 for EB05, 0.95 for EB95).
        Q: Prior mixture weights.
        a1: Shape parameters for component 1 (alpha1 + n11).
        b1: Rate/scale parameters for component 1 (beta1 + E).
        a2: Shape parameters for component 2 (alpha2 + n11).
        b2: Rate/scale parameters for component 2 (beta2 + E).
        max_iter: Maximum number of bisection iterations (default 35).
        tol: Convergence tolerance on CDF error |F(x) - threshold| (default 1e-5).

    Returns:
        Quantile values corresponding to the target threshold (scalar or ndarray).
    """
    is_scalar = np.ndim(Q) == 0
    Q_arr = np.asarray(Q, dtype=np.float64)
    a1_arr = np.asarray(a1, dtype=np.float64)
    b1_arr = np.asarray(b1, dtype=np.float64)
    a2_arr = np.asarray(a2, dtype=np.float64)
    b2_arr = np.asarray(b2, dtype=np.float64)

    if is_scalar:
        Q_arr, a1_arr, b1_arr, a2_arr, b2_arr = np.atleast_1d(
            Q_arr, a1_arr, b1_arr, a2_arr, b2_arr
        )

    max_mean = np.maximum(
        1000.0,
        10.0 * np.maximum(
            a1_arr / np.maximum(b1_arr, 1e-6),
            a2_arr / np.maximum(b2_arr, 1e-6),
        ),
    )
    M = max_mean.copy()
    m = np.zeros_like(M)
    x = np.ones_like(M)

    for _ in range(max_iter):
        one = Q_arr * gdtr(x, a1_arr, b1_arr)
        two = (1.0 - Q_arr) * gdtr(x, a2_arr, b2_arr)
        one = np.nan_to_num(one, nan=0.0)
        two = np.nan_to_num(two, nan=0.0)
        cost = one + two - threshold

        if np.all(np.abs(cost) < tol):
            break

        S = np.sign(cost)
        xnew = np.where(S > 0, (x + m) * 0.5, (M + x) * 0.5)
        M = np.where(S > 0, x, M)
        m = np.where(S > 0, m, x)
        x = xnew

    if is_scalar:
        return float(x[0])
    return x


def f_cost_quantiles(p, threshold, Q, a1, b1, a2, b2):
    """Compute the difference between the mixture CDF and the threshold."""
    one = Q * gdtr(p, a1, b1)
    two = (1.0 - Q) * gdtr(p, a2, b2)
    one = np.nan_to_num(one, nan=0.0)
    two = np.nan_to_num(two, nan=0.0)
    return one + two - threshold
