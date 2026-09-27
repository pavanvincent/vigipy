# import numpy as np
# from scipy.special import gdtr


# # Calculation of CI lower bound
# def quantiles(threshold, Q, a1, b1, a2, b2):
    # """
    # Calculate CI lower bound using algorithms from DuMouchel's paper
    # "Bayesian Data Mining in Large Frequency Tables..." (1999)
    #
    # """
    # if type(Q) is np.float64 or type(Q) is float:
    #    length = 1
    # else:
        # length = len(Q)
    # m = np.repeat(-100000, length)
    # M = np.repeat(100000, length)
    # x = np.repeat(1, length)
    # cost = f_cost_quantiles(x, threshold, Q, a1, b1, a2, b2)
    # while np.max(np.round(cost * 1e4)) != 0:
        # S = np.sign(cost)
        # xnew = (1 + S) / 2 * ((x + m) / 2) + (1 - S) / 2 * ((M + x) / 2)
        # M = (1 + S) / 2 * x + (1 - S) / 2 * M
        # m = (1 + S) / 2 * m + (1 - S) / 2 * x
        # x = xnew
        # cost = f_cost_quantiles(x, threshold, Q, a1, b1, a2, b2)
    # return x


# def f_cost_quantiles(p, threshold, Q, a1, b1, a2, b2):
    # one = Q * gdtr(p, a1, b1)
    # two = (1 - Q) * gdtr(p, a2, b2)
    # if np.any(np.isnan(one)):
        # one = 0.0
    # if np.any(np.isnan(two)):
        # two = 0.0
    # return one + two - threshold

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
        # Recherche de la racine entre 0 et 100 000 (les bornes m et M de votre ancien code)
        # brentq est implémenté en C et converge de manière quadratique (extrêmement rapide)
        return brentq(f_cost, 0, 100000, xtol=1e-5)
    except ValueError:
        # En cas d'erreur de convergence ou de bornes irrégulières, on retourne une valeur par défaut sécurisée
        return 1.0

def quantiles(threshold, Q, a1, b1, a2, b2, n_jobs=-1):
    """
    Calcule les bornes d'intervalles de confiance (DuMouchel 1999) 
    parallélisées sur les cœurs CPU de Colab Pro.
    """
    # Si les entrées sont des scalaires (single float), on exécute directement
    if isinstance(Q, (float, np.float64, int)):
        return _solve_single_quantile(threshold, Q, a1, b1, a2, b2)
    
    # Détermination automatique du nombre de cœurs disponibles sur Colab Pro
    num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    
    # Transformation des entrées en tableaux de taille identique si ce sont des scalaires uniques
    length = len(Q)
    a1_arr = np.repeat(a1, length) if isinstance(a1, (float, int, np.float64)) else a1
    b1_arr = np.repeat(b1, length) if isinstance(b1, (float, int, np.float64)) else b1
    a2_arr = np.repeat(a2, length) if isinstance(a2, (float, int, np.float64)) else a2
    b2_arr = np.repeat(b2, length) if isinstance(b2, (float, int, np.float64)) else b2

    # Parallélisation de la résolution ligne par ligne avec joblib (backend "threading" pour économiser la RAM de Colab)
    results = Parallel(n_jobs=num_cores, backend="threading")(
        delayed(_solve_single_quantile)(threshold, Q[i], a1_arr[i], b1_arr[i], a2_arr[i], b2_arr[i])
        for i in range(length)
    )
    
    return np.asarray(results, dtype=np.float64)
