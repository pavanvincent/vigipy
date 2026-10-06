# ﻿import pandas as pd
# import numpy as np
# import warnings
# from scipy.special import gdtr, digamma
# from scipy.stats import nbinom
# from scipy.optimize import minimize

# from ..utils import Container
# from ..utils import calculate_expected
# from ..utils.distribution_funcs.negative_binomials import dnbinom, pnbinom
# from ..utils.distribution_funcs.quantile_funcs import quantiles

# dnbinom = np.vectorize(dnbinom)
# pnbinom = np.vectorize(pnbinom)

# EPS = np.finfo(np.float64).eps
# BOUNDED_METHODS = {
#     "Nelder-Mead",
#     "L-BFGS-B",
#     "TNC",
#     "SLSQP",
#     "Powell",
#     "trust-constr",
#     "COBYLA",
#     "COBYQA",
# }


# def gps(
#     container,
#     min_events=4,
#     truncate=False,
#     maxiter=500,
#     criterion= "EB05>=2",
#     n_jobs = -1,
# ):
#     """
#     Computes signal detection based on Multi-item enabled Gamma Poisson Shrinkage (GPS) using prior distributions
#     for adverse event and product feature data.

#     Parameters:
#     -----------
#     container : object
#         A container object holding the input data, including event counts (`events`),
#         product-event pairs (`product_aes`), and across-brand counts (`count_across_brands`).
#     min_events : int, optional (default=4)
#         The minimum number of events required for an adverse event to be considered in the analysis.
#     truncate : bool, optional (default=False)
#     maxiter : maximal number of iterations in optimization algorithm
#     criterion : usual FDA safety alert criterion. Can be either
#         - "EB05 >=2"  or - "EB05>1 & EGBM>=2"
#     n_jobs = number of cores for paralelization
    
#     Returns:
#     --------
#     RES : object
#         A container object with the following attributes:
#         - `param`: A dictionary of input parameters, including prior initialization and optimization results.
#         - `all_signals`: A DataFrame containing detailed results of signal detection, including posterior probabilities,
#           expected counts, and ranking statistics.
#         - `signals`: A DataFrame of filtered signals according to the decision metric and threshold. Here this is EB05 >= 2 (FDA decision metric)
#         - `num_signals`: The number of signals detected based on the decision rule.

#     Notes:
#     ------
#     - This function implements a Bayesian model to calculate posterior probabilities using a mixture of two negative
#       binomial distributions.
#     - The optimization process is used to estimate the prior parameters unless provided manually.
#     - The function can handle truncation for numerical stability when dealing with sparse data.
#     """

    
    
#     input_params = locals()
#     del input_params["container"]

#     relative_risk=1,
#     truncate_thres=1
#     prior_param=None
#     expected_method="mantel-haentzel"
#     method_alpha=1
#     minimization_method="SLSQP"
#     minimization_bounds=((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))
#     minimization_options=None

#     alpha1 =0.5*(EPS + 20) 
#     beta1 =0.5* (EPS + 10)
#     alpha2 = 0.5*(EPS + 20)
#     beta2 = 0.5*(EPS +10)
#     w = 0.5*(0 + 1)
#     priors = np.asarray([alpha1, beta1, alpha2, beta2, w])

#     computation_params = pd.DataFrame.from_dict({
#         "minimization_method": minimization_method,
#         "minimization_bounds": minimization_bounds,
#         "priori_init": priors,
#         "method_alpha": method_alpha,
#         "expected_method": expected_method,
#         "minimization_options": minimization_options
#         }, orient="index", columns=["Value"])

#     DATA = container.data
#     N = container.N

#     #-------------------------------------------------
#     # Compute expected values using expected_method
#     #------------------------------------------------
#     n11 = np.asarray(DATA["events"], dtype=np.float64)
#     n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
#     ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
#     expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)
#     p_out = True

#     #--------------------------------------------------------
#     # exclude product / ae pairs with low numbers of events
#     #--------------------------------------------------------
#     DATA = DATA[DATA.events >= min_events]
#     expected = expected[n11 >= min_events]
#     n1j = n1j[n11 >= min_events]
#     ni1 = ni1[n11 >= min_events]
#     n11 = n11[n11 >= min_events]
   
#     n10 = n1j - n11
#     n01 = ni1 - n11 + 1e-7
#     n00 = N - (n11 + n10 + n01)

#     #----------------------------------------------------------------------------------------
#     # Launch optimization algorithm to find hypergeometrical parameters of the prior
#     # priori is sum of two independant Gamma laws, whose parameters are 
#     # alpha_1, beta_1, alpha_2, beta_2, w
#     # such that prior density is : w * Gamma(alpha_1, beta_1) + (1-w) * Gamma(alpha_2,beta_2)
#     # optimization algorithm finds alpha_1, beta_1, alpha_2, beta_2, w by minimizing
#     # 1) either non truncated likelihood (imput argument truncated = False)
#     # 2) either truncated objective likelihood (imput argument truncated = true
#     #-----------------------------------------------------------------------------------------
#     p_out = False
#     if minimization_method not in BOUNDED_METHODS:
#         minimization_bounds = None

#     if minimization_options is None:
#         minimization_options = {}

#     if not truncate:
#         data_cont = container.contingency
#         n1__mat = data_cont.sum(axis=1)
#         n_1_mat = data_cont.sum(axis=0)
#         rep = len(n_1_mat)
#         n1__c = np.tile(np.asarray(n1__mat), reps=rep) 
#         rep = len(n1__mat)
#         n_1_c = np.repeat(np.asarray(n_1_mat), repeats=rep) 
#         E_c = np.asarray(n1__c, dtype=np.float64) * n_1_c / N
#         n11_c = np.asarray(data_cont).flatten(order="F")
        
#         p_out = minimize(
#             non_truncated_likelihood,
#             x0=priors,
#             args=(n11_c, E_c),
#             options={"maxiter": maxiter},
#             method=minimization_method,
#             bounds=minimization_bounds,
#             **minimization_options,
#         )
#     elif truncate:
#         trunc = truncate_thres - 1
#         p_out = minimize(
#             truncated_likelihood,
#             x0=priors,
#             args=(
#                 n11[n11 >= truncate_thres],
#                 expected[n11 >= truncate_thres],
#                 trunc,
#             ),
#             options={"maxiter": maxiter},
#             method=minimization_method,
#             bounds=minimization_bounds,
#             **minimization_options,
#         )

#     #--------------------------------------------------------------------------------
#     # get prior parameters alpha_1, beta_1, alpha_2, beta_2, w in "priors" variable
#     #--------------------------------------------------------------------------------
#     priors = p_out.x
#     if np.any(priors < 0) or priors[4] > 1:
#         warnings.warn(
#             f"Calculated priors violate distribution constraints. Alpha and Beta parameters should be >0 and mixture weight should be >=0 and <=1. Current priors: {priors}. Numerical instability likely during processing. Considering using a minimization method that supports bounds."
#         )
#     code_convergence = p_out.message
   
#     #------------------------------------------------------------------------
#     # Calculation of the posterior probability of the null hypothesis p_{H0}
#     #------------------------------------------------------------------------
#     num_cell = len(n11)
#     posterior_probability = []

#     qdb1 = nbinom(n=priors[0], p=priors[1] / (priors[1] + expected)).pmf(n11)
#     qdb2 = nbinom(n=priors[2], p=priors[3] / (priors[3] + expected)).pmf(n11)

#     Qn = priors[4] * qdb1 / (priors[4] * qdb1 + (1 - priors[4]) * qdb2)

#     gd1 = gdtr(relative_risk, priors[0] + n11, priors[1] + expected)
#     gd2 = gdtr(relative_risk, priors[2] + n11, priors[3] + expected)
#     posterior_probability = Qn * gd1 + (1 - Qn) * gd2
    
#     #----------------------------
#     # Calculation of log2(EBGM)
#     #----------------------------
#     dg1 = digamma(priors[0] + n11)
#     dgterm1 = dg1 - np.log(priors[1] + expected)
#     dg2 = digamma(priors[2] + n11)
#     dgterm2 = dg2 - np.log(priors[3] + expected)
#     EBlog2 = (np.log(2) ** -1) * (Qn * dgterm1 + (1 - Qn) * dgterm2)

#     #-----------------------------------------------------------------------
#     # Calculation of the Lower Bound at level 5% using DuMouchel algorithm
#     #-----------------------------------------------------------------------
#     LB05 = quantiles(
#         0.05,
#         Qn,
#         priors[0] + n11,
#         priors[1] + expected,
#         priors[2] + n11,
#         priors[3] + expected,
#         # n_jobs = n_jobs,
#     )

#     #-------------------------------------------------------------------------
#     # Calculation of the Uppoer Bound at level 95% using DuMouchel algorithm
#     #-------------------------------------------------------------------------
#     UB95 = quantiles(
#         0.95,
#         Qn,
#         priors[0] + n11,
#         priors[1] + expected,
#         priors[2] + n11,
#         priors[3] + expected,
#         # n_jobs = n_jobs,
#     )
   
#     #----------------------------------------------------------------
#     # Compute FDR (False Detection Rate), FNR (False Negative Rate)
#     # Compte Se (sensitivity) and Sp (specificity)
#     #----------------------------------------------------------------
    
#     # 1. Compute Null (H0) and Alternative (H1) hypothesis probability
#     #----------------------------------------------------------------
#     p_h0 = np.asarray(posterior_probability)  # posterior_probability represents P(H0), the null hypothesis probability
#     p_h1 = 1.0 - p_h0                         # Alternative hypothesis probability (true signal)
    
#     # 2. Compute total expected masses in the baseline
#     #-------------------------------------------------
#     total_true_signals = np.sum(p_h1)
#     total_true_negatives = np.sum(p_h0)
    
#     # 3. Cumulative sums from left to right (from highest to lowest signal)
#     #-----------------------------------------------------------------------
#     true_positives_cum = np.cumsum(p_h1)
#     false_positives_cum = np.cumsum(p_h0)
#     post_range = np.arange(1, num_cell + 1)
    
    
#     # FDR: proportion of false positives among the k raised alerts
#     #--------------------------------------------------------------
#     FDR = false_positives_cum / post_range
    
#     # Sensitivity (Se): proportion of captured true signals out of the total available
#     #----------------------------------------------------------------------------------
#     Se = true_positives_cum / (total_true_signals + 1e-7)
    
#     # FNR: proportion of missed true signals (those remaining to the right of the threshold)
#     # Strictly equivalent to: FNR = 1.0 - Se
#     #----------------------------------------------------------------------------------------
#     missed_true_signals = total_true_signals - true_positives_cum
#     FNR = missed_true_signals / (total_true_signals + 1e-7)

#     # Specificity (Sp): proportion of correctly identified true negatives
#     # Those are the true negatives that were NOT raised as alerts (remaining to the right)
#     #---------------------------------------------------------------------------------------
#     true_negatives_remaining = total_true_negatives - false_positives_cum
#     Sp = true_negatives_remaining / (total_true_negatives + 1e-7)

#     #-------------------------
#     # return results
#     #------------------------
#     name = DATA["product_name"]
#     ae = DATA["ae_name"]
#     RES = Container(params=True)
#     # list of the parameters used
#     RES.param["input_params"] = input_params
#     RES.param["computation_params"] = computation_params
#     RES.param["convergence"] = code_convergence
#     RES.param["priors"] = priors

#     #--------------------------------------
#     # SIGNALS RESULTS and presentation
#     #--------------------------------------
#     RES.all_signals = pd.DataFrame(
#         {
#             "Product": name,
#             "Adverse Event": ae,
#             "EBGM": np.float64(2**EBlog2),
#             "EB05" : LB05,
#             "EB95" : UB95,
#             "p_{H0}": p_h0,
#             "N_{11}": n11,
#             "N_{10}": n10,
#             "N_{01}": n01,
#             "N_{00}": n00,
#             "FDR": FDR,
#             "FNR": FNR,
#             "Se": Se,
#             "Sp": Sp, 
#         }
#     )
#     RES.all_signals = RES.all_signals.sort_values(by=["EB05"], ascending=False)
    

#     # List of Signals generated according to the method
#     RES.all_signals.index = np.arange(0, len(RES.all_signals.index))

#     # Number of signal according to standar FDA: EBGM >= 2 & LB05 > 1
#     # num_signals = np.sum(LB05 >= np.float64(2))
#     # RES.num_signals = num_signals
#     # RES.signals = RES.all_signals.iloc[0:num_signals,]
#     # 1. Tri par EBGM décroissant 
#     RES.all_signals = RES.all_signals.sort_values(by=["EB05"], ascending=False)
#     RES.all_signals.index = np.arange(0, len(RES.all_signals.index))

    
#     # 2. Application of the FDA dual criterion: EBGM >= 2 AND EB05 (LB05) > 1 or EB05 >= 2
#     # We create a Boolean mask to identify the rows that meet both conditions
#     if  criterion == "EB05>=2" :
#         signal_mask = (RES.all_signals["EB05"] >= np.float64(2)) 
#     elif criterion == "EB05 > 1 & EBGM >=2":
#         signal_mask = (RES.all_signals["EBGM"] >= np.float64(2)) & (RES.all_signals["EB05"] > np.float64(1))
#     # 3. Signal extraction and counting
#     RES.signals = RES.all_signals[signal_mask].copy()
#     RES.num_signals = len(RES.signals)
   

#     return RES


# def non_truncated_likelihood(p, n11, E):
#     dnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).pmf(n11)
#     dnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).pmf(n11)
#     term = (p[4] * dnb1 + (1 - p[4]) * dnb2) + 1e-7
#     return np.sum(-np.log(term))


# def truncated_likelihood(p, n11, E, truncate):
#     dnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).pmf(n11)
#     dnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).pmf(n11)
#     term1 = p[4] * dnb1 + (1 - p[4]) * dnb2 + 1e-7

#     pnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).cdf(truncate)
#     pnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).cdf(truncate)
#     term2 = 1 - (p[4] * pnb1 + (1 - p[4]) * pnb2) + 1e-12

#     return np.sum(-np.log(term1 / term2))

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.special import digamma, gdtr, gammaln, betainc
from scipy.optimize import minimize

from ..utils.Container import Container
from ..utils import calculate_expected
# from ..utils.common import (
#    compute_bayesian_metrics,
#    determine_num_signals,
#    build_params,
#    build_bayesian_result,
#)
# from ..utils.types import DecisionMetric, GPSRankingStatistic, ExpectedMethod
from ..utils.distribution_funcs.quantile_funcs import quantiles

EPS = np.finfo(np.float32).eps
BOUNDED_METHODS = {
    "Nelder-Mead",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "Powell",
    "trust-constr",
    "COBYLA",
    "COBYQA",
}


def _optimize_gps_priors(
    container: Container,
    priors: np.ndarray,
    truncate: bool,
    truncate_thres: float,
    n11: np.ndarray,
    expected: np.ndarray,
    N: int,
    minimization_method: str,
    minimization_bounds: tuple[tuple[float, float], ...] | None,
    minimization_options: dict | None,
) -> tuple[np.ndarray, str]:
    """Estimate empirical Bayes hyperprior parameters via maximum likelihood."""
    if minimization_method not in BOUNDED_METHODS:
        minimization_bounds = None
    elif minimization_bounds is None:
        minimization_bounds = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))

    if minimization_options is None:
        minimization_options = {}

    if not truncate:
        data_cont = container.contingency
        n1__mat = data_cont.sum(axis=1)
        n_1_mat = data_cont.sum(axis=0)
        E_c = (np.outer(n1__mat.values, n_1_mat.values) / N).ravel(order="F")
        n11_c = np.asarray(data_cont.values, dtype=np.float64).ravel(order="F")
        gammaln_n11_1_c = gammaln(n11_c + 1.0)
        p_out = minimize(
            non_truncated_likelihood,
            x0=priors,
            args=(n11_c, E_c, gammaln_n11_1_c),
            options={"maxiter": 500},
            method=minimization_method,
            bounds=minimization_bounds,
            **minimization_options,
        )
    else:
        trunc = truncate_thres - 1
        n11_trunc = n11[n11 >= truncate_thres]
        E_trunc = expected[n11 >= truncate_thres]
        gammaln_n11_1 = gammaln(n11_trunc + 1.0)
        p_out = minimize(
            truncated_likelihood,
            x0=priors,
            args=(
                n11_trunc,
                E_trunc,
                trunc,
                gammaln_n11_1,
            ),
            options={"maxiter": 500},
            method=minimization_method,
            bounds=minimization_bounds,
            **minimization_options,
        )

    priors_opt = p_out.x
    if np.any(priors_opt < 0) or priors_opt[4] > 1:
        warnings.warn(
            f"Calculated priors violate distribution constraints. Alpha and Beta parameters should be >0 and mixture weight should be >=0 and <=1. Current priors: {priors_opt}. Numerical instability likely during processing. Considering using a minimization method that supports bounds."
        )
    return priors_opt, p_out.message


def gps(
    container,
    min_events=4,
    truncate=False,
    maxiter=500,
    criterion= "EB05>=2",
): 
    """Computes signal detection based on Multi-item enabled Gamma Poisson Shrinkage (GPS).

    Clinical Intuition:
        GPS models the true relative risk distribution across all drug-event pairs as a mixture
        of two Gamma distributions: a dominant background null component (capturing non-signals
        centered near RR=1.0) and an elevated risk component. By shrinking observed counts toward
        this empirical Bayesian mixture, GPS stabilizes high-variance small counts (preventing
        spurious alerts from 1 or 2 isolated reports) while producing robust empirical Bayes
        geometric mean (EBGM) estimates and conservative 5th percentile lower bounds (EB05).
        It is the foundational methodology behind the FDA's Empirica Signal / MGPS system.

    Parameters:
        container: A container object holding the input data, including event counts (`events`),
            product-event pairs (`product_aes`), and across-brand counts (`count_across_brands`).
        relative_risk: The threshold for relative risk used in posterior probability calculations.
        min_events: Minimum number of events required for a pair to be retained.
        decision_metric: Decision rule for signal detection ('rank', 'fdr', or 'signals').
        decision_thres: Threshold used in the decision rule to filter significant signals.
        ranking_statistic: Ranking statistic to order results ('log2', 'p_value', or 'quantile').
        truncate: Whether to truncate likelihoods below a threshold for numerical stability.
        truncate_thres: Truncation threshold for likelihood values if truncate is True.
        prior_init: Initial values for the prior distributions (alpha1, beta1, alpha2, beta2, w).
        prior_param: Manually provided prior distribution parameters. If None, estimates priors via ML.
        expected_method: Method for calculating expected counts ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter used in the expected value calculation method.
        minimization_method: Optimization algorithm for estimating prior parameters (default: 'Nelder-Mead').
        minimization_bounds: Bounds on prior parameters during optimization.
        minimization_options: Options passed directly to scipy.optimize.minimize.

    Returns:
        AnalysisResult containing detected signals, all evaluated pairs, signal count,
        and model parameters.
    """
    relative_risk = 1.0
    decision_metric = "rank"
    decision_thres = 0.05
    ranking_statistic = "log2"
    truncate_thres = 1.0
    prior_init = None
    prior_param = None
    expected_method = "mantel-haentzel"
    method_alpha = 1.0
    minimization_method = "SLSQP"
    minimization_bounds = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))
    minimization_options = None

    
    if prior_init is None:
        prior_init = {
            "alpha1": 0.2041,
            "beta1": 0.05816,
            "alpha2": 1.415,
            "beta2": 1.838,
            "w": 0.0969,
        }
    elif isinstance(prior_init, (list, tuple, np.ndarray)):
        prior_init = {
            "alpha1": float(prior_init[0]),
            "beta1": float(prior_init[1]),
            "alpha2": float(prior_init[2]),
            "beta2": float(prior_init[3]),
            "w": float(prior_init[4]),
        }

    input_params = {
        "relative_risk": relative_risk,
        "min_events": min_events,
        "decision_metric": decision_metric,
        "decision_thres": decision_thres,
        "ranking_statistic": ranking_statistic,
        "truncate": truncate,
        "truncate_thres": truncate_thres,
        "expected_method": expected_method,
        "method_alpha": method_alpha,
        "minimization_method": minimization_method,
    }

    computation_params = pd.DataFrame.from_dict({
         "minimization_method": minimization_method,
         "minimization_bounds": minimization_bounds,
         "priori_init": priors,
         "method_alpha": method_alpha,
         "expected_method": expected_method,
         "minimization_options": minimization_options
         }, orient="index", columns=["Value"])


    priors = np.asarray(
        [
            prior_init["alpha1"],
            prior_init["beta1"],
            prior_init["alpha2"],
            prior_init["beta2"],
            prior_init["w"],
        ]
    )
    DATA = container.data
    N = container.N

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)
    code_convergence = "User-provided priors"

    if prior_param is None:
        priors, code_convergence = _optimize_gps_priors(
            container=container,
            priors=priors,
            truncate=truncate,
            truncate_thres=truncate_thres,
            n11=n11,
            expected=expected,
            N=N,
            minimization_method=minimization_method,
            minimization_bounds=minimization_bounds,
            minimization_options=minimization_options,
        )
    else:
        priors = np.asarray(prior_param, dtype=np.float64)

    if min_events > 1:
        mask = n11 >= min_events
        DATA = DATA[mask].reset_index(drop=True)
        expected = expected[mask]
        n1j = n1j[mask]
        ni1 = ni1[mask]
        n11 = n11[mask]

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    num_cell = len(n11)

    # Posterior probability of the null hypothesis
    #------------------------------------------------
    _p_post1 = np.clip(priors[1] / (priors[1] + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    _p_post2 = np.clip(priors[3] / (priors[3] + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    gammaln_n11_1_post = gammaln(n11 + 1.0)
    r1, r2 = priors[0], priors[2]
    log_qdb1 = gammaln(n11 + r1) - gammaln_n11_1_post - gammaln(r1) + r1 * np.log(_p_post1) + n11 * np.log(1.0 - _p_post1)
    log_qdb2 = gammaln(n11 + r2) - gammaln_n11_1_post - gammaln(r2) + r2 * np.log(_p_post2) + n11 * np.log(1.0 - _p_post2)
    qdb1 = np.exp(log_qdb1)
    qdb2 = np.exp(log_qdb2)

    _qn_denom = priors[4] * qdb1 + (1 - priors[4]) * qdb2
    Qn = np.where(_qn_denom > 0, priors[4] * qdb1 / _qn_denom, priors[4])

    gd1 = gdtr(relative_risk, priors[0] + n11, np.maximum(priors[1] + expected, 1e-10))
    gd2 = gdtr(relative_risk, priors[2] + n11, np.maximum(priors[3] + expected, 1e-10))
    posterior_probability = Qn * gd1 + (1 - Qn) * gd2

    dg1 = digamma(priors[0] + n11)
    dgterm1 = dg1 - np.log(np.maximum(priors[1] + expected, 1e-10))
    dg2 = digamma(priors[2] + n11)
    dgterm2 = dg2 - np.log(np.maximum(priors[3] + expected, 1e-10))
    EBlog2 = (np.log(2) ** -1) * (Qn * dgterm1 + (1 - Qn) * dgterm2)
    ebgm = np.power(2.0, np.asarray(EBlog2, dtype=np.float64))

    # Calculation of the Lower Bound (EB05) and Upper Bound (EB95)
    #--------------------------------------------------------------
    EB05 = quantiles(
        0.05,
        Qn,
        priors[0] + n11,
        priors[1] + expected,
        priors[2] + n11,
        priors[3] + expected,
    )
    EB95 = quantiles(
        0.95,
        Qn,
        priors[0] + n11,
        priors[1] + expected,
        priors[2] + n11,
        priors[3] + expected,
    )

    #----------------------------------------------------------------
    # Compute FDR (False Detection Rate), FNR (False Negative Rate)
    # Compte Se (sensitivity) and Sp (specificity)
    #----------------------------------------------------------------
    
    # 1. Compute Null (H0) and Alternative (H1) hypothesis probability
    #----------------------------------------------------------------
    p_h0 = np.asarray(posterior_probability)  # posterior_probability represents P(H0), the null hypothesis probability
    p_h1 = 1.0 - p_h0                         # Alternative hypothesis probability (true signal)
    
    # 2. Compute total expected masses in the baseline
    #-------------------------------------------------
    total_true_signals = np.sum(p_h1)
    total_true_negatives = np.sum(p_h0)
    
    # 3. Cumulative sums from left to right (from highest to lowest signal)
    #-----------------------------------------------------------------------
    true_positives_cum = np.cumsum(p_h1)
    false_positives_cum = np.cumsum(p_h0)
    post_range = np.arange(1, num_cell + 1)
    
    
    # FDR: proportion of false positives among the k raised alerts
    #--------------------------------------------------------------
    FDR = false_positives_cum / post_range
    
    # Sensitivity (Se): proportion of captured true signals out of the total available
    #----------------------------------------------------------------------------------
    Se = true_positives_cum / (total_true_signals + 1e-7)
    
    # FNR: proportion of missed true signals (those remaining to the right of the threshold)
    # Strictly equivalent to: FNR = 1.0 - Se
    #----------------------------------------------------------------------------------------
    missed_true_signals = total_true_signals - true_positives_cum
    FNR = missed_true_signals / (total_true_signals + 1e-7)

    # Specificity (Sp): proportion of correctly identified true negatives
    # Those are the true negatives that were NOT raised as alerts (remaining to the right)
    #---------------------------------------------------------------------------------------
    true_negatives_remaining = total_true_negatives - false_positives_cum
    Sp = true_negatives_remaining / (total_true_negatives + 1e-7)

    #-------------------------
    # return results
    #------------------------
    name = DATA["product_name"]
    ae = DATA["ae_name"]
    RES = Container(params=True)
    # list of the parameters used
    RES.param["input_params"] = input_params
    RES.param["computation_params"] = computation_params
    RES.param["convergence"] = code_convergence
    RES.param["priors"] = priors

    #--------------------------------------
    # SIGNALS RESULTS and presentation
    #--------------------------------------
    RES.all_signals = pd.DataFrame(
        {
            "Product": name,
            "Adverse Event": ae,
            "EBGM": ebgm,
            "EB05" : EB05,
            "EB95" : EB95,
            "p_{H0}": p_h0,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "FDR": FDR,
            "FNR": FNR,
            "Se": Se,
            "Sp": Sp, 
        }
    )
    RES.all_signals = RES.all_signals.sort_values(by=["EB05"], ascending=False)
    

    # List of Signals generated according to the method
    #-------------------------------------------------------
    RES.all_signals.index = np.arange(0, len(RES.all_signals.index))

    # 1. Tri par EBGM décroissant 
    #---------------------------------------------------
    RES.all_signals = RES.all_signals.sort_values(by=["EB05"], ascending=False)
    RES.all_signals.index = np.arange(0, len(RES.all_signals.index))

    
    # 2. Application of the FDA dual criterion: EBGM >= 2 AND EB05 (LB05) > 1 or EB05 >= 2
    # We create a Boolean mask to identify the rows that meet both conditions
    if  criterion == "EB05>=2" :
        signal_mask = (RES.all_signals["EB05"] >= np.float64(2)) 
    elif criterion == "EB05 > 1 & EBGM >=2":
        signal_mask = (RES.all_signals["EBGM"] >= np.float64(2)) & (RES.all_signals["EB05"] > np.float64(1))
    # 3. Signal extraction and counting
    RES.signals = RES.all_signals[signal_mask].copy()
    RES.num_signals = len(RES.signals)
   
    return RES


def non_truncated_likelihood(p, n11, E, gammaln_n11_1=None):
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    p_nb1 = np.clip(p[1] / (p[1] + E + 1e-10), 1e-10, 1.0 - 1e-10)
    p_nb2 = np.clip(p[3] / (p[3] + E + 1e-10), 1e-10, 1.0 - 1e-10)

    r1, r2, w = p[0], p[2], p[4]
    log_dnb1 = gammaln(n11 + r1) - gammaln_n11_1 - gammaln(r1) + r1 * np.log(p_nb1) + n11 * np.log(1.0 - p_nb1)
    log_dnb2 = gammaln(n11 + r2) - gammaln_n11_1 - gammaln(r2) + r2 * np.log(p_nb2) + n11 * np.log(1.0 - p_nb2)
    dnb1 = np.exp(log_dnb1)
    dnb2 = np.exp(log_dnb2)
    term = (w * dnb1 + (1.0 - w) * dnb2) + 1e-7
    return np.sum(-np.log(term))


def truncated_likelihood(p, n11, E, truncate, gammaln_n11_1=None):
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    p_nb1 = np.clip(p[1] / (p[1] + E + 1e-10), 1e-10, 1.0 - 1e-10)
    p_nb2 = np.clip(p[3] / (p[3] + E + 1e-10), 1e-10, 1.0 - 1e-10)

    r1, r2, w = p[0], p[2], p[4]
    log_dnb1 = gammaln(n11 + r1) - gammaln_n11_1 - gammaln(r1) + r1 * np.log(p_nb1) + n11 * np.log(1.0 - p_nb1)
    log_dnb2 = gammaln(n11 + r2) - gammaln_n11_1 - gammaln(r2) + r2 * np.log(p_nb2) + n11 * np.log(1.0 - p_nb2)
    dnb1 = np.exp(log_dnb1)
    dnb2 = np.exp(log_dnb2)
    term1 = w * dnb1 + (1.0 - w) * dnb2

    if truncate == 0:
        pnb1 = np.exp(r1 * np.log(p_nb1))
        pnb2 = np.exp(r2 * np.log(p_nb2))
    else:
        pnb1 = betainc(r1, truncate + 1, p_nb1)
        pnb2 = betainc(r2, truncate + 1, p_nb2)
    term2 = 1.0 - (w * pnb1 + (1.0 - w) * pnb2)

    return np.sum(-np.log(np.maximum(term1, 1e-300)) + np.log(np.maximum(term2, 1e-7)))
