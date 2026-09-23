import pandas as pd
import numpy as np
import warnings
from scipy.special import gdtr
from scipy.stats import nbinom
from scipy.optimize import minimize
from sympy.functions.special import gamma_functions

from ..utils import Container
from ..utils import calculate_expected
from ..utils.distribution_funcs.negative_binomials import dnbinom, pnbinom
from ..utils.distribution_funcs.quantile_funcs import quantiles

dnbinom = np.vectorize(dnbinom)
pnbinom = np.vectorize(pnbinom)
digamma = np.vectorize(gamma_functions.digamma)
quantiles = np.vectorize(quantiles)

EPS = np.finfo(np.float64).eps
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


def gps(
    container,
    # relative_risk=1,
    min_events=1,
    # decision_metric="rank",
    # decision_thres=0.05,
    # ranking_statistic="quantile",
    truncate=False,
    # truncate_thres=1,
    # prior_param=None,
    # expected_method="mantel-haentzel",
    # method_alpha=1,
    # minimization_method="SLSQP",
    # minimization_bounds=((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1)),
    # minimization_options=None,
):
    """
    Computes signal detection based on Multi-item enabled Gamma Poisson Shrinkage (GPS) using prior distributions
    for adverse event and product feature data.

    Parameters:
    -----------
    container : object
        A container object holding the input data, including event counts (`events`),
        product-event pairs (`product_aes`), and across-brand counts (`count_across_brands`).
    relative_risk : float, optional (default=1)
        The threshold for relative risk used in the posterior probability calculations.
    min_events : int, optional (default=1)
        The minimum number of events required for an adverse event to be considered in the analysis.
    decision_metric : str, optional (default="rank")
        The decision rule for signal detection. Options are 'rank', 'fdr', or 'signals'.
    decision_thres : float, optional (default=0.05)
        The threshold used in the decision rule to filter significant signals.
    ranking_statistic : str, optional (default="log2")
        The ranking statistic used to order the results. Options include 'log2', 'p_value', or 'quantile'.
    truncate : bool, optional (default=False)
        Whether to truncate likelihoods below a certain threshold for stability in signal detection.
    truncate_thres : float, optional (default=1)
        The truncation threshold for likelihood values if `truncate` is set to True.
    prior_init : dict, optional
        Initial values for the prior distributions used in Bayesian inference. Contains parameters for two Poisson
        distributions (alpha1, beta1, alpha2, beta2) and the mixture weight (w).
    prior_param : array, optional (default=None)
        Manually provided prior distribution parameters. If None, the function estimates priors using optimization.
    expected_method : str, optional (default="mantel-haentzel")
        The method used to calculate the expected event counts. Options include "mantel-haentzel", "negative-binomial" and "poisson".
    method_alpha : float, optional (default=1)
        Dispersion parameter used in the expected value calculation method.
    minimization_method : str, optional (default="SLSQP")
        The optimization method used for estimating prior parameters if `prior_param` is None.
    minimization_bounds : tuple, optional
        Bounds on the prior parameter values for the optimization process.
    minimization_options : dict, optional
        Options for the minimization routine.

    Returns:
    --------
    RES : object
        A container object with the following attributes:
        - `param`: A dictionary of input parameters, including prior initialization and optimization results.
        - `all_signals`: A DataFrame containing detailed results of signal detection, including posterior probabilities,
          expected counts, and ranking statistics.
        - `signals`: A DataFrame of filtered signals according to the decision metric and threshold.
        - `num_signals`: The number of signals detected based on the decision rule.

    Notes:
    ------
    - This function implements a Bayesian model to calculate posterior probabilities using a mixture of two negative
      binomial distributions.
    - The function can apply different ranking statistics to order results, such as p-value, quantile, or log2.
    - The optimization process is used to estimate the prior parameters unless provided manually.
    - The function can handle truncation for numerical stability when dealing with sparse data.
    """
   
    
    input_params = locals()
    del input_params["container"]

    relative_risk=1,
    decision_metric="rank"
    decision_thres=0.05
    truncate_thres=1
    prior_param=None
    expected_method="mantel-haentzel"
    method_alpha=1
    minimization_method="SLSQP"
    minimization_bounds=((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))
    minimization_options=None

    

    alpha1 =0.5*(EPS + 20) 
    beta1 =0.5* (EPS + 10)
    alpha2 = 0.5*(EPS + 20)
    beta2 = 0.5*(EPS +10)
    w = 0.5*(0 + 1)
    priors = np.asarray([alpha1, beta1, alpha2, beta2, w])
    input_params["prior_init"] = priors
    DATA = container.data
    N = container.N

    #-------------------------------------------------
    # Compute expected values using expected_method
    #------------------------------------------------
    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)
    p_out = True

    #----------------------------------------------------------------------------------------
    # Launch optimization algorithm to find hypergeometrical parameters of the prior
    # priori is sum of two independant Gamma laws, whose parameters are 
    # alpha_1, beta_1, alpha_2, beta_2, w
    # such that prior density is : w * Gamma(alpha_1, beta_1) + (1-w) * Gamma(alpha_2,beta_2)
    # optimization algorithm finds alpha_1, beta_1, alpha_2, beta_2, w by minimizing
    # 1) either non truncated likelihood (imput argument truncated = False)
    # 2) either truncated objective likelihood (imput argument truncated = true
    #-----------------------------------------------------------------------------------------
    if prior_param is None:
        p_out = False
        if minimization_method not in BOUNDED_METHODS:
            minimization_bounds = None

        if minimization_options is None:
            minimization_options = {}

        if not truncate:
            data_cont = container.contingency
            n1__mat = data_cont.sum(axis=1)
            n_1_mat = data_cont.sum(axis=0)
            rep = len(n_1_mat)
            n1__c = np.tile(n1__mat.values, reps=rep)
            rep = len(n1__mat)
            n_1_c = np.repeat(n_1_mat.values, repeats=rep)
            E_c = np.asarray(n1__c, dtype=np.float64) * n_1_c / N
            n11_c_temp = []
            for col in data_cont:
                n11_c_temp.extend(list(data_cont[col]))
            n11_c = np.asarray(n11_c_temp)

            p_out = minimize(
                non_truncated_likelihood,
                x0=priors,
                args=(n11_c, E_c),
                options={"maxiter": 500},
                method=minimization_method,
                bounds=minimization_bounds,
                **minimization_options,
            )
        elif truncate:
            trunc = truncate_thres - 1
            p_out = minimize(
                truncated_likelihood,
                x0=priors,
                args=(
                    n11[n11 >= truncate_thres],
                    expected[n11 >= truncate_thres],
                    trunc,
                ),
                options={"maxiter": 500},
                method=minimization_method,
                bounds=minimization_bounds,
                **minimization_options,
            )

        #--------------------------------------------------------------------------------
        # get prior parameters alpha_1, beta_1, alpha_2, beta_2, w in "priors" variable
        #--------------------------------------------------------------------------------
        priors = p_out.x
        if np.any(priors < 0) or priors[4] > 1:
            warnings.warn(
                f"Calculated priors violate distribution constraints. Alpha and Beta parameters should be >0 and mixture weight should be >=0 and <=1. Current priors: {priors}. Numerical instability likely during processing. Considering using a minimization method that supports bounds."
            )
        code_convergence = p_out.message

    #--------------------------------------------------------
    # exclude product / ae pairs with low numbers of events
    #--------------------------------------------------------
    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]
        expected = expected[n11 >= min_events]
        n1j = n1j[n11 >= min_events]
        ni1 = ni1[n11 >= min_events]
        n11 = n11[n11 >= min_events]

    #------------------------------------------------------------------------
    # Calculation of the posterior probability of the null hypothesis p_{H0}
    #------------------------------------------------------------------------
    num_cell = len(n11)
    posterior_probability = []

    qdb1 = nbinom(n=priors[0], p=priors[1] / (priors[1] + expected)).pmf(n11)
    qdb2 = nbinom(n=priors[2], p=priors[3] / (priors[3] + expected)).pmf(n11)

    Qn = priors[4] * qdb1 / (priors[4] * qdb1 + (1 - priors[4]) * qdb2)

    gd1 = gdtr(relative_risk, priors[0] + n11, priors[1] + expected)
    gd2 = gdtr(relative_risk, priors[2] + n11, priors[3] + expected)
    posterior_probability = Qn * gd1 + (1 - Qn) * gd2
    
    #----------------------------
    # Calculation of log2(EBGM)
    #----------------------------
    dg1 = digamma(priors[0] + n11)
    dgterm1 = dg1 - np.log(priors[1] + expected)
    dg2 = digamma(priors[2] + n11)
    dgterm2 = dg2 - np.log(priors[3] + expected)
    EBlog2 = (np.log(2) ** -1) * (Qn * dgterm1 + (1 - Qn) * dgterm2)

    #-----------------------------------------------------------------------
    # Calculation of the Lower Bound at level 5% using DuMouchel algorithm
    #-----------------------------------------------------------------------
    LB05 = quantiles(
        0.05,
        Qn,
        priors[0] + n11,
        priors[1] + expected,
        priors[2] + n11,
        priors[3] + expected,
    )

    #-------------------------------------------------------------------------
    # Calculation of the Uppoer Bound at level 95% using DuMouchel algorithm
    #-------------------------------------------------------------------------
    UB95 = quantiles(
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
    RES.param["prior_param"] = priors
    RES.param["convergence"] = code_convergence

    #--------------------------------------
    # SIGNALS RESULTS and presentation
    #--------------------------------------
    RES.all_signals = pd.DataFrame(
        {
            "Product": name,
            "Adverse Event": ae,
            "EBGM": np.float64(2**EBlog2),
            "LB05 FDA" : LB05,
            "UB95 FDA" : UB95,
            "p_{H0}": p_h0,
            "N_{11}": n11,
            "product margin": n1j,
            "event margin": ni1,
            "FDR": FDR,
            "FNR": FNR,
            "Se": Se,
            "Sp": Sp, 
        }
    )
    RES.all_signals = RES.all_signals.sort_values(by=["LB05 FDA"], ascending=False)
    

    # List of Signals generated according to the method
    RES.all_signals.index = np.arange(0, len(RES.all_signals.index))

    # Number of signal according to standar FDA: LB05 >= 2
    num_signals = np.sum(LB05 >= np.float64(2))
    RES.num_signals = num_signals
    RES.signals = RES.all_signals.iloc[0:num_signals,]
   

    return RES


def non_truncated_likelihood(p, n11, E):
    dnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).pmf(n11)
    dnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).pmf(n11)
    term = (p[4] * dnb1 + (1 - p[4]) * dnb2) + 1e-7
    return np.sum(-np.log(term))


def truncated_likelihood(p, n11, E, truncate):
    dnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).pmf(n11)
    dnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).pmf(n11)
    term1 = p[4] * dnb1 + (1 - p[4]) * dnb2

    pnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).cdf(truncate)
    pnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).cdf(truncate)
    term2 = 1 - (p[4] * pnb1 + (1 - p[4]) * pnb2)

    return np.sum(-np.log(term1 / term2))
