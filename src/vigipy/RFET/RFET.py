import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom, norm
from ..utils import Container


def rfet(
    container,
    min_events=4,
    mid_pval=False,
):
    """
    Calculate the reporting odds ratio (ROR):
    - Confidence Intervalle [LB, UP] at 95% estimated using Woolf method 
    - Alert Signal if log(LB) > 0
    - p-value computed using exact Fischer test
    
    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        min_events: The min number of AE reports to be considered a signal

    Outputs:
        product name, adverse event
        N_00, N_01, N_10, N_11,
        ROR, LB, UP,
        p-value,
        number of signals

    """
    DATA = container.data
    N = container.N

    # discard (product name / adverse event) pairs with event <= min_events
    #----------------------------------------------------------------------
    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]

    # get contingency values
    #------------------------------------------------------
    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    num_cell = len(n11)

    n10 = n1j - n11
    n01 = ni1 - n11 + 1e-7
    n00 = N - (n11 + n10 + n01)

    # compute log (ROR) and VAR(log(ROR)) using Woolf method
    #----------------------------------------------------------
    log_rfet = np.log(n11 * n00 / (n10 * n01))
    var_log_rfet = 1.0 / n11 + 1.0 / n10 + 1.0 / n01 + 1.0 / n00

    # Get credibility interval at 95%
    #--------------------------------------------------------
    log_LB = norm.ppf(0.025, log_rfet, np.sqrt(var_log_rfet))
    log_UB = norm.ppf(0.975,log_rfet, np.sqrt(var_log_rfet))

    # exception when log_UB > max_log_value
    #---------------------------------------------
    max_log_value = np.log(np.finfo(np.float64).max)
    ub_exp = np.full_like(log_UB, np.inf)
    safe_mask = log_UB <= max_log_value
    ub_exp[safe_mask] = np.exp(log_UB[safe_mask])
    
    # Hypergeometric distribution parameters (all in integer/array format)
    # M: Total number N
    # n: Row total: n11 + n10 = n1j
    # N_succ: Column total: n11 + n01 = ni1
    #------------------------------------------
    M = N.astype(np.int64)
    n = n1j.astype(np.int64)
    N_succ = ni1.astype(np.int64)
    
    # Greater one-sided p-value: P(X >= n11)
    # In terms of CDF/SF: P(X >= x) = P(X > x - 1) = sf(x - 1)
    # We force n11 to an integer for hypergeom precision
    #----------------------------------------------------------
    k = n11.astype(np.int64)
    pval_uni = hypergeom.sf(k - 1, M, n, N_succ)
    
    # Applying the mid-p-value in a purely vectorial way
    #---------------------------------------------------
    if mid_pval:
        pmf_val = hypergeom.pmf(k, M, n, N_succ)
        pval_uni = pval_uni - 0.5 * pmf_val

    # Efficient replacement of out-of-bounds values ​​[0, 1]
    #-----------------------------------------------------
    np.clip(pval_uni, 0, 1, out=pval_uni)

    # count number of signal using decision criterion log_LB > 0
    #-------------------------------------------------------------
    num_signals = (log_LB > 0).sum()

    # Return results
    #------------------------
    RC = Container()
    RC.all_signals = pd.DataFrame(
        {
            "Product": DATA["product_name"].values,
            "Adverse Event": DATA["ae_name"].values,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "ROR": np.exp(log_rfet),
            "LB(CI 95%)" : np.exp(log_LB),
            "UB(CI 95%)" : ub_exp,
            "p-value" : pval_uni,
        },
        index=np.arange(len(n11)),
    ).sort_values(by=["LB(CI 95%)"], ascending = False)

    RC.signals = RC.all_signals.iloc[
        0:num_signals,
    ]
    RC.num_signals = num_signals
    return RC
