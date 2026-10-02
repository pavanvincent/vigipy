import numpy as np
import pandas as pd
from scipy.stats import norm
from ..utils import Container


def prr(
    container,
    min_events=4,
):
    """
   Calculate the Proportional Reporting Ratio (PRR):
    - Confidence Intervalle [LB, UP] at 95% estimated using Woolf method 
    - Alert Signal accord Evans criterion: (PRR >=2) & (Chi2_yates >=4) & (n11 >= 3)
    
    Arguments:
    
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        min_events: The min number of AE reports to be considered a signal

    Outputs:
    
        product name, adverse event
        N_00, N_01, N_10, N_11,
        PRR, LB, UP,
        chi2_yates,
        is_signal,
        number of signals

    """
    DATA = container.data
    N = container.N

    # discard (product name / adverse event) pairs with event <= min_events
    #----------------------------------------------------------------
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

    
    # compute log(PRR) and VAR(LOG(PRR)) and prr
    #-------------------------------------------------------------
    log_prr = np.log((n11 / (n11 + n10)) / (n01 / (n01 + n00)))
    var_log_prr = 1 / n11 - 1 / (n11 + n10) + 1 / n01 - 1 / (n01 + n00)
    prr = np.exp(log_prr)
    
    # compute lower and upper bound
    #------------------------------------------------------
    log_LB = norm.ppf(0.025, log_prr, np.sqrt(var_log_prr))
    log_UB = norm.ppf(0.975, log_prr, np.sqrt(var_log_prr))

    # exception when log_UB > max_log_value
    #---------------------------------------------
    max_log_value = np.log(np.finfo(np.float64).max)
    ub_exp = np.full_like(log_UB, np.inf)
    safe_mask = log_UB <= max_log_value
    ub_exp[safe_mask] = np.exp(log_UB[safe_mask])

    # compute p-value
    #--------------------------------------
    prr_H0 = 1
    pval_uni = 1 - norm.cdf(log_prr, np.log(prr_H0), np.sqrt(var_log_prr))
    pval_uni[pval_uni > 1] = 1
    pval_uni[pval_uni < 0] = 0

    chi2_yates = (
        N * (np.maximum(0, np.abs(n11 * n00 - n10 * n01) - N / 2) ** 2)
    ) / ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))

    # Define signal criteria mask: N_11 >= 3 AND Chi2 >= 4 AND PRR >= 2
    #-------------------------------------------------------------------
    signal_mask = (n11 >= 3) & (chi2_yates >= 4) & (prr >= 2)
    num_signals = signal_mask.sum()

   
    # count number of signals, unsin criterion decison (log_LB > 0
    # store results
    #----------------------------------------------------------------
    RC = Container()
    RC.all_signals = pd.DataFrame(
        {
            "Product": DATA["product_name"].values,
            "Adverse Event": DATA["ae_name"].values,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "PRR": prr,
            "LB(CI 95%)" : np.exp(log_LB),
            "UB(CI 95%)" : ub_exp,
            "Chi2_Yates": chi2_yates,
            "Is_Signal": signal_mask,
        },
        index=np.arange(len(n11)),
    ).sort_values(by=["PRR","Chi2_Yates"], ascending=False)

    RC.signals = RC.all_signals.iloc[
        0:num_signals,
    ]
    RC.num_signals = num_signals
    return RC
