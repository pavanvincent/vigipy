import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from ..utils.lbe import lbe
from ..utils import Container
from ..utils import calculate_expected


def ror(
    container,
    min_events=1,
):
    """
    Calculate the reporting odd ratio (ROR)
    - Confidence Intervalle of 95% estimated using Woolf method 
    - Signal alert if log(LB) > 0
    - p-value computed using Wald unilateral test
    
    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        min_events: The min number of AE reports to be considered a signal

    """
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    num_cell = len(n11)
    #---------------------------
    n10 = n1j - n11
    n01 = ni1 - n11 + 1e-7
    n00 = N - (n11 + n10 + n01)
    #---------------------------------------------------------
    # Computing ROR and ROR Variance
    log_ror = np.log(n11 * n00 / (n10 * n01))
    var_log_ror = 1.0 / n11 + 1.0 / n10 + 1.0 / n01 + 1.0 / n00
    #----------------------------------------------------------
    # Computing Confidence Interval from Woolf Method
    LB = norm.ppf(0.025, log_ror, np.sqrt(var_log_ror))
    UB = norm.ppf(0.975,log_ror, np.sqrt(var_log_ror))
    #----------------------------------------------------
    # computing p-value using Wald unilateral Wald test
    ror_H0 = 1
    pval_uni = 1 - norm.cdf(log_ror, np.log(ror_H0), np.sqrt(var_log_ror))
    #--------------------------------------------
    # correcting p_value in case of necessity
    pval_uni[pval_uni > 1] = 1
    pval_uni[pval_uni < 0] = 0
    # computing number of signals
    num_signals = (LB > 0).sum()

    RC = Container()
    RC.all_signals = pd.DataFrame(
        {
            "Product": DATA["product_name"].values,
            "Adverse Event": DATA["ae_name"].values,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "ROR": np.exp(log_ror),
            "LB(CI 95%)" : np.exp(LB),
            "UP(CI 95%)" : np.exp(UB),
            "p-value" : pval_uni,
        },
        index=np.arange(len(n11)),
    ).sort_values(by=["LB(CI 95%)"], ascending = False)

    RC.signals = RC.all_signals.iloc[
        0:num_signals,
    ]
    RC.num_signals = num_signals
    return RC
