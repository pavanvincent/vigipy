import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.lbe import lbe
from ..utils import Container
from ..utils import calculate_expected


def prr(
    container,
    min_events=1,
):
    """
   Calculate the Proportional Reporting Ratio (PRR):
    - Confidence Intervalle [LB, UP] at 95% estimated using Woolf method 
    - Alert Signal if log(LB) > 0
    - p-value computed using Wald unilateral test
    
    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        min_events: The min number of AE reports to be considered a signal

    Outputs:
        product name, adverse event
        N_00, N_01, N_10, N_11,
        PRR, LB, UP,
        p-value,
        number of signals

    """
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    num_cell = len(n11)
   
    n10 = n1j - n11
    n01 = ni1 - n11 + 1e-7
    n00 = N - (n11 + n10 + n01)

    log_prr = np.log((n11 / (n11 + n10)) / (n01 / (n01 + n00)))
    var_log_prr = 1 / n11 - 1 / (n11 + n10) + 1 / n01 - 1 / (n01 + n00)

    prr_H0 = 1
    pval_uni = 1 - norm.cdf(log_prr, np.log(prr_H0), np.sqrt(var_log_prr))
    pval_uni[pval_uni > 1] = 1
    pval_uni[pval_uni < 0] = 0

   
    log_LB = norm.ppf(0.025, log_prr, np.sqrt(var_log_prr))
    log_UB = norm.ppf(0.975, log_prr, np.sqrt(var_log_prr))
            
    num_signals = (log_LB > 0).sum()
    RC = Container()
    RC.all_signals = pd.DataFrame(
        {
            "Product": DATA["product_name"].values,
            "Adverse Event": DATA["ae_name"].values,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "PRR": np.exp(log_prr),
            "LB(CI 95%)" : np.exp(log_LB),
            "UP(CI 95%)" : np.exp(log_UB),
            "p-value" : pval_uni,
        },
        index=np.arange(len(n11)),
    ).sort_values(by=["LB(CI 95%)"], ascending = False)

    RC.signals = RC.all_signals.iloc[
        0:num_signals,
    ]
    RC.num_signals = num_signals
    return RC
